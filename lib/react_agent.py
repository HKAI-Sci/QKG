"""Reusable ReAct (Reasoning + Acting) agent loop.

Domain-agnostic: callers supply tools, an LLM function, and a system prompt.
"""

import re
import json
import inspect
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Message:
    """Lightweight chat message (no dependency on f1)."""
    role: str       # "system" | "user" | "assistant"
    content: str


@dataclass
class ReactResult:
    """Result of a ReAct agent run."""
    final_answer: dict | None
    conversation: list[dict]   # [{"role": ..., "content": ..., "turn": N}, ...]
    tool_calls: list[dict]     # [{"turn": N, "action": ..., "observation_length": N}, ...]
    num_turns: int
    compression_log: list[dict] | None = None  # [{"original_chars": N, "compressed_chars": N, "summary": "..."}]


COMPRESSION_MIN_CHARS = 300    # skip compression for short observations

COMPRESSION_PROMPT = """You are summarizing a KG tool observation for a medical QA agent.
The agent has already read this observation and produced its reasoning.
Compress the observation to only the key facts the agent may need later.

Rules:
- For entity relations: "{entity} has {N} {relation_type} relations. Key findings: [list top relevant ones]. {M} removed as NOT Applicable." Keep any _patient_relevance annotations.
- For empty results: state that no results were found
- Maximum 200 words
- Output plain text, not JSON"""



# ============================================================
# System prompts — base (shared) + task-specific answer blocks
# ============================================================

# ------------------------------------------------------------------
# Base prompts: everything except the FINAL_ANSWER format block.
# Task scripts append their own answer instruction to these bases.
# ------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are a medical expert answering a multiple-choice question.
You have access to PrimeKG, a biomedical knowledge graph with drug, disease, gene/protein, phenotype, and anatomy entities.

Available tools (use ACTION: to invoke):
- list_relation_types() — list all KG relation types grouped by category. CALL THIS FIRST to learn valid relation_type values.
- search_entity(query, type=None, source=None, limit=10) — find entities by name (case-insensitive substring match)
- get_entity_relations(entity_index, relation_type=None, limit=20) — get KG edges for an entity by its index
- check_relation(entity_a_index, entity_b_index) — check all relations between two specific entities

CRITICAL: The relation_type parameter in get_entity_relations must be an exact PrimeKG relation name.
For example, drug side effects use "drug_effect" (NOT "side_effect"). Always call list_relation_types() first
to see the valid names before filtering by relation_type.
If get_entity_relations returns an empty list, double check whether your relation_type matches the schema
by calling list_relation_types() — you may be using a wrong relation name.

Usage format (one tool call per turn):
  ACTION: list_relation_types()
  ACTION: search_entity("aspirin", type="drug")
  ACTION: get_entity_relations(12345, relation_type="drug_effect")
  ACTION: check_relation(12345, 67890)

After each ACTION, you will receive an OBSERVATION with the results.

IMPORTANT PROCEDURE — you MUST follow these steps:
1. Call list_relation_types() to learn the available relation types
2. Briefly analyze the question to identify key medical entities (drugs, diseases, symptoms, etc.)
3. Use search_entity to look up at least 2-3 key entities from the question or answer options
4. Use get_entity_relations to explore relationships relevant to the question
5. After gathering KG evidence, synthesize your medical knowledge WITH the KG findings
6. Only then provide your FINAL_ANSWER

You MUST make at least 2 tool calls before answering. Do NOT skip the KG lookup step.
Query KG symmetrically for all relevant options when applicable.

Output exactly one ACTION per turn, OR a FINAL_ANSWER (never both)."""


SYSTEM_PROMPT_PATIENT_CONTEXT_BASE = """You are a medical expert answering a multiple-choice question.
You have access to PrimeKG, a biomedical knowledge graph with drug, disease, gene/protein, phenotype, and anatomy entities.

Available tools (use ACTION: to invoke):
- list_relation_types() — list all KG relation types grouped by category. CALL THIS FIRST to learn valid relation_type values.
- search_entity(query, type=None, source=None, limit=10) — find entities by name (case-insensitive substring match)
- get_entity_relations_with_context(entity_index, relation_type=None, limit=20) — get KG edges for an entity, with patient-relevance annotations
- check_relation_with_context(entity_a_index, entity_b_index) — check all relations between two specific entities, with patient-relevance annotations

CRITICAL: The relation_type parameter in get_entity_relations_with_context must be an exact PrimeKG relation name.
For example, drug side effects use "drug_effect" (NOT "side_effect"). Always call list_relation_types() first
to see the valid names before filtering by relation_type.
If get_entity_relations_with_context returns an empty list, double check whether your relation_type matches the schema
by calling list_relation_types() — you may be using a wrong relation name.

Usage format (one tool call per turn):
  ACTION: list_relation_types()
  ACTION: search_entity("aspirin", type="drug")
  ACTION: get_entity_relations_with_context(12345, relation_type="drug_effect")
  ACTION: check_relation_with_context(12345, 67890)

After each ACTION, you will receive an OBSERVATION with the results.

Some relations returned by get_entity_relations_with_context include patient-relevance annotations:
- _patient_relevance: "Definitely Applicable" or "Increased Likelihood" — this relation is
  supported by evidence for the patient in the question. Give it HIGH weight in your reasoning.
Relations without this annotation are unenriched — use them normally.
Relations that are NOT applicable to this patient have already been removed.

IMPORTANT PROCEDURE — you MUST follow these steps:
1. Call list_relation_types() to learn the available relation types
2. Briefly analyze the question to identify key medical entities (drugs, diseases, symptoms, etc.)
3. Use search_entity to look up at least 2-3 key entities from the question or answer options
4. Use get_entity_relations_with_context to explore relationships relevant to the question
5. After gathering KG evidence, synthesize your medical knowledge WITH the KG findings
6. Only then provide your FINAL_ANSWER

You MUST make at least 2 tool calls before answering. Do NOT skip the KG lookup step.
Query KG symmetrically for all relevant options when applicable.

Output exactly one ACTION per turn, OR a FINAL_ANSWER (never both)."""


# ------------------------------------------------------------------
# MCQ answer instructions: FINAL_ANSWER format for multiple-choice QA
# ------------------------------------------------------------------

MCQ_ANSWER_INSTRUCTION = """
When you are done reasoning, output your answer in exactly this format:
FINAL_ANSWER: {"llm_answer_choice": "X", "selected_option_text": "...", "reasoning": "..."}

Rules:
- llm_answer_choice must be a single capital letter (A-J)
- selected_option_text must exactly match the chosen option text"""

MCQ_ANSWER_INSTRUCTION_PATIENT_CONTEXT = MCQ_ANSWER_INSTRUCTION  # same format


# ------------------------------------------------------------------
# Composed prompts: backward-compatible with conditionKgTestAgentic.py
# ------------------------------------------------------------------

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + "\n" + MCQ_ANSWER_INSTRUCTION

SYSTEM_PROMPT_PATIENT_CONTEXT = SYSTEM_PROMPT_PATIENT_CONTEXT_BASE + "\n" + MCQ_ANSWER_INSTRUCTION_PATIENT_CONTEXT


class ReactAgent:
    """ReAct agent that alternates reasoning and tool use until a final answer."""

    def __init__(
        self,
        tools: dict[str, Callable],
        llm_call: Callable,          # async (list[Message]) -> str
        system_prompt: str,
        max_turns: int = 10,
        max_result_chars: int = 8000,
        result_hook: Callable | None = None,  # async (result, tool_name, patient_context) -> result
        patient_context: str | None = None,
        memory_compression: bool = False,
        compression_llm_call: Callable | None = None,  # async (list[Message]) -> str
    ):
        self.tools = tools
        self.llm_call = llm_call
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_result_chars = max_result_chars
        self.result_hook = result_hook
        self.patient_context = patient_context
        self.memory_compression = memory_compression
        self.compression_llm_call = compression_llm_call

    # ------------------------------------------------------------------
    # Memory compression
    # ------------------------------------------------------------------

    async def _compress_observation(self, observation: str, tool_action: str, question: str) -> str:
        """Compress a tool observation into a compact summary via LLM."""
        llm = self.compression_llm_call or self.llm_call
        msgs = [
            Message(role="system", content=COMPRESSION_PROMPT),
            Message(role="user", content=(
                f"Question context: {question[:500]}\n\n"
                f"Tool call: {tool_action}\n\n"
                f"Observation:\n{observation}"
            )),
        ]
        summary = await llm(msgs)
        return f"[SUMMARY] {summary}"

    async def _compress_old_in_messages(self, messages: list[Message],
                                       log: list[dict] | None = None,
                                       conversation: list[dict] | None = None) -> None:
        """Compress the oldest uncompressed observation that exceeds the min size.

        Skips the latest observation (last user message) so the agent always
        has full access to the most recent tool result.
        """
        # Walk backwards from second-to-last message, skip latest assistant+observation pair
        for i in range(len(messages) - 3, 1, -1):
            msg = messages[i]
            if msg.role == "user" and msg.content.startswith("OBSERVATION:"):
                if msg.content.startswith("OBSERVATION:\n[SUMMARY]"):
                    break  # already compressed, nothing to do
                if len(msg.content) < COMPRESSION_MIN_CHARS:
                    break  # short enough, skip
                # Found a long uncompressed observation — compress it
                original_chars = len(msg.content)
                tool_action = messages[i - 1].content  # assistant message with ACTION
                compressed = await self._compress_observation(
                    msg.content, tool_action, messages[1].content  # messages[1] = user question
                )
                compressed_msg = f"OBSERVATION:\n{compressed}"
                messages[i] = Message(role="user", content=compressed_msg)
                if log is not None:
                    log.append({
                        "original_chars": original_chars,
                        "compressed_chars": len(compressed_msg),
                        "summary": compressed[:300],
                    })
                # Tag the matching conversation entry with compressed content
                if conversation is not None:
                    # Find the observation entry whose content matches the original
                    obs_text = msg.content[len("OBSERVATION:\n"):]
                    for conv_entry in conversation:
                        if (conv_entry["role"] == "observation"
                                and conv_entry["content"] == obs_text
                                and "compressed" not in conv_entry):
                            conv_entry["compressed"] = compressed
                            break
                break

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _format_result(self, result: Any) -> str:
        result_str = json.dumps(result, ensure_ascii=False, indent=1)
        if len(result_str) > self.max_result_chars:
            result_str = result_str[:self.max_result_chars] + "\n... (truncated)"
        return result_str

    async def _execute_action(self, action_str: str) -> str:
        """Parse ACTION: tool_name(args) and execute."""
        m = re.match(r'(\w+)\((.*)\)', action_str.strip(), re.DOTALL)
        if not m:
            return f"Error: Could not parse tool call: {action_str}"

        tool_name = m.group(1)
        args_str = m.group(2).strip()

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"

        try:
            kwargs = {}
            positional = []

            if args_str:
                eval_code = f"__args_wrapper__({args_str})"
                captured = {}

                def args_wrapper(*args, **kw):
                    captured["args"] = args
                    captured["kwargs"] = kw

                eval(eval_code, {"__builtins__": {}, "__args_wrapper__": args_wrapper})
                positional = list(captured.get("args", []))
                kwargs = captured.get("kwargs", {})

            func = self.tools[tool_name]
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            for i, val in enumerate(positional):
                if i < len(params):
                    kwargs[params[i]] = val

            result = func(**kwargs)
            if self.result_hook:
                result = await self.result_hook(result, tool_name, self.patient_context)
            return self._format_result(result)

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def _execute_by_name(self, tool_name: str, kwargs: dict) -> str:
        """Execute a tool by name with keyword arguments."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
        try:
            func = self.tools[tool_name]
            result = func(**kwargs)
            if self.result_hook:
                result = await self.result_hook(result, tool_name, self.patient_context)
            return self._format_result(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_xml_calls(response: str) -> list[dict]:
        """Parse XML-format tool calls from LLM response."""
        calls = []
        for invoke_match in re.finditer(r'<invoke\s+name="(\w+)">(.*?)</invoke>', response, re.DOTALL):
            tool_name = invoke_match.group(1)
            params_block = invoke_match.group(2)

            kwargs = {}
            for param_match in re.finditer(r'<parameter\s+name="(\w+)">(.*?)</parameter>', params_block, re.DOTALL):
                key = param_match.group(1)
                val_str = param_match.group(2).strip()

                if val_str.lower() in ("none", "null"):
                    val = None
                elif val_str.isdigit():
                    val = int(val_str)
                else:
                    try:
                        val = int(val_str)
                    except ValueError:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str

                if val is not None:
                    kwargs[key] = val

            calls.append({"tool_name": tool_name, "kwargs": kwargs})

        return calls

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract a JSON object from text using bracket balancing."""
        text = text.strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        return None
        return None

    @staticmethod
    def _extract_json_from_response(response: str) -> dict | None:
        """Try to find a JSON object with llm_answer_choice in response."""
        if "llm_answer_choice" not in response:
            return None
        return ReactAgent._extract_json(response[response.find("{"):])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, user_message: str) -> ReactResult:
        """Run the ReAct loop. Returns ReactResult with conversation trace and final answer."""
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=user_message),
        ]

        conversation = []
        tool_calls = []
        compression_log = [] if self.memory_compression else None
        final_answer = None

        for turn in range(self.max_turns):
            response = await self.llm_call(messages)
            conversation.append({"role": "assistant", "content": response, "turn": turn})

            action_match = re.search(r'ACTION:\s*(.+)', response)
            xml_calls = self._parse_xml_calls(response)

            if action_match:
                action_str = action_match.group(1).strip()
                tool_calls.append({"turn": turn, "action": action_str})

                observation = await self._execute_action(action_str)
                tool_calls[-1]["observation_length"] = len(observation)

                action_pos = response.find("ACTION:")
                clean_response = response[:action_pos].rstrip() + f"\nACTION: {action_str}"

                messages.append(Message(role="assistant", content=clean_response))
                messages.append(Message(role="user", content=f"OBSERVATION:\n{observation}"))
                conversation.append({"role": "observation", "content": observation, "turn": turn})

                if self.memory_compression and len(tool_calls) >= 2:
                    await self._compress_old_in_messages(messages, compression_log, conversation)

            elif xml_calls:
                trunc_pos = response.find("<function_calls>")
                if trunc_pos == -1:
                    trunc_pos = response.find("<invoke")
                clean_response = response[:trunc_pos].rstrip() if trunc_pos > 0 else response

                call = xml_calls[0]
                action_desc = f"{call['tool_name']}({call['kwargs']})"
                tool_calls.append({"turn": turn, "action": action_desc})

                obs = await self._execute_by_name(call["tool_name"], call["kwargs"])
                tool_calls[-1]["observation_length"] = len(obs)

                remaining_info = ""
                if len(xml_calls) > 1:
                    remaining_names = [c["tool_name"] for c in xml_calls[1:]]
                    remaining_info = (
                        f"\n\n(You planned {len(xml_calls)} tool calls but only one is "
                        f"executed per turn. Continue with the remaining calls: {remaining_names})"
                    )

                messages.append(Message(role="assistant", content=clean_response + f"\nACTION: {action_desc}"))
                messages.append(Message(role="user", content=f"OBSERVATION:\n[{call['tool_name']}] {obs}{remaining_info}"))
                conversation.append({"role": "observation", "content": f"[{call['tool_name']}] {obs}", "turn": turn})

                if self.memory_compression and len(tool_calls) >= 2:
                    await self._compress_old_in_messages(messages, compression_log, conversation)

            else:
                fa_match = re.search(r'FINAL_ANSWER:\s*(\{.*\})', response, re.DOTALL)
                if fa_match:
                    final_answer = self._extract_json(fa_match.group(1))
                    break

                final_answer = self._extract_json_from_response(response)
                if final_answer:
                    break

                messages.append(Message(role="assistant", content=response))
                messages.append(Message(role="user", content="Please provide your final answer now using the FINAL_ANSWER format."))
                conversation.append({"role": "system_nudge", "content": "Nudged for final answer", "turn": turn})

        if final_answer is None:
            messages.append(Message(
                role="user",
                content=(
                    "You have reached the maximum number of tool calls. You MUST provide your "
                    'FINAL_ANSWER now. Output: FINAL_ANSWER: {"llm_answer_choice": "X", '
                    '"selected_option_text": "...", "reasoning": "..."}'
                ),
            ))
            response = await self.llm_call(messages)
            conversation.append({"role": "assistant", "content": response, "turn": self.max_turns})

            fa_match = re.search(r'FINAL_ANSWER:\s*(\{.*\})', response, re.DOTALL)
            if fa_match:
                final_answer = self._extract_json(fa_match.group(1))

            if final_answer is None:
                final_answer = self._extract_json_from_response(response)

        return ReactResult(
            final_answer=final_answer,
            conversation=conversation,
            tool_calls=tool_calls,
            num_turns=len([c for c in conversation if c["role"] == "assistant"]),
            compression_log=compression_log,
        )
