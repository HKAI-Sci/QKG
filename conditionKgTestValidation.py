"""
Multi-Agent KG Validation Evaluation

Two-agent architecture:
  Agent 1 (Reasoner): Pure LLM, no KG — proposes answer + reasoning claims
  Agent 2 (Validator): KG (optionally with patient context) — validates claims
  Loop: Reasoner proposes → Validator checks → Reasoner reconsiders

Use --no-patient-context to disable patient context enrichment.

Reuses: lib/kg_tools.py, lib/patient_context.py, lib/react_agent.py
"""

import os
import json
import yaml
import asyncio
import random
import re
from pathlib import Path
from tqdm import tqdm

from f1 import config as f1_config
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole
from f1.common.llm_provider import LLMProviderFactory

from lib.react_agent import ReactAgent, Message, ReactResult, MCQ_ANSWER_INSTRUCTION
from lib.patient_context import PatientContextAnalyzer
from lib.kg_tools import (
    tool_search_entity, tool_list_relation_types,
    tool_get_entity_relations_with_context, tool_check_relation_with_context,
    tool_get_entity_relations, tool_check_relation,
    DRUG_DISEASE_RELATIONS,
)

import argparse
from pydantic import BaseModel

# ============================================================
# Schema
# ============================================================

class ReasonerClaim(BaseModel):
    option: str
    claim: str
    supports: bool

class ReasonerResponse(BaseModel):
    llm_answer_choice: str
    selected_option_text: str
    key_claims: list[ReasonerClaim]

# ============================================================
# Config
# ============================================================

def load_config():
    with open("./conf/config.yaml", "r") as f:
        return yaml.safe_load(f)

CFG = load_config()
os.environ["AWS_ACCESS_KEY_ID"] = CFG["aws"]["access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = CFG["aws"]["secret_access_key"]
os.environ["AWS_REGION_NAME"] = CFG["aws"]["region"]

# INPUT_FILE = "output/v3_bfs_wrong_v1_correct.jsonl"
INPUT_FILE = "data/top_samples.jsonl"
SAMPLE_SIZE = 3000
EXCLUDE_KEYS = {"medqa_839"}
MAX_VALIDATION_ROUNDS = 2   # max reasoner-validator round trips
VALIDATOR_MAX_TURNS = 20    # max tool calls for validator per round
LLM_PER_AGENT = 5
MAX_RETRIES = 6

# Patient-context mode: haiku-bedrock (reasoner) + qwen-openrouter (validator/context)
llm_cfg_reasoner_pc = LLMConfig(**f1_config["haiku-bedrock"])
provider_reasoner_pc = LLMProviderFactory.create_instance(llm_cfg_reasoner_pc)

llm_cfg_validator_pc = LLMConfig(**f1_config["qwen-openrouter"])
provider_validator_pc = LLMProviderFactory.create_instance(llm_cfg_validator_pc)

# No-patient-context mode: validator-llm for both
llm_cfg_nopc = LLMConfig(**f1_config["haiku-bedrock"])
provider_nopc = LLMProviderFactory.create_instance(llm_cfg_nopc)


# ============================================================
# LLM call wrapper
# ============================================================

def make_llm_call(semaphore: asyncio.Semaphore, provider):
    stats = {"calls": 0, "wait_s": 0.0, "llm_s": 0.0}

    async def llm_call(messages: list[Message]) -> str:
        loop = asyncio.get_event_loop()
        f1_msgs = [LLMMessageTextOnly(role=LLMRole(m.role), content=m.content) for m in messages]
        t0 = loop.time()
        async with semaphore:
            t1 = loop.time()
            stats["wait_s"] += t1 - t0
            for attempt in range(MAX_RETRIES):
                try:
                    response = await provider.async_chat_completion(
                        messages=f1_msgs, temperature=0, max_tokens=3000,
                    )
                    if isinstance(response, dict):
                        response = (response.get("content") or response.get("text")
                                    or json.dumps(response))
                    if not response:
                        raise ValueError("Empty response from LLM (finish_reason=stop, no content)")
                    t2 = loop.time()
                    stats["llm_s"] += t2 - t1
                    stats["calls"] += 1
                    return str(response)
                except Exception:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt + random.uniform(1, 5))
        return ""

    llm_call.stats = stats
    return llm_call


# ============================================================
# Question text
# ============================================================

def compose_full_question(sample):
    question = sample.get("question", "").strip()
    options = sample.get("options", "")
    if isinstance(options, list):
        return question + "\n" + "\n".join(options)
    elif isinstance(options, str) and options:
        return question + "\n" + options
    return question


# ============================================================
# System Prompts
# ============================================================

REASONER_SYSTEM_PROMPT = f"""You are a medical expert answering a multiple-choice question.
You do NOT have access to any external tools or databases. Use your own medical knowledge to reason through the question carefully.

Your task:
1. Analyze the question and all answer options
2. For your chosen answer, state the key medical claims that SUPPORT it
3. For EACH rejected option, state the key medical claim that ELIMINATES it
4. Each claim must be a specific, verifiable medical statement

Return JSON matching this schema exactly:

{json.dumps(ReasonerResponse.model_json_schema(), indent=2)}

Rules:
- JSON only. No markdown. No extra text outside the JSON.
- llm_answer_choice must be a single capital letter (A-J)
- selected_option_text must exactly match the chosen option text
- key_claims must include at least one claim per answer option:
  - For the chosen option: "supports": true, with a claim explaining WHY it is correct
  - For each rejected option: "supports": false, with a claim explaining WHY it is eliminated
- Each claim should be a specific, verifiable medical fact (e.g., "Indomethacin reduces renal lithium clearance", "Metformin is contraindicated in eGFR < 30")"""


VALIDATOR_SYSTEM_PROMPT_PC = """You are a medical knowledge validator. Your job is to validate a list of medical claims using PrimeKG, a biomedical knowledge graph with patient-specific annotations.

You will receive:
1. A medical question (for context)
2. A list of claims to validate, each tagged with an answer option and whether it supports or eliminates that option

For EACH claim, use the KG tools to find supporting or contradicting evidence. Validate ALL claims with equal rigor — do not prioritize any particular option.

Available tools (use ACTION: to invoke):
- list_relation_types() — list all KG relation types. CALL THIS FIRST.
- search_entity(query, type=None, source=None, limit=10) — find entities by name
- get_entity_relations_with_context(entity_index, relation_type=None, limit=20) — get KG edges with patient-relevance annotations
- check_relation_with_context(entity_a_index, entity_b_index) — check relations between two entities with patient-relevance annotations

Some relations include patient-relevance annotations:
- _patient_relevance: "Definitely Applicable" or "Increased Likelihood" — HIGH weight evidence
Relations that are NOT applicable to this patient have already been removed.

IMPORTANT:
- Validate ALL claims, not just a subset
- Check elimination claims as carefully as support claims — a wrong elimination is as important as a wrong support
- Be specific about what KG evidence you found

When done, output your validation report:
FINAL_ANSWER: {"validation_report": [{"option": "A", "claim": "...", "supports": true, "status": "SUPPORTED|CONTRADICTED|NO_COVERAGE", "evidence": "..."}, ...]}"""


VALIDATOR_SYSTEM_PROMPT_NOPC = """You are a medical knowledge validator. Your job is to validate a list of medical claims using PrimeKG, a biomedical knowledge graph.

You will receive:
1. A medical question (for context)
2. A list of claims to validate, each tagged with an answer option and whether it supports or eliminates that option

For EACH claim, use the KG tools to find supporting or contradicting evidence. Validate ALL claims with equal rigor — do not prioritize any particular option.

Available tools (use ACTION: to invoke):
- list_relation_types() — list all KG relation types. CALL THIS FIRST.
- search_entity(query, type=None, source=None, limit=10) — find entities by name
- get_entity_relations(entity_index, relation_type=None, limit=20) — get KG edges for an entity
- check_relation(entity_a_index, entity_b_index) — check relations between two entities

IMPORTANT:
- Validate ALL claims, not just a subset
- Check elimination claims as carefully as support claims — a wrong elimination is as important as a wrong support
- Be specific about what KG evidence you found

When done, output your validation report:
FINAL_ANSWER: {"validation_report": [{"option": "A", "claim": "...", "supports": true, "status": "SUPPORTED|CONTRADICTED|NO_COVERAGE", "evidence": "..."}, ...]}"""


RECONSIDER_PROMPT_TEMPLATE = """You previously answered this question and provided key claims for each option.
A knowledge graph validator has checked ALL your claims against a biomedical KG{pc_note}. Here is the validation report:

{{validation_report}}

Interpret the validation results:

SUPPORT claims (why you chose your answer):
- SUPPORTED: KG backs your choice — keep your answer unless there is compelling evidence for an alternative
- NO_COVERAGE: KG has no data on this claim — this is NEUTRAL, not evidence against you; your clinical reasoning still stands
- CONTRADICTED: KG opposes your support claim — reconsider carefully

ELIMINATE claims (why you rejected other options):
- CONTRADICTED: KG contradicts your elimination — that option deserves reconsideration
- NO_COVERAGE: KG has no data on this claim — this is NEUTRAL, not evidence against you; your clinical reasoning still stands
- SUPPORTED: KG confirms your elimination — that option is unlikely correct

Decision rule (apply in order):
1. If your chosen answer's support claim is SUPPORTED: keep your answer UNLESS another option's elimination is CONTRADICTED AND that option also has its own SUPPORTED evidence.
2. If your chosen answer's support claim is CONTRADICTED AND an alternative option has SUPPORTED evidence: switch to that option.
3. If NO_COVERAGE only: KEEP your original answer — your clinical reasoning stands.
4. Default to keeping your original answer when uncertain — a wrong switch is worse than no switch.

Output your final answer:
FINAL_ANSWER: {{"llm_answer_choice": "X", "selected_option_text": "...", "reasoning": "..."}}

Rules:
- llm_answer_choice must be a single capital letter (A-J)
- selected_option_text must exactly match the chosen option text"""

RECONSIDER_PROMPT_PC = RECONSIDER_PROMPT_TEMPLATE.format(pc_note=" with patient-specific annotations")
RECONSIDER_PROMPT_NOPC = RECONSIDER_PROMPT_TEMPLATE.format(pc_note="")


# ============================================================
# Compact relations (used in patient-context mode)
# ============================================================

def _compact_relations(relations: list[dict]) -> list[dict]:
    compacted = []
    for r in relations:
        c = {
            "relation": r["relation"],
            "target_name": r["target_name"],
            "target_index": r["target_index"],
        }
        if "_patient_relevance" in r:
            c["_patient_relevance"] = r["_patient_relevance"]
        compacted.append(c)
    compacted.sort(key=lambda x: (0 if "_patient_relevance" in x else 1,
                                  x["target_name"]))
    return compacted


# ============================================================
# Evaluate one sample: Two-agent validation
# ============================================================

async def evaluate_sample(record: dict, verbose: bool = False,
                          use_patient_context: bool = True) -> dict:
    sample_key = record["sample_key"]
    original = record["original_sample"]
    gold_answer = original.get("answer_key") or original.get("gold_answer")
    v1_answer = record.get("v1_eval", {}).get("parsed", {}).get("llm_answer_choice")
    v3_answer = record.get("v3_eval", {}).get("parsed", {}).get("llm_answer_choice")

    full_question = compose_full_question(original)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[{sample_key}] gold={gold_answer} v1={v1_answer} v3={v3_answer}")
        print(f"{'='*60}")

    agent_semaphore = asyncio.Semaphore(LLM_PER_AGENT)

    if use_patient_context:
        llm_call_context = make_llm_call(agent_semaphore, provider_validator_pc)
        llm_call_reasoner = make_llm_call(agent_semaphore, provider_reasoner_pc)
        llm_call_validator = make_llm_call(agent_semaphore, provider_validator_pc)
    else:
        llm_call_reasoner = make_llm_call(agent_semaphore, provider_nopc)
        llm_call_validator = make_llm_call(agent_semaphore, provider_nopc)

    # ── Phase 0: Extract patient context (PC mode only) ───────
    patient_context = None
    _pc_extract_calls = 0
    if use_patient_context:
        analyzer = PatientContextAnalyzer(llm_call_context)
        patient_context = await analyzer.extract(full_question)
        _pc_extract_calls = llm_call_context.stats["calls"]
        if verbose:
            print(f"[PATIENT CONTEXT] {patient_context[:200]}...")

    # ── Phase 1: Reasoner proposes answer (pure LLM, no KG) ──
    if verbose:
        print(f"\n--- PHASE 1: Reasoner (pure LLM) ---")

    reasoner_messages = [
        Message(role="system", content=REASONER_SYSTEM_PROMPT),
        Message(role="user", content=f"Question:\n{full_question}"),
    ]
    reasoner_response = await llm_call_reasoner(reasoner_messages)

    # Parse reasoner's answer and claims
    reasoner_answer = None
    reasoner_claims = []
    reasoner_final = ReactAgent._extract_json_from_response(reasoner_response)
    if not reasoner_final:
        fa_match = re.search(r'FINAL_ANSWER:\s*(\{.*\})', reasoner_response, re.DOTALL)
        if fa_match:
            reasoner_final = ReactAgent._extract_json(fa_match.group(1))

    if reasoner_final:
        reasoner_answer = reasoner_final.get("llm_answer_choice")
        if not reasoner_answer:
            print(f"  [WARN:{sample_key}] reasoner_final parsed but llm_answer_choice missing. "
                  f"Keys: {list(reasoner_final.keys())}\n"
                  f"  reasoner_final: {json.dumps(reasoner_final, ensure_ascii=False)[:500]}")
        reasoner_claims = reasoner_final.get("key_claims", [])
        if isinstance(reasoner_claims, str):
            reasoner_claims = [{"option": "?", "claim": reasoner_claims, "supports": True}]
        normalized_claims = []
        for c in reasoner_claims:
            if isinstance(c, str):
                normalized_claims.append({"option": "?", "claim": c, "supports": True})
            elif isinstance(c, dict):
                normalized_claims.append(c)
        reasoner_claims = normalized_claims
    else:
        print(f"  [WARN:{sample_key}] reasoner parse failed entirely. "
              f"Response (first 800 chars):\n{reasoner_response[:800]}")

    if verbose:
        print(f"  Reasoner answer: {reasoner_answer}")
        print(f"  Key claims ({len(reasoner_claims)}):")
        for i, c in enumerate(reasoner_claims):
            tag = "SUPPORT" if c.get("supports") else "ELIMINATE"
            print(f"    [{c.get('option','?')}:{tag}] {c.get('claim','?')}")

    # ── Phase 2: Validator checks claims against KG ───────────
    if verbose:
        mode_label = "enriched KG + patient context" if use_patient_context else "KG, no patient context"
        print(f"\n--- PHASE 2: Validator ({mode_label}) ---")

    hook_log = []
    safety_judgments = []

    if use_patient_context:
        async def result_hook(result, tool_name, patient_ctx):
            if tool_name not in ("get_entity_relations_with_context",
                                 "check_relation_with_context"):
                return result
            if not isinstance(result, list) or not result:
                return result

            # Drug-disease reclassification
            if result[0].get("_requested_relation"):
                requested = result[0]["_requested_relation"]
                for r in result:
                    r.pop("_requested_relation", None)
                before = len(result)
                result = await analyzer.analyze_drug_disease(
                    result, patient_ctx, requested_relation=requested)
                after = len(result)

                judgments = await analyzer.judge_drug_safety(result, patient_ctx)
                safety_judgments.extend(judgments)

                hook_log.append({"total": before, "requested": requested,
                                 "kept": after, "removed": before - after,
                                 "path": "drug_disease_reclassify"})
                if verbose:
                    print(f"  [HOOK:drug_disease] {before} → {after} kept as '{requested}'")
                    for j in judgments:
                        print(f"  [SAFETY] {j.get('drug')}: {j.get('status')} — {j.get('reason','')[:80]}")
                compacted = _compact_relations(result)
                for j in judgments:
                    compacted.append({"_drug_safety_judgment": j.get("drug"),
                                      "status": j.get("status"),
                                      "reason": j.get("reason")})
                return compacted

            # check_relation split
            if tool_name == "check_relation_with_context":
                dd = [r for r in result if r.get("relation") in DRUG_DISEASE_RELATIONS]
                other = [r for r in result if r.get("relation") not in DRUG_DISEASE_RELATIONS]
                dd_judgments = []
                if dd:
                    dd = await analyzer.analyze_drug_disease(dd, patient_ctx)
                    dd_judgments = await analyzer.judge_drug_safety(dd, patient_ctx)
                    safety_judgments.extend(dd_judgments)
                if other:
                    other = await analyzer.analyze(other, patient_ctx)
                combined = dd + other
                for j in dd_judgments:
                    combined.append({"_drug_safety_judgment": j.get("drug"),
                                     "status": j.get("status"),
                                     "reason": j.get("reason")})
                hook_log.append({"total": len(result), "kept": len(combined),
                                 "path": "check_relation"})
                if verbose and dd_judgments:
                    for j in dd_judgments:
                        print(f"  [SAFETY] {j.get('drug')}: {j.get('status')} — {j.get('reason','')[:80]}")
                return combined

            # Standard enrichment filtering
            enriched = [r for r in result if "context_constraints" in r]
            if not enriched:
                hook_log.append({"total": len(result), "enriched": 0})
                return _compact_relations(result)
            before = len(result)
            result = await analyzer.analyze(result, patient_ctx)
            after = len(result)
            annotated = sum(1 for r in result if "_patient_relevance" in r)
            hook_log.append({"total": before, "enriched": len(enriched),
                             "kept": after, "removed": before - after,
                             "annotated": annotated})
            if verbose:
                print(f"  [HOOK] {before} → {after} kept, {annotated} annotated")
            return _compact_relations(result)

        validator_tools = {
            "list_relation_types": tool_list_relation_types,
            "search_entity": tool_search_entity,
            "get_entity_relations_with_context": tool_get_entity_relations_with_context,
            "check_relation_with_context": tool_check_relation_with_context,
        }
        validator_system_prompt = VALIDATOR_SYSTEM_PROMPT_PC
        reconsider_prompt_template = RECONSIDER_PROMPT_PC
    else:
        result_hook = None
        validator_tools = {
            "list_relation_types": tool_list_relation_types,
            "search_entity": tool_search_entity,
            "get_entity_relations": tool_get_entity_relations,
            "check_relation": tool_check_relation,
        }
        validator_system_prompt = VALIDATOR_SYSTEM_PROMPT_NOPC
        reconsider_prompt_template = RECONSIDER_PROMPT_NOPC

    # Build validator input: question + structured claims only (no answer, no reasoning)
    claims_lines = []
    for i, c in enumerate(reasoner_claims):
        tag = "SUPPORTS" if c.get("supports") else "ELIMINATES"
        claims_lines.append(f"  {i+1}. [Option {c.get('option','?')}, {tag}] {c.get('claim','?')}")
    claims_text = "\n".join(claims_lines)
    validator_input = (
        f"Question:\n{full_question}\n\n"
        f"Claims to validate:\n{claims_text}"
    )

    validator_agent_kwargs = dict(
        tools=validator_tools,
        llm_call=llm_call_validator,
        system_prompt=validator_system_prompt,
        max_turns=VALIDATOR_MAX_TURNS,
        memory_compression=True,
        compression_llm_call=llm_call_validator,
    )
    if use_patient_context:
        validator_agent_kwargs["result_hook"] = result_hook
        validator_agent_kwargs["patient_context"] = patient_context

    validator_agent = ReactAgent(**validator_agent_kwargs)
    validator_result = await validator_agent.run(validator_input)

    # Parse validation report
    validation_report = []
    if validator_result.final_answer:
        validation_report = validator_result.final_answer.get("validation_report", [])
    if not validation_report:
        last_asst = [m for m in validator_result.conversation if m["role"] == "assistant"]
        if last_asst:
            raw_text = last_asst[-1]["content"]
            fa_match = re.search(r'FINAL_ANSWER:\s*(\{.*\})', raw_text, re.DOTALL)
            if fa_match:
                parsed = ReactAgent._extract_json(fa_match.group(1))
                if parsed and "validation_report" in parsed:
                    validation_report = parsed["validation_report"]
    if isinstance(validation_report, str):
        try:
            validation_report = json.loads(validation_report)
        except Exception:
            validation_report = [{"claim": validation_report, "status": "PARSE_ERROR"}]

    if verbose:
        print(f"  Validator report ({len(validation_report)} claims):")
        for vr in validation_report:
            status = vr.get("status", "?")
            claim = vr.get("claim", "?")[:80]
            evidence = vr.get("evidence", "")[:100]
            print(f"    [{status}] {claim}")
            if evidence:
                print(f"           Evidence: {evidence}")

    has_contradiction = any(vr.get("status") == "CONTRADICTED" for vr in validation_report)

    final_answer = reasoner_answer
    reconsidered = False
    reconsider_final = None

    if has_contradiction and reasoner_answer:
        print(f"  [INFO] Contradiction found in validation report, triggering reconsideration")
        if verbose:
            print(f"\n--- PHASE 3: Reasoner reconsiders (contradictions found) ---")

        report_text = json.dumps(validation_report, indent=2, ensure_ascii=False)
        reconsider_prompt = reconsider_prompt_template.format(
            validation_report=report_text)

        reconsider_messages = [
            Message(role="system", content=REASONER_SYSTEM_PROMPT),
            Message(role="user", content=f"Question:\n{full_question}"),
            Message(role="assistant", content=reasoner_response),
            Message(role="user", content=reconsider_prompt),
        ]

        reconsider_response = await llm_call_reasoner(reconsider_messages)
        reconsider_final = ReactAgent._extract_json_from_response(reconsider_response)
        if not reconsider_final:
            fa_match = re.search(r'FINAL_ANSWER:\s*(\{.*\})', reconsider_response, re.DOTALL)
            if fa_match:
                reconsider_final = ReactAgent._extract_json(fa_match.group(1))

        if reconsider_final and reconsider_final.get("llm_answer_choice"):
            new_answer = reconsider_final["llm_answer_choice"]
            if new_answer != reasoner_answer:
                reconsidered = True
                if verbose:
                    print(f"  Answer changed: {reasoner_answer} → {new_answer}")
            else:
                if verbose:
                    print(f"  Answer unchanged: {reasoner_answer}")
            final_answer = new_answer
        print(f"  Reconsideration response:\n{reconsider_response[:1000]}")
    else:
        if verbose:
            print(f"\n--- PHASE 3: Skipped (no contradictions) ---")

    # ── Build result ──────────────────────────────────────────

    result = {
        "sample_key": sample_key,
        "gold_answer": gold_answer,
        "v1_answer": v1_answer,
        "v3_answer": v3_answer,
        # Reasoner output
        "reasoner_answer": reasoner_answer,
        "reasoner_claims": reasoner_claims,
        "reconsider_reasoning": reconsider_final.get("reasoning") if reconsider_final else None,
        # Validator output
        "validation_report": validation_report,
        "validator_turns": validator_result.num_turns,
        "validator_tool_calls": len(validator_result.tool_calls),
        "validator_conversation": validator_result.conversation,
        # Final output
        "final_answer": final_answer,
        "reconsidered": reconsidered,
        "final_correct": final_answer == gold_answer if final_answer else False,
        "reasoner_correct": reasoner_answer == gold_answer if reasoner_answer else False,
        "v1_correct": v1_answer == gold_answer if v1_answer else False,
        "v3_correct": v3_answer == gold_answer if v3_answer else False,
        # Diagnostics
        "has_contradiction": has_contradiction,
        "num_supported": sum(1 for v in validation_report if v.get("status") == "SUPPORTED"),
        "num_contradicted": sum(1 for v in validation_report if v.get("status") == "CONTRADICTED"),
        "num_partial": sum(1 for v in validation_report if v.get("status") in ("PARTIAL_COVERAGE", "NO_COVERAGE")),
        "num_claims": len(validation_report),
    }

    if use_patient_context:
        result["hook_log"] = hook_log
        result["drug_safety_judgments"] = safety_judgments
        result["patient_context"] = patient_context
        result["pc_extract_calls"] = _pc_extract_calls
        result["pc_analyze_calls"] = llm_call_context.stats["calls"] - _pc_extract_calls
        result["pc_call_stats"] = analyzer.call_stats
        result["llm_stats"] = {
            "calls": (llm_call_context.stats["calls"] + llm_call_reasoner.stats["calls"]
                      + llm_call_validator.stats["calls"]),
            "wait_s": round(llm_call_context.stats["wait_s"] + llm_call_reasoner.stats["wait_s"]
                            + llm_call_validator.stats["wait_s"], 1),
            "llm_s": round(llm_call_context.stats["llm_s"] + llm_call_reasoner.stats["llm_s"]
                           + llm_call_validator.stats["llm_s"], 1),
            "calls_context": llm_call_context.stats["calls"],
            "calls_reasoner": llm_call_reasoner.stats["calls"],
            "calls_validator": llm_call_validator.stats["calls"],
            "pc_extract_calls": _pc_extract_calls,
            "pc_analyze_calls": llm_call_context.stats["calls"] - _pc_extract_calls,
        }
    else:
        result["llm_stats"] = {
            "calls": llm_call_reasoner.stats["calls"] + llm_call_validator.stats["calls"],
            "wait_s": round(llm_call_reasoner.stats["wait_s"] + llm_call_validator.stats["wait_s"], 1),
            "llm_s": round(llm_call_reasoner.stats["llm_s"] + llm_call_validator.stats["llm_s"], 1),
            "calls_reasoner": llm_call_reasoner.stats["calls"],
            "calls_validator": llm_call_validator.stats["calls"],
        }

    if validator_result.compression_log:
        result["compression_log"] = validator_result.compression_log

    return result


# ============================================================
# Verbose trace
# ============================================================

def print_trace(result: dict):
    sk = result["sample_key"]
    print(f"\n{'='*60}")
    print(f"[{sk}] gold={result['gold_answer']}")
    print(f"  Reasoner: {result['reasoner_answer']} "
          f"({'correct' if result['reasoner_correct'] else 'wrong'})")
    print(f"  Final:    {result['final_answer']} "
          f"({'correct' if result['final_correct'] else 'wrong'})"
          f"{' (RECONSIDERED)' if result['reconsidered'] else ''}")
    print(f"  Validation: {result['num_supported']} supported, "
          f"{result['num_contradicted']} contradicted, "
          f"{result['num_partial']} partial/no-coverage "
          f"(of {result['num_claims']} claims)")
    print(f"  Validator: {result['validator_turns']} turns, "
          f"{result['validator_tool_calls']} tool calls")
    print(f"  LLM: {result['llm_stats']['calls']} calls, "
          f"{result['llm_stats']['llm_s']}s inference")

    if result.get("reasoner_claims"):
        print(f"\n  Reasoner claims:")
        for c in result["reasoner_claims"]:
            if isinstance(c, dict):
                tag = "SUPPORT" if c.get("supports") else "ELIMINATE"
                print(f"    [{c.get('option','?')}:{tag}] {c.get('claim','?')}")
            else:
                print(f"    - {c}")

    if result.get("validation_report"):
        print(f"\n  Validation report:")
        for vr in result["validation_report"]:
            opt = vr.get("option", "?")
            supports = vr.get("supports", "?")
            tag = "SUPPORT" if supports else "ELIMINATE"
            print(f"    [{opt}:{tag}] [{vr.get('status', '?')}] {vr.get('claim', '?')[:80]}")
            if vr.get("evidence"):
                print(f"      Evidence: {vr['evidence'][:120]}")

    if result.get("drug_safety_judgments"):
        print(f"\n  Drug safety judgments ({len(result['drug_safety_judgments'])}):")
        for j in result["drug_safety_judgments"]:
            print(f"    [{j.get('status','?')}] {j.get('drug','?')}: {j.get('reason','')[:100]}")

    if result["reconsidered"]:
        print(f"\n  Answer changed: {result['reasoner_answer']} → {result['final_answer']}")
    print()


# ============================================================
# Resume support
# ============================================================

def load_finished_keys(output_file: str) -> set:
    finished = set()
    path = Path(output_file)
    if not path.exists():
        return finished
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("final_answer") is not None:
                    finished.add(obj["sample_key"])
            except Exception:
                pass
    return finished


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent KG validation evaluation.",
        epilog="Examples:\n"
               "  python conditionKgTestValidation_v4.py --run medqa_839\n"
               "  python conditionKgTestValidation_v4.py --run medqa_839,qa_2027\n"
               "  python conditionKgTestValidation_v4.py --concurrency 10\n"
               "  python conditionKgTestValidation_v4.py --no-patient-context\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Run specific cases by sample_key (comma-separated)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent agents (default: 3)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N records in the remaining list (default: 0)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                        help=f"Number of samples to randomly select (default: {SAMPLE_SIZE})")
    parser.add_argument("--no-patient-context", action="store_true",
                        help="Disable patient context enrichment")
    args = parser.parse_args()

    use_patient_context = not args.no_patient_context
    pc_tag = "" if use_patient_context else "_nopc"
    run_keys = [k.strip() for k in args.run.split(",")] if args.run else None

    output_file = args.output or (
        f"output/run_validation{pc_tag}_{'_'.join(run_keys)}.jsonl" if run_keys
        else f"output/validation{pc_tag}_eval_all{'_offset' + str(args.offset) if args.offset else ''}.jsonl"
    )

    mode_str = "with patient context" if use_patient_context else "no patient context"
    print(f"Multi-Agent KG Validation Mode ({mode_str})")
    print(f"  Max validation rounds: {MAX_VALIDATION_ROUNDS}")
    print(f"  Validator max turns: {VALIDATOR_MAX_TURNS}")

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Total records: {len(records)}")

    if run_keys:
        by_key = {r["sample_key"]: r for r in records}
        selected = [by_key[k] for k in run_keys if k in by_key]
        for k in run_keys:
            if k not in by_key:
                print(f"WARNING: {k} not found")
        remaining = selected[args.offset:] if args.offset else selected
        verbose = True
    else:
        candidates = [r for r in records if r["sample_key"] not in EXCLUDE_KEYS]
        selected = candidates[:args.sample_size]
        finished = load_finished_keys(output_file)
        remaining = [r for r in selected if r["sample_key"] not in finished]
        if args.offset:
            remaining = remaining[args.offset:]
        print(f"Selected {len(selected)}, already done {len(finished)}, "
              f"remaining {len(remaining)}"
              + (f" (offset {args.offset})" if args.offset else ""))
        verbose = False

    if not remaining:
        print("Nothing to run.")
    else:
        agent_gate = asyncio.Semaphore(args.concurrency)

        async def run_with_gate(record):
            async with agent_gate:
                t0 = asyncio.get_event_loop().time()
                result = await evaluate_sample(record, verbose=verbose,
                                               use_patient_context=use_patient_context)
                result["elapsed_s"] = round(asyncio.get_event_loop().time() - t0, 1)
                return result

        tasks = [run_with_gate(r) for r in remaining]

        with open(output_file, "a", encoding="utf-8") as f_out:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                             disable=verbose):
                try:
                    result = await coro
                except Exception as e:
                    print(f"  [ERROR] sample failed: {e}")
                    continue
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                if verbose:
                    print_trace(result)
                elif use_patient_context:
                    r_mark = "ok" if result["reasoner_correct"] else "X"
                    f_mark = "ok" if result["final_correct"] else "X"
                    recon = " RECON" if result["reconsidered"] else ""
                    print(f"  {result['sample_key']}: reasoner={result['reasoner_answer']}({r_mark}) "
                          f"→ final={result['final_answer']}({f_mark}){recon} | "
                          f"S={result['num_supported']} C={result['num_contradicted']} "
                          f"P={result.get('num_partial', 0)} ({result.get('num_claims', 0)} claims) | "
                          f"pc_extract={result['llm_stats']['pc_extract_calls']} "
                          f"pc_analyze={result['llm_stats']['pc_analyze_calls']} | "
                          f"{result.get('elapsed_s', '?')}s")
                else:
                    r_mark = "ok" if result["reasoner_correct"] else "X"
                    f_mark = "ok" if result["final_correct"] else "X"
                    recon = " RECON" if result["reconsidered"] else ""
                    print(f"  {result['sample_key']}: reasoner={result['reasoner_answer']}({r_mark}) "
                          f"→ final={result['final_answer']}({f_mark}){recon} | "
                          f"S={result['num_supported']} C={result['num_contradicted']} "
                          f"P={result.get('num_partial', 0)} ({result.get('num_claims', 0)} claims) | "
                          f"{result.get('elapsed_s', '?')}s")

        print(f"\nResults saved to {output_file}")

    # Summary table
    all_results = []
    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    if all_results:
        total = len(all_results)
        v1_c = sum(1 for r in all_results if r["v1_correct"])
        reasoner_c = sum(1 for r in all_results if r["reasoner_correct"])
        final_c = sum(1 for r in all_results if r["final_correct"])
        reconsidered = sum(1 for r in all_results if r["reconsidered"])
        recon_helped = sum(1 for r in all_results
                          if r["reconsidered"] and r["final_correct"]
                          and not r["reasoner_correct"])
        recon_hurt = sum(1 for r in all_results
                        if r["reconsidered"] and not r["final_correct"]
                        and r["reasoner_correct"])

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  v1 (LLM only, precomputed): {v1_c}/{total} ({100*v1_c/total:.1f}%)")
        print(f"  Reasoner (LLM only, fresh):  {reasoner_c}/{total} ({100*reasoner_c/total:.1f}%)")
        print(f"  Final (after validation):    {final_c}/{total} ({100*final_c/total:.1f}%)")
        print(f"\n  Reconsidered: {reconsidered}/{total}")
        print(f"  Reconsider helped (wrong→right): {recon_helped}")
        print(f"  Reconsider hurt (right→wrong):   {recon_hurt}")
        print(f"  Net effect of validation: {recon_helped - recon_hurt:+d}")


if __name__ == "__main__":
    asyncio.run(main())
