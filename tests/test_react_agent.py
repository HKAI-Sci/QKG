"""Unit tests for lib.react_agent — no LLM or network calls needed."""

import asyncio
import json
import pytest

from lib.react_agent import ReactAgent, Message, ReactResult


# ============================================================
# Helpers
# ============================================================

def make_agent(tools=None, llm_call=None, max_turns=5, max_result_chars=4000):
    """Build a ReactAgent with sensible defaults for testing."""
    if tools is None:
        tools = {
            "add": lambda a, b: {"sum": a + b},
            "greet": lambda name, greeting="hello": f"{greeting}, {name}!",
        }
    if llm_call is None:
        async def llm_call(msgs):
            return ""
    return ReactAgent(
        tools=tools,
        llm_call=llm_call,
        system_prompt="You are a test agent.",
        max_turns=max_turns,
        max_result_chars=max_result_chars,
    )


def scripted_llm(responses: list[str]):
    """Return an async llm_call that yields canned responses in order."""
    it = iter(responses)
    async def llm_call(msgs):
        return next(it)
    return llm_call


# ============================================================
# _extract_json
# ============================================================

class TestExtractJson:
    def test_plain_json(self):
        assert ReactAgent._extract_json('{"a": 1}') == {"a": 1}

    def test_json_in_markdown_block(self):
        text = '```json\n{"a": 1}\n```'
        assert ReactAgent._extract_json(text) == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = 'Here is the answer: {"key": "val"} done.'
        assert ReactAgent._extract_json(text) == {"key": "val"}

    def test_nested_json(self):
        obj = {"outer": {"inner": [1, 2]}}
        text = f"prefix {json.dumps(obj)} suffix"
        assert ReactAgent._extract_json(text) == obj

    def test_no_json(self):
        assert ReactAgent._extract_json("no json here") is None

    def test_empty_string(self):
        assert ReactAgent._extract_json("") is None

    def test_malformed_json(self):
        assert ReactAgent._extract_json("{bad json}") is None


# ============================================================
# _extract_json_from_response
# ============================================================

class TestExtractJsonFromResponse:
    def test_finds_answer_choice(self):
        resp = 'Some reasoning.\n{"llm_answer_choice": "A", "reasoning": "because"}'
        result = ReactAgent._extract_json_from_response(resp)
        assert result["llm_answer_choice"] == "A"

    def test_no_answer_choice_keyword(self):
        resp = '{"key": "value"}'
        assert ReactAgent._extract_json_from_response(resp) is None

    def test_answer_choice_in_markdown(self):
        resp = '```json\n{"llm_answer_choice": "B"}\n```'
        result = ReactAgent._extract_json_from_response(resp)
        assert result["llm_answer_choice"] == "B"


# ============================================================
# _parse_xml_calls
# ============================================================

class TestParseXmlCalls:
    def test_single_call(self):
        xml = '<invoke name="search_entity"><parameter name="query">aspirin</parameter></invoke>'
        calls = ReactAgent._parse_xml_calls(xml)
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "search_entity"
        assert calls[0]["kwargs"] == {"query": "aspirin"}

    def test_int_coercion(self):
        xml = '<invoke name="get_rel"><parameter name="index">12345</parameter></invoke>'
        calls = ReactAgent._parse_xml_calls(xml)
        assert calls[0]["kwargs"]["index"] == 12345
        assert isinstance(calls[0]["kwargs"]["index"], int)

    def test_null_skipped(self):
        xml = '<invoke name="search"><parameter name="q">x</parameter><parameter name="type">None</parameter></invoke>'
        calls = ReactAgent._parse_xml_calls(xml)
        assert "type" not in calls[0]["kwargs"]

    def test_multiple_calls(self):
        xml = (
            '<invoke name="a"><parameter name="x">1</parameter></invoke>'
            '<invoke name="b"><parameter name="y">2</parameter></invoke>'
        )
        calls = ReactAgent._parse_xml_calls(xml)
        assert len(calls) == 2
        assert calls[0]["tool_name"] == "a"
        assert calls[1]["tool_name"] == "b"

    def test_no_xml(self):
        assert ReactAgent._parse_xml_calls("plain text") == []


# ============================================================
# _format_result / _execute_action / _execute_by_name
# ============================================================

class TestToolExecution:
    def test_format_result_normal(self):
        agent = make_agent(max_result_chars=4000)
        out = agent._format_result({"a": 1})
        assert json.loads(out) == {"a": 1}

    def test_format_result_truncation(self):
        agent = make_agent(max_result_chars=20)
        out = agent._format_result({"long_key": "x" * 100})
        assert out.endswith("... (truncated)")
        assert len(out) < 100

    def test_execute_action_positional(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_action('add(2, 3)'))
        assert json.loads(out) == {"sum": 5}

    def test_execute_action_keyword(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_action('greet("world", greeting="hi")'))
        assert json.loads(out) == "hi, world!"

    def test_execute_action_unknown_tool(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_action('unknown_tool(1)'))
        assert "Unknown tool" in out

    def test_execute_action_parse_error(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_action('not a call'))
        assert "Could not parse" in out

    def test_execute_by_name(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_by_name("add", {"a": 10, "b": 20}))
        assert json.loads(out) == {"sum": 30}

    def test_execute_by_name_unknown(self):
        agent = make_agent()
        out = asyncio.run(agent._execute_by_name("nope", {}))
        assert "Unknown tool" in out


# ============================================================
# run() — full loop with mocked LLM
# ============================================================

class TestRunLoop:
    def test_action_then_final_answer(self):
        """LLM calls a tool, gets observation, then gives FINAL_ANSWER."""
        llm = scripted_llm([
            'Let me look this up.\nACTION: add(1, 2)',
            'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "1+2=3"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("What is 1+2?"))

        assert isinstance(result, ReactResult)
        assert result.final_answer["llm_answer_choice"] == "A"
        assert result.num_turns == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["action"] == "add(1, 2)"

    def test_xml_tool_call(self):
        """LLM uses XML format for tool call."""
        llm = scripted_llm([
            'Thinking.\n<invoke name="greet"><parameter name="name">Alice</parameter></invoke>',
            'FINAL_ANSWER: {"llm_answer_choice": "B", "reasoning": "greeted"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Greet Alice"))

        assert result.final_answer["llm_answer_choice"] == "B"
        assert len(result.tool_calls) == 1
        assert "greet" in result.tool_calls[0]["action"]

    def test_immediate_final_answer(self):
        """LLM answers without any tool calls."""
        llm = scripted_llm([
            'FINAL_ANSWER: {"llm_answer_choice": "C", "reasoning": "obvious"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Easy question"))

        assert result.final_answer["llm_answer_choice"] == "C"
        assert result.num_turns == 1
        assert len(result.tool_calls) == 0

    def test_nudge_then_answer(self):
        """LLM gives text without ACTION or FINAL_ANSWER, gets nudged, then answers."""
        llm = scripted_llm([
            "I'm thinking about this...",
            'FINAL_ANSWER: {"llm_answer_choice": "D", "reasoning": "after nudge"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Question"))

        assert result.final_answer["llm_answer_choice"] == "D"
        assert result.num_turns == 2
        # Check nudge was recorded
        nudges = [c for c in result.conversation if c["role"] == "system_nudge"]
        assert len(nudges) == 1

    def test_max_turns_forces_answer(self):
        """When all turns are used without answering, agent forces a final answer."""
        responses = ["Hmm let me think..."] * 3  # fill max_turns=3
        responses.append('FINAL_ANSWER: {"llm_answer_choice": "E", "reasoning": "forced"}')
        llm = scripted_llm(responses)
        agent = make_agent(llm_call=llm, max_turns=3)
        result = asyncio.run(agent.run("Hard question"))

        assert result.final_answer["llm_answer_choice"] == "E"

    def test_max_turns_no_answer(self):
        """When even the forced final prompt gets no answer, final_answer is None."""
        responses = ["still thinking..."] * 4  # 3 turns + 1 forced
        llm = scripted_llm(responses)
        agent = make_agent(llm_call=llm, max_turns=3)
        result = asyncio.run(agent.run("Impossible question"))

        assert result.final_answer is None

    def test_hallucinated_observation_truncated(self):
        """Text after ACTION: line should not leak into the assistant message."""
        llm = scripted_llm([
            'Thinking.\nACTION: add(1, 2)\nOBSERVATION: {"sum": 999}\nThe answer is 999.',
            'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "done"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Question"))

        # The observation in conversation should be real (sum=3), not hallucinated (sum=999)
        obs = [c for c in result.conversation if c["role"] == "observation"]
        assert len(obs) == 1
        assert '"sum": 3' in obs[0]["content"]

    def test_multiple_tool_calls(self):
        """Multiple rounds of tool use before final answer."""
        llm = scripted_llm([
            'ACTION: add(1, 2)',
            'ACTION: add(3, 4)',
            'ACTION: greet("world")',
            'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "done"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Multi-step question"))

        assert len(result.tool_calls) == 3
        assert result.num_turns == 4

    def test_json_without_final_answer_prefix(self):
        """LLM outputs JSON with llm_answer_choice but no FINAL_ANSWER: prefix."""
        llm = scripted_llm([
            'ACTION: add(1, 2)',
            'Based on my analysis: {"llm_answer_choice": "B", "reasoning": "sum is 3"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Question"))

        assert result.final_answer["llm_answer_choice"] == "B"

    def test_xml_multiple_calls_only_first_executed(self):
        """When LLM emits multiple XML calls, only the first is executed per turn."""
        xml_response = (
            'Let me search.\n'
            '<invoke name="add"><parameter name="a">1</parameter><parameter name="b">2</parameter></invoke>'
            '<invoke name="greet"><parameter name="name">Bob</parameter></invoke>'
        )
        llm = scripted_llm([
            xml_response,
            'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "done"}',
        ])
        agent = make_agent(llm_call=llm)
        result = asyncio.run(agent.run("Question"))

        assert len(result.tool_calls) == 1
        assert "add" in result.tool_calls[0]["action"]
        # Check remaining-calls info was sent
        obs_msgs = [c for c in result.conversation if c["role"] == "observation"]
        assert len(obs_msgs) == 1


# ============================================================
# run() — message construction checks
# ============================================================

class TestMessageConstruction:
    def test_system_and_user_messages_present(self):
        """First two messages should be system prompt and user message."""
        captured_messages = []

        async def capturing_llm(msgs):
            captured_messages.append([Message(role=m.role, content=m.content) for m in msgs])
            return 'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "x"}'

        agent = make_agent(llm_call=capturing_llm)
        asyncio.run(agent.run("Test question"))

        first_call = captured_messages[0]
        assert first_call[0].role == "system"
        assert first_call[0].content == "You are a test agent."
        assert first_call[1].role == "user"
        assert "Test question" in first_call[1].content

    def test_observation_appended_as_user_message(self):
        """After a tool call, observation should appear as a user message."""
        captured_messages = []

        async def capturing_llm(msgs):
            captured_messages.append(list(msgs))
            if len(captured_messages) == 1:
                return "ACTION: add(1, 2)"
            return 'FINAL_ANSWER: {"llm_answer_choice": "A", "reasoning": "x"}'

        agent = make_agent(llm_call=capturing_llm)
        asyncio.run(agent.run("Question"))

        # Second LLM call should have 4 messages: system, user, assistant, observation
        second_call = captured_messages[1]
        assert len(second_call) == 4
        assert second_call[2].role == "assistant"
        assert second_call[3].role == "user"
        assert "OBSERVATION" in second_call[3].content
