"""
Blind Agentic KG Evaluation

ReAct-style multi-turn prompting where the LLM answers medical QA
questions using KG tools WITHOUT seeing the gold answer.
Compares agentic results against v1 (no KG) and v3 (BFS paths).
"""

import os
import json
import asyncio
import random
from pathlib import Path
from tqdm import tqdm

from f1 import config as f1_config
from f1.common.schema import LLMConfig, LLMMessageTextOnly, LLMRole
from f1.common.llm_provider import LLMProviderFactory

from lib.react_agent import ReactAgent, Message, SYSTEM_PROMPT, SYSTEM_PROMPT_PATIENT_CONTEXT
from lib.patient_context import PatientContextAnalyzer
from lib.kg_tools import (
    tool_search_entity, tool_get_entity_relations,
    tool_get_entity_relations_with_context, tool_list_relation_types,
    tool_check_relation, tool_check_relation_with_context,
    DRUG_DISEASE_RELATIONS,
)
from lib.runtime_config import apply_optional_aws_env, get_optional_path_config

import argparse

# ============================================================
# Config
# ============================================================

apply_optional_aws_env(f1_config)

# INPUT_FILE = "output/v3_bfs_wrong_v1_correct.jsonl"
INPUT_FILE = get_optional_path_config("qa_eval_jsonl", "data/top_2875_path_samples_all_v4.jsonl")


SAMPLE_SIZE = 3000
EXCLUDE_KEYS = {"medqa_839"}
MAX_TURNS = 40
LLM_PER_AGENT = 5       # max concurrent LLM calls within one agent
MAX_RETRIES = 6


def _get_llm_role_config(role: str) -> dict:
    llm_roles = f1_config.get("llm_roles") or {}
    llm_backends = f1_config.get("llm_backends") or {}
    backend_name = llm_roles.get(role)
    if not backend_name:
        raise KeyError(f"Missing llm_roles.{role}")
    backend_cfg = llm_backends.get(backend_name)
    if not backend_cfg:
        raise KeyError(f"Missing llm_backends.{backend_name} for role '{role}'")
    return backend_cfg


reasoner_llm_cfg = LLMConfig(**_get_llm_role_config("reasoner"))
reasoner_provider = LLMProviderFactory.create_instance(reasoner_llm_cfg)

validator_llm_cfg = LLMConfig(**_get_llm_role_config("validator"))
validator_provider = LLMProviderFactory.create_instance(validator_llm_cfg)



# ============================================================
# LLM call wrapper
# ============================================================

def make_llm_call(semaphore: asyncio.Semaphore, provider):
    """Create an LLM call function bound to a per-agent semaphore."""
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
                        messages=f1_msgs,
                        temperature=0,
                        max_tokens=3000,
                    )
                    if isinstance(response, dict):
                        response = (
                            response.get("content")
                            or response.get("text")
                            or json.dumps(response)
                        )
                    t2 = loop.time()
                    stats["llm_s"] += t2 - t1
                    stats["calls"] += 1
                    return str(response)
                except Exception:
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt + random.random())
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
# Evaluate one sample
# ============================================================

def _compact_relations(relations: list[dict]) -> list[dict]:
    """Compact relation dicts and sort annotated ones first.

    Drops verbose fields (display_relation, direction, target_type) to reduce
    JSON size so more relations survive the 8000-char truncation. Sorts
    annotated relations (_patient_relevance) to the front so the most valuable
    data is always visible to the agent.
    """
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
    # Annotated first, then alphabetical by target_name for stability
    compacted.sort(key=lambda x: (0 if "_patient_relevance" in x else 1,
                                  x["target_name"]))
    return compacted


async def evaluate_sample(record: dict, patient_context_enabled: bool = False,
                          verbose: bool = False) -> dict:
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

    # Per-agent semaphore: controls concurrent LLM calls within this agent
    agent_semaphore = asyncio.Semaphore(LLM_PER_AGENT)
    llm_call = make_llm_call(agent_semaphore, reasoner_provider)

    # Patient context extraction and result hook setup
    patient_context = None
    hook = None
    hook_log = []
    if patient_context_enabled:
        validator_llm_call = make_llm_call(agent_semaphore, validator_provider)
        analyzer = PatientContextAnalyzer(validator_llm_call)
        patient_context = await analyzer.extract(full_question)
        if verbose:
            print(f"[PATIENT CONTEXT] {patient_context[:200]}...")

        async def result_hook(result, tool_name, patient_ctx):
            if tool_name not in ("get_entity_relations_with_context",
                                 "check_relation_with_context"):
                return result
            if not isinstance(result, list) or not result:
                return result

            # --- Drug-disease reclassification path ---
            if result[0].get("_requested_relation"):
                requested = result[0]["_requested_relation"]
                for r in result:
                    r.pop("_requested_relation", None)
                before = len(result)
                result = await analyzer.analyze_drug_disease(
                    result, patient_ctx, requested_relation=requested
                )
                after = len(result)
                entry = {"total": before, "requested": requested,
                         "kept": after, "removed": before - after,
                         "path": "drug_disease_reclassify"}
                hook_log.append(entry)
                if verbose:
                    print(f"  [HOOK:drug_disease] {before} relations → "
                          f"reclassify → {after} kept as '{requested}', "
                          f"{before - after} removed/reclassified")
                return _compact_relations(result)

            # --- check_relation_with_context: split drug-disease vs other ---
            if tool_name == "check_relation_with_context":
                dd = [r for r in result if r.get("relation") in DRUG_DISEASE_RELATIONS]
                other = [r for r in result if r.get("relation") not in DRUG_DISEASE_RELATIONS]
                if dd:
                    dd = await analyzer.analyze_drug_disease(dd, patient_ctx)
                if other:
                    other = await analyzer.analyze(other, patient_ctx)
                combined = dd + other
                hook_log.append({"total": len(result), "kept": len(combined),
                                 "path": "check_relation"})
                if verbose:
                    print(f"  [HOOK:check_relation] {len(result)} → {len(combined)} kept")
                return combined

            # --- Standard path: non-drug-disease get_entity_relations_with_context ---
            enriched = [r for r in result if "context_constraints" in r]
            total = len(result)
            if not enriched:
                if verbose:
                    print(f"  [HOOK] {total} relations, 0 enriched")
                hook_log.append({"total": total, "enriched": 0})
                return _compact_relations(result)
            before = len(result)
            result = await analyzer.analyze(result, patient_ctx)
            after = len(result)
            annotated = sum(1 for r in result if "_patient_relevance" in r)
            entry = {"total": before, "enriched": len(enriched),
                     "kept": after, "removed": before - after,
                     "annotated": annotated}
            hook_log.append(entry)
            if verbose:
                print(f"  [HOOK] {before} relations, {len(enriched)} enriched → "
                      f"{after} kept, {before - after} removed, {annotated} annotated")
            return _compact_relations(result)

        hook = result_hook

    tools = {
        "list_relation_types": tool_list_relation_types,
        "search_entity": tool_search_entity,
    }
    if patient_context_enabled:
        tools["get_entity_relations_with_context"] = tool_get_entity_relations_with_context
        tools["check_relation_with_context"] = tool_check_relation_with_context
        prompt = SYSTEM_PROMPT_PATIENT_CONTEXT
    else:
        tools["get_entity_relations"] = tool_get_entity_relations
        tools["check_relation"] = tool_check_relation
        prompt = SYSTEM_PROMPT

    agent = ReactAgent(
        tools=tools,
        llm_call=llm_call,
        system_prompt=prompt,
        max_turns=MAX_TURNS,
        result_hook=hook,
        patient_context=patient_context,
        memory_compression=patient_context_enabled,
        compression_llm_call=llm_call if patient_context_enabled else None,
    )

    react_result = await agent.run(f"Question:\n{full_question}")

    agentic_answer = None
    if react_result.final_answer:
        agentic_answer = react_result.final_answer.get("llm_answer_choice")

    result = {
        "sample_key": sample_key,
        "gold_answer": gold_answer,
        "v1_answer": v1_answer,
        "v3_answer": v3_answer,
        "agentic_answer": agentic_answer,
        "agentic_correct": agentic_answer == gold_answer if agentic_answer else False,
        "v1_correct": v1_answer == gold_answer if v1_answer else False,
        "v3_correct": v3_answer == gold_answer if v3_answer else False,
        "agentic_reasoning": react_result.final_answer.get("reasoning") if react_result.final_answer else None,
        "num_turns": react_result.num_turns,
        "num_tool_calls": len(react_result.tool_calls),
        "tool_calls": react_result.tool_calls,
        "conversation": react_result.conversation,
    }
    result["llm_stats"] = {
        "calls": llm_call.stats["calls"],
        "wait_s": round(llm_call.stats["wait_s"], 1),
        "llm_s": round(llm_call.stats["llm_s"], 1),
    }
    if patient_context_enabled:
        result["patient_context"] = patient_context
        result["hook_log"] = hook_log
    if react_result.compression_log:
        result["compression_log"] = react_result.compression_log

    return result


def print_trace(result: dict):
    """Print detailed trace of a single evaluation run."""
    sk = result["sample_key"]
    status = "CORRECT" if result["agentic_correct"] else "WRONG"
    print(f"\n{'='*60}")
    print(f"[{sk}] agentic={result['agentic_answer']} gold={result['gold_answer']} "
          f"({status}) | turns={result['num_turns']} tools={result['num_tool_calls']}")
    print(f"{'='*60}")

    if result.get("patient_context"):
        print(f"\nPatient context: {result['patient_context'][:300]}")

    if result.get("hook_log"):
        print(f"\nHook log:")
        for i, entry in enumerate(result["hook_log"]):
            print(f"  [{i}] {entry}")

    if result.get("compression_log"):
        print(f"\nCompression log ({len(result['compression_log'])} compressions):")
        total_orig = sum(e["original_chars"] for e in result["compression_log"])
        total_comp = sum(e["compressed_chars"] for e in result["compression_log"])
        for i, entry in enumerate(result["compression_log"]):
            print(f"  [{i}] {entry['original_chars']} → {entry['compressed_chars']} chars "
                  f"({100*(1-entry['compressed_chars']/entry['original_chars']):.0f}% reduction)")
            print(f"      {entry['summary'][:150]}")
        print(f"  Total: {total_orig} → {total_comp} chars ({100*(1-total_comp/total_orig):.0f}% reduction)")

    # Tool call trace
    print(f"\nTool calls:")
    for tc in result["tool_calls"]:
        obs = tc.get("observation_length", "?")
        print(f"  Turn {tc.get('turn','?')}: {tc.get('action','?')} → {obs} chars")

    # Agent reasoning (assistant messages only, skip system)
    print(f"\nAgent reasoning:")
    for msg in result["conversation"]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "assistant":
            preview = content[:500]
            if len(content) > 500:
                preview += "..."
            print(f"\n  [ASSISTANT] {preview}")
        elif role == "observation":
            pr_count = content.count("_patient_relevance")
            print(f"\n  [OBSERVATION] {len(content)} chars"
                  + (f", {pr_count} annotations" if pr_count else ""))

    # Final reasoning
    if result.get("agentic_reasoning"):
        print(f"\nFinal reasoning:\n  {result['agentic_reasoning']}")
    print()


# ============================================================
# Conversation log
# ============================================================

def write_conversation_log(result: dict, log_path: str):
    """Write a detailed markdown conversation log with compression info."""
    lines = []
    sk = result["sample_key"]
    status = "CORRECT" if result["agentic_correct"] else "WRONG"
    lines.append(f"# {sk} Conversation Log\n")
    lines.append(f"**Answer**: {result['agentic_answer']} (gold: {result['gold_answer']}) — {status}")
    lines.append(f"**Turns**: {result['num_turns']}, **Tool calls**: {result['num_tool_calls']}\n")

    if result.get("patient_context"):
        lines.append("## Patient Context\n")
        lines.append(f"```\n{result['patient_context']}\n```\n")

    if result.get("compression_log"):
        lines.append("## Compression Summary\n")
        for i, entry in enumerate(result["compression_log"]):
            pct = 100 * (1 - entry["compressed_chars"] / entry["original_chars"])
            lines.append(f"- [{i}] {entry['original_chars']} → {entry['compressed_chars']} chars "
                         f"({pct:.0f}% reduction)")
        total_orig = sum(e["original_chars"] for e in result["compression_log"])
        total_comp = sum(e["compressed_chars"] for e in result["compression_log"])
        lines.append(f"- **Total: {total_orig} → {total_comp} chars "
                     f"({100*(1-total_comp/total_orig):.0f}% reduction)**\n")

    lines.append("---\n")

    for msg in result["conversation"]:
        role = msg["role"].upper()
        turn = msg.get("turn", "?")
        content = msg["content"]

        if role == "ASSISTANT":
            lines.append(f"## Turn {turn} — ASSISTANT\n")
            lines.append(f"```\n{content}\n```\n")
        elif role == "OBSERVATION":
            compressed = msg.get("compressed")
            if compressed:
                lines.append(f"## Turn {turn} — OBSERVATION ({len(content)} chars → compressed)\n")
                lines.append(f"**Original** ({len(content)} chars):\n")
                lines.append(f"```\n{content}\n```\n")
                lines.append(f"**Compressed** ({len(compressed)} chars):\n")
                lines.append(f"```\n{compressed}\n```\n")
            else:
                lines.append(f"## Turn {turn} — OBSERVATION ({len(content)} chars, kept verbatim)\n")
                lines.append(f"```\n{content}\n```\n")
        elif role == "SYSTEM_NUDGE":
            lines.append(f"## Turn {turn} — SYSTEM NUDGE\n")
            lines.append(f"_{content}_\n")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
                if obj.get("agentic_answer") is not None:
                    finished.add(obj["sample_key"])
            except Exception:
                pass
    return finished


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Agentic KG evaluation. Run batch eval or specific cases.",
        epilog="Examples:\n"
               "  python conditionKgTestAgentic.py                          # batch eval (10 samples)\n"
               "  python conditionKgTestAgentic.py --run medqa_839          # single case\n"
               "  python conditionKgTestAgentic.py --run medqa_839,qa_2027  # multiple cases\n"
               "  python conditionKgTestAgentic.py --run medqa_839 --patient-context\n"
               "  python conditionKgTestAgentic.py --run medqa_839 --output output/test.jsonl\n"
               "  python conditionKgTestAgentic.py --run medqa_839 --patient-context --conv-log output/conv.md\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--patient-context", action="store_true",
                        help="Use context-aware KG tool with patient-specific filtering")
    parser.add_argument("--run", type=str, default=None,
                        help="Run specific cases by sample_key (comma-separated). "
                             "Bypasses EXCLUDE_KEYS and SAMPLE_SIZE.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: auto-named in output/)")
    parser.add_argument("--conv-log", type=str, default=None,
                        help="Write detailed conversation log (markdown) for each case. "
                             "For single case: path to .md file. "
                             "For multiple cases: directory (one .md per case).")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max concurrent agents (default: 3)")
    args = parser.parse_args()
    patient_context_enabled = args.patient_context
    run_keys = [k.strip() for k in args.run.split(",")] if args.run else None

    # Output file
    if args.output:
        output_file = args.output
    elif run_keys:
        suffix = "_patient_context" if patient_context_enabled else ""
        output_file = f"output/run_{'_'.join(run_keys)}{suffix}.jsonl"
    else:
        output_file = ("output/agentic_blind_eval_all_patient_context.jsonl"
                       if patient_context_enabled else "output/agentic_blind_eval_10.jsonl")

    if patient_context_enabled:
        print("Patient context mode ENABLED")

    # Load all records
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Total records in input: {len(records)}")

    # Select records
    if run_keys:
        # --run mode: find specific cases (no EXCLUDE_KEYS filter)
        by_key = {r["sample_key"]: r for r in records}
        selected = []
        for k in run_keys:
            if k in by_key:
                selected.append(by_key[k])
            else:
                print(f"WARNING: {k} not found in {INPUT_FILE}")
        print(f"Running {len(selected)} specified case(s): {[r['sample_key'] for r in selected]}")
        remaining = selected  # no resume for --run mode
        verbose = True
    else:
        # Batch mode: filter, sample, resume
        candidates = [r for r in records if r["sample_key"] not in EXCLUDE_KEYS]
        print(f"After excluding {EXCLUDE_KEYS}: {len(candidates)}")
        random.seed(42)
        selected = random.sample(candidates, min(SAMPLE_SIZE, len(candidates)))
        print(f"Selected {len(selected)} samples")
        finished = load_finished_keys(output_file)
        remaining = [r for r in selected if r["sample_key"] not in finished]
        print(f"Already completed: {len(finished)}, remaining: {len(remaining)}")
        verbose = False

    if not remaining:
        print("Nothing to run.")
    else:
        # Agent-level concurrency: controls how many cases run simultaneously
        agent_gate = asyncio.Semaphore(args.concurrency)

        async def run_with_gate(record):
            async with agent_gate:
                t0 = asyncio.get_event_loop().time()
                result = await evaluate_sample(record, patient_context_enabled, verbose=verbose)
                result["elapsed_s"] = round(asyncio.get_event_loop().time() - t0, 1)
                return result

        tasks = [run_with_gate(r) for r in remaining]

        with open(output_file, "a", encoding="utf-8") as f_out:
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                             disable=verbose):
                result = await coro
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                if verbose:
                    print_trace(result)
                else:
                    status = "correct" if result["agentic_correct"] else "wrong"
                    ls = result.get("llm_stats", {})
                    print(f"  {result['sample_key']}: agentic={result['agentic_answer']} "
                          f"gold={result['gold_answer']} ({status}), "
                          f"turns={result['num_turns']}, tools={result['num_tool_calls']}, "
                          f"time={result.get('elapsed_s', '?')}s | "
                          f"llm_calls={ls.get('calls', '?')}, "
                          f"wait={ls.get('wait_s', '?')}s, "
                          f"llm={ls.get('llm_s', '?')}s")
                # Write conversation log if requested
                if args.conv_log:
                    if args.conv_log.endswith(".md"):
                        log_path = args.conv_log
                    else:
                        os.makedirs(args.conv_log, exist_ok=True)
                        log_path = os.path.join(args.conv_log, f"{result['sample_key']}_conv.md")
                    write_conversation_log(result, log_path)
                    print(f"  Conversation log: {log_path}")

        print(f"\nResults saved to {output_file}")

    # Print comparison table (for all results in the output file)
    all_results = []
    if Path(output_file).exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

    if all_results:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        print(f"\n{'Sample':<20} | {'Gold':>4} | {'v1':>5} | {'v3':>5} | {'Agentic':>8} | {'Turns':>5} | {'Tools':>5} | {'Time':>6}")
        print("-" * 80)

        v1_correct = 0
        v3_correct = 0
        ag_correct = 0
        total = len(all_results)

        for r in sorted(all_results, key=lambda x: x["sample_key"]):
            v1_mark = "ok" if r["v1_correct"] else "X"
            v3_mark = "ok" if r["v3_correct"] else "X"
            ag_mark = "ok" if r["agentic_correct"] else "X"

            v1_str = f"{r['v1_answer']} {v1_mark}"
            v3_str = f"{r['v3_answer']} {v3_mark}"
            ag_str = f"{r['agentic_answer']} {ag_mark}"

            elapsed = r.get("elapsed_s", "")
            time_str = f"{elapsed}s" if elapsed else ""
            print(f"{r['sample_key']:<20} | {r['gold_answer']:>4} | {v1_str:>5} | {v3_str:>5} | {ag_str:>8} | {r['num_turns']:>5} | {r['num_tool_calls']:>5} | {time_str:>6}")

            if r["v1_correct"]:
                v1_correct += 1
            if r["v3_correct"]:
                v3_correct += 1
            if r["agentic_correct"]:
                ag_correct += 1

        print("-" * 80)
        print(f"{'Accuracy':<20} | {'':>4} | {v1_correct}/{total:>2} | {v3_correct}/{total:>2} | {ag_correct}/{total:>5} |")
        print(f"\nv1: {v1_correct}/{total} ({100*v1_correct/total:.0f}%)")
        print(f"v3: {v3_correct}/{total} ({100*v3_correct/total:.0f}%)")
        print(f"Agentic: {ag_correct}/{total} ({100*ag_correct/total:.0f}%)")


if __name__ == "__main__":
    asyncio.run(main())
