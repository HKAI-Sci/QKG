"""Re-label the C->W cases that the regex heuristic in
paper/figures/classify_leakage_c2w.py left as UNCLASSIFIED, by asking the
Haiku LLM (conf/config.yaml key 'patient-context-llm') to read the
validator's decisive CONTRADICTED evidence and pick one of the four
buckets.

Mirrors classify_unclassified_with_llm.py (W->C) but operates on the C->W
record set (reasoner_correct=True, final_correct=False) and uses the C->W
decisive-claim rule from paper/figures/classify_leakage_c2w.py.

The script only relabels UNCLASSIFIED cases -- buckets that the regex
already decided are kept as-is. It then rewrites the C->W per-case CSV
with LLM columns and recomputes the C->W summary CSV using the merged
labels. Both CSVs are written in place next to the originals in
paper/figures/.

Inputs:
    QKG_NO_PC_LOG   — path to the no-patient-context Qwen-validator JSONL log
    QKG_WITH_PC_LOG — path to the with-patient-context Qwen-validator JSONL log
    conf/config.yaml with a 'patient-context-llm' role + AWS creds

Run:
    python3 classify_unclassified_c2w_with_llm.py [--recompute-only]
"""

import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "paper" / "figures"))
import classify_leakage as ck  # noqa: E402  (label_evidence lives here)
import classify_leakage_c2w as ck_c2w  # noqa: E402  (is_decisive_c2w, case_label_c2w)


def load_config():
    with open(REPO / "conf" / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()

PROVIDER = None
SEM = None
LLMMessageTextOnly = None
LLMRole = None
CONCURRENCY = 5
MAX_RETRIES = 6


def init_llm_provider():
    global PROVIDER, SEM, LLMMessageTextOnly, LLMRole
    if PROVIDER is not None:
        return
    os.environ["AWS_ACCESS_KEY_ID"] = CFG["aws"]["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = CFG["aws"]["secret_access_key"]
    os.environ["AWS_REGION_NAME"] = CFG["aws"]["region"]
    from f1 import config as f1_config
    from f1.common.schema import LLMConfig, LLMMessageTextOnly as _LMT, LLMRole as _LR
    from f1.common.llm_provider import LLMProviderFactory
    LLMMessageTextOnly = _LMT
    LLMRole = _LR
    llm_cfg = LLMConfig(**f1_config["patient-context-llm"])
    PROVIDER = LLMProviderFactory.create_instance(llm_cfg)
    SEM = asyncio.Semaphore(CONCURRENCY)


NO_PC_LOG = os.environ.get("QKG_NO_PC_LOG", "")
WITH_PC_LOG = os.environ.get("QKG_WITH_PC_LOG", "")
RUNS = [
    ("KG w/o context (Qwen V, no patient context)", NO_PC_LOG),
    ("QKG w/ context (Qwen V, with patient context)", WITH_PC_LOG),
]
FIG_DIR = REPO / "paper" / "figures"
OUT_PER_CASE = FIG_DIR / "leakage_classification_c2w_per_case.csv"
OUT_SUMMARY = FIG_DIR / "leakage_classification_c2w_summary.csv"


PROMPT = """You are auditing a medical-QA validator's reasoning to determine why a correct-to-wrong (C->W) regression happened.

The Reasoner originally picked the gold answer, but after the Validator emitted per-claim CONTRADICTED status with a free-text "evidence" string, the Reasoner switched to a wrong option. We classify each C->W regression into one of four buckets, depending on what the *decisive* CONTRADICTED evidence rests on:

  LIKELY_KG_SUPPORTED : the contradiction's stated basis is a specific knowledge-graph entity / relation / edge, OR a patient-context-conditioned QKG applicability annotation (AVOID / RECOMMENDED / CAUTION / ConstraintItem / safety judgment). Per Section 5.2 of the paper, KG-supported C->W reflects QKG (or KG alone) correctly eliminating an option on graph-grounded or patient-context grounds, where the eliminated option happens to be the MCQ gold --- i.e., weak benchmark gold, not a validator hallucination.
  LIKELY_LEAKAGE      : the contradiction's stated basis is the validator's own clinical or guideline knowledge, with no relevant KG edge cited (often signalled by "KG lacks ... However, medically/clinically/per AHA guidelines ...", or by pure parametric assertions with no KG citation). Under Section 5.2, LEAKAGE C->W reflects validator-supplied prior knowledge pushing the Reasoner off gold.
  MIXED               : decisive evidence has both KG-supported and leakage signals (e.g., one decisive claim cites a KG edge, another concedes a KG gap and pivots to guideline knowledge).
  UNCLASSIFIED        : the evidence is truly indeterminate --- no KG citation and no clinical-knowledge assertion that you can reasonably attribute the contradiction to.

A C->W claim is "decisive" if it either (i) contradicts the option the Reasoner originally chose (the gold), or (ii) un-eliminates the option that eventually became the final (wrong) answer. These are the claims that drove the C->W switch. Look at the decisive evidence strings below and pick ONE label.

Sample key: {sample_key}
Reasoner's original answer (=gold): {reasoner_answer}
Final answer (after reconsider, wrong): {final_answer}
Gold answer: {gold_answer}
{patient_context_block}
Decisive CONTRADICTED claim(s):

{decisive_block}

Respond with a single JSON object on one line, no markdown fence:
{{"label": "<one of LIKELY_KG_SUPPORTED|LIKELY_LEAKAGE|MIXED|UNCLASSIFIED>", "reasoning": "<one short sentence justifying the label, citing the most decisive evidence phrase>"}}
"""


def build_prompt(record):
    decisive = [it for it in record["validation_report"]
                if ck_c2w.is_decisive_c2w(it, record["reasoner_answer"], record["final_answer"])]
    if not decisive:
        decisive = [it for it in record["validation_report"]
                    if it.get("status") == "CONTRADICTED"]
    block_parts = []
    for it in decisive:
        block_parts.append(
            f"[option {it.get('option')}]  supports={it.get('supports')}\n"
            f"  evidence: {(it.get('evidence') or '').strip()}"
        )
    decisive_block = "\n\n".join(block_parts) if block_parts else "(none)"
    pc = record.get("patient_context")
    pc_block = f"Patient context (when available): {str(pc).strip()}\n" if pc else ""
    return PROMPT.format(
        sample_key=record["sample_key"],
        reasoner_answer=record["reasoner_answer"],
        final_answer=record["final_answer"],
        gold_answer=record["gold_answer"],
        patient_context_block=pc_block,
        decisive_block=decisive_block,
    )


def parse_response(text: str):
    txt = (text or "").strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.startswith("json"):
            txt = txt[4:].strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in response: {text[:300]}")
    obj = json.loads(txt[start:end + 1])
    label = (obj.get("label") or "").strip().upper()
    reasoning = (obj.get("reasoning") or "").strip()
    valid = {"LIKELY_KG_SUPPORTED", "LIKELY_LEAKAGE", "MIXED", "UNCLASSIFIED"}
    if label not in valid:
        raise ValueError(f"label '{label}' not in {valid}")
    return label, reasoning


async def classify_one(record):
    prompt = build_prompt(record)
    async with SEM:
        for attempt in range(MAX_RETRIES):
            try:
                response = await PROVIDER.async_chat_completion(
                    messages=[LLMMessageTextOnly(role=LLMRole.USER, content=prompt)],
                    temperature=0,
                    max_tokens=400,
                )
                if not response:
                    raise ValueError("empty LLM response")
                return parse_response(response)
            except Exception as e:
                wait = min(60, 2 ** attempt) + random.random()
                msg = str(e).replace("\n", " ")[:200]
                print(f"  ⚠️ {record['sample_key']} attempt {attempt + 1}: {msg}; sleeping {wait:.1f}s",
                      flush=True)
                if attempt == MAX_RETRIES - 1:
                    return "UNCLASSIFIED", f"LLM_ERROR: {msg}"
                await asyncio.sleep(wait)
    return "UNCLASSIFIED", "LLM_ERROR: exhausted"


async def relabel_run(label: str, path: str):
    init_llm_provider()
    if not os.path.isfile(path):
        print(f"[skip] {label}: missing file {path}", file=sys.stderr)
        return []
    recs = [json.loads(l) for l in open(path)]
    c2w = [r for r in recs if r["reasoner_correct"] and not r["final_correct"]]

    rows = []
    to_relabel = []
    for r in c2w:
        regex_label, ctx_flag = ck_c2w.case_label_c2w(r)
        rows.append({
            "run_label": label,
            "sample_key": r["sample_key"],
            "reasoner_answer": r["reasoner_answer"],
            "final_answer": r["final_answer"],
            "gold_answer": r["gold_answer"],
            "regex_label": regex_label,
            "of_which_context_driven": ctx_flag,
            "llm_label": "",
            "llm_reasoning": "",
            "final_label": regex_label,
            "label_source": "regex",
        })
        if regex_label == "UNCLASSIFIED":
            to_relabel.append((len(rows) - 1, r))

    print(f"[{label}] C->W={len(c2w)}; UNCLASSIFIED={len(to_relabel)}; relabelling with LLM ...",
          flush=True)

    async def task(idx, rec):
        lbl, why = await classify_one(rec)
        rows[idx]["llm_label"] = lbl
        rows[idx]["llm_reasoning"] = why
        rows[idx]["final_label"] = lbl
        rows[idx]["label_source"] = "llm"
        print(f"  {rec['sample_key']:>24s}  ->  {lbl}", flush=True)

    await asyncio.gather(*[task(i, r) for i, r in to_relabel])
    return rows


def write_per_case(rows):
    fields = ["run_label", "sample_key", "reasoner_answer", "final_answer",
              "gold_answer", "regex_label", "of_which_context_driven",
              "llm_label", "llm_reasoning", "final_label", "label_source"]
    with open(OUT_PER_CASE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_PER_CASE}  ({len(rows)} rows)")


def recompute_summary(rows):
    from collections import Counter
    summaries = []
    for label, path in RUNS:
        run_rows = [r for r in rows if r["run_label"] == label]
        cnt = Counter(r["final_label"] for r in run_rows)
        n_kg = cnt.get("LIKELY_KG_SUPPORTED", 0)
        n_kg_ctx = sum(1 for r in run_rows
                       if r["final_label"] == "LIKELY_KG_SUPPORTED"
                       and str(r["of_which_context_driven"]).lower() in {"true", "1"})
        summaries.append({
            "run_label": label,
            "input_path": path,
            "N": 0,
            "c2w_total": len(run_rows),
            "kg_supported": n_kg,
            "kg_supported_of_which_context_driven": n_kg_ctx,
            "mixed": cnt.get("MIXED", 0),
            "leakage": cnt.get("LIKELY_LEAKAGE", 0),
            "unclassified": cnt.get("UNCLASSIFIED", 0),
            "llm_relabelled": sum(1 for r in run_rows if r["label_source"] == "llm"),
        })
    if summaries:
        with open(OUT_SUMMARY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"Wrote {OUT_SUMMARY}")
    for s in summaries:
        print(f"\n=== {s['run_label']} ===")
        for k in ("c2w_total", "kg_supported", "kg_supported_of_which_context_driven",
                  "mixed", "leakage", "unclassified", "llm_relabelled"):
            print(f"  {k:40s} = {s[k]}")


def load_per_case_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["of_which_context_driven"] = (
                str(r.get("of_which_context_driven", "")).lower() in {"true", "1"}
            )
            rows.append(r)
    return rows


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute-only", action="store_true",
                    help="Skip the LLM re-label pass; rebuild the summary CSV "
                         "from the existing per-case CSV.")
    args = ap.parse_args()

    if args.recompute_only:
        if not OUT_PER_CASE.is_file():
            sys.exit(f"--recompute-only requires {OUT_PER_CASE} to exist")
        all_rows = load_per_case_csv(OUT_PER_CASE)
        print(f"Loaded {len(all_rows)} rows from {OUT_PER_CASE}", flush=True)
    else:
        all_rows = []
        for label, path in RUNS:
            all_rows.extend(await relabel_run(label, path))
        write_per_case(all_rows)

    recompute_summary(all_rows)


if __name__ == "__main__":
    asyncio.run(main())
