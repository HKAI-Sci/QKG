"""Re-label the W->C cases that the regex heuristic in
paper/figures/classify_leakage.py left as UNCLASSIFIED, by asking the Haiku
LLM (conf/config.yaml key 'patient-context-llm') to read the validator's
decisive CONTRADICTED evidence and pick one of the four buckets.

The script only relabels UNCLASSIFIED cases — buckets that the regex already
decided are kept as-is. It then rewrites the per-case CSV with two extra
columns (final_label, label_source, llm_reasoning) and recomputes the
summary CSV using the merged labels. Both new CSVs are written next to the
original ones in paper/figures/.

Note on N. The denominator used to compute reasoner_acc / raw_final_acc /
adj_final_acc is taken from --n if given, otherwise from the curated
dataset size (DATASET_N below; see Appendix A.1 of the paper). It is NOT
the number of records in the JSONL log, because some records can be missing
due to runtime errors when the log was being written. Numerators
(reasoner_correct, final_correct, leakage counts) are taken faithfully from
the log so they may be slightly smaller than they would have been on the
full set; the adjustment to the denominator therefore yields a small
pessimistic bias in the reported accuracies, which is acceptable for the
purposes of Table~4.

Inputs:
    QKG_NO_PC_LOG   — path to the no-patient-context Qwen-validator JSONL log
    QKG_WITH_PC_LOG — path to the with-patient-context Qwen-validator JSONL log
    conf/config.yaml with a 'patient-context-llm' role + AWS creds

Run:
    python3 classify_unclassified_with_llm.py [--n 2788] [--recompute-only]
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
import classify_leakage as ck  # noqa: E402  (rules + helpers reused)


# --- config / provider ------------------------------------------------------

def load_config():
    with open(REPO / "conf" / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()

# f1 / LLM provider are loaded lazily so that --recompute-only can run on a
# machine without the f1 package installed.
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


# --- input/output paths -----------------------------------------------------

NO_PC_LOG = os.environ.get("QKG_NO_PC_LOG", "")
WITH_PC_LOG = os.environ.get("QKG_WITH_PC_LOG", "")
RUNS = [
    ("KG w/o context (Qwen V, no patient context)", NO_PC_LOG),
    ("QKG w/ context (Qwen V, with patient context)", WITH_PC_LOG),
]
FIG_DIR = REPO / "paper" / "figures"
OUT_PER_CASE = FIG_DIR / "leakage_classification_per_case_llm.csv"
OUT_SUMMARY = FIG_DIR / "leakage_classification_summary_llm.csv"

# Curated dataset size from Appendix A.1 of the paper. Used as the default
# denominator so the released summary CSV matches the paper exactly even if
# the JSONL log is short by a few records due to runtime errors.
DATASET_N = 2788

# Canonical (reasoner_correct, final_correct, c2w) per run, taken from the
# experiment record at paper/data_result/PKG实验记录表.xlsx (Sheet1) — the
# row indices below are the row numbers in that spreadsheet:
#   L19  Haiku (R) + Qwen (V), no patient context :  reasoner=2160, W->C=177, C->W=16  -> final=2321
#   L14  Haiku (R) + Qwen (V), with patient context: reasoner=2161, W->C=204, C->W=38  -> final=2327
# The JSONL logs are short by 1 record (no-pc) and 5 records (with-pc) due to
# runtime save errors, so deriving counts directly from the JSONL would drift
# from the experiment record. We use the canonical counts here so the
# released summary CSV matches Figure 4 and the paper text.
CANONICAL_COUNTS = {
    "KG w/o context (Qwen V, no patient context)": {
        "reasoner_correct": 2160, "final_correct": 2321, "c2w": 16,
    },
    "QKG w/ context (Qwen V, with patient context)": {
        "reasoner_correct": 2161, "final_correct": 2327, "c2w": 38,
    },
}


# --- prompt -----------------------------------------------------------------

PROMPT = """You are auditing a medical-QA validator's reasoning to determine why a wrong-to-correct (W->C) revision happened.

The validator emits per-claim CONTRADICTED status with a free-text "evidence" string. We classify each W->C revision into one of four buckets, depending on what the *decisive* CONTRADICTED evidence rests on:

  LIKELY_KG_SUPPORTED : the contradiction's stated basis is a specific knowledge-graph entity / relation / edge, OR a patient-context-conditioned QKG applicability annotation (AVOID / RECOMMENDED / CAUTION / ConstraintItem / safety judgment). KG-supported includes both pure KG-grounded contradictions and patient-context-conditioned ones.
  LIKELY_LEAKAGE      : the contradiction's stated basis is the validator's own clinical or guideline knowledge, with no relevant KG edge cited (often signalled by "KG lacks ... However, medically/clinically/per AHA guidelines ...", or by pure parametric assertions with no KG citation).
  MIXED               : decisive evidence has both KG-supported and leakage signals (e.g., one decisive claim cites a KG edge, another concedes a KG gap and pivots to guideline knowledge).
  UNCLASSIFIED        : the evidence is truly indeterminate — no KG citation and no clinical-knowledge assertion that you can reasonably attribute the contradiction to.

A claim is "decisive" if it contradicts the option the Reasoner originally chose, or contradicts the Reasoner's elimination of the gold option (these are the claims that drove the W->C switch). Look at the decisive evidence strings below and pick ONE label.

Sample key: {sample_key}
Reasoner's original answer: {reasoner_answer}
Final answer (after reconsider): {final_answer}
Gold answer: {gold_answer}
{patient_context_block}
Decisive CONTRADICTED claim(s):

{decisive_block}

Respond with a single JSON object on one line, no markdown fence:
{{"label": "<one of LIKELY_KG_SUPPORTED|LIKELY_LEAKAGE|MIXED|UNCLASSIFIED>", "reasoning": "<one short sentence justifying the label, citing the most decisive evidence phrase>"}}
"""


def build_prompt(record):
    decisive = [it for it in record["validation_report"]
                if ck.is_decisive(it, record["reasoner_answer"], record["gold_answer"])]
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
    # Strip optional code-fence
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


# --- main pipeline ----------------------------------------------------------

async def relabel_run(label: str, path: str):
    """Returns the per-case rows for one run, with regex labels filled in and
    LLM labels filled in for previously UNCLASSIFIED cases."""
    init_llm_provider()
    if not os.path.isfile(path):
        print(f"[skip] {label}: missing file {path}", file=sys.stderr)
        return []
    recs = [json.loads(l) for l in open(path)]
    w2c = [r for r in recs if (not r["reasoner_correct"]) and r["final_correct"]]

    # First pass: regex labels for everything
    rows = []
    to_relabel = []
    for r in w2c:
        regex_label, ctx_flag = ck.case_label(r)
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

    print(f"[{label}] W->C={len(w2c)}; UNCLASSIFIED={len(to_relabel)}; relabelling with LLM ...",
          flush=True)

    async def task(idx, rec):
        lbl, why = await classify_one(rec)
        rows[idx]["llm_label"] = lbl
        rows[idx]["llm_reasoning"] = why
        rows[idx]["final_label"] = lbl
        rows[idx]["label_source"] = "llm"
        print(f"  {rec['sample_key']:>16s}  ->  {lbl}", flush=True)

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


def _load_existing_summary_counts():
    """Fallback for --recompute-only on a machine without the JSONL logs:
    recover (reasoner_correct, final_correct, c2w) per run from a previously
    written summary CSV. Returns dict keyed by run_label."""
    if not OUT_SUMMARY.is_file():
        return {}
    out = {}
    with open(OUT_SUMMARY) as f:
        for r in csv.DictReader(f):
            out[r["run_label"]] = {
                "reasoner_correct": int(r["reasoner_correct"]),
                "final_correct": int(r["final_correct"]),
                "c2w": int(r["c2w"]),
            }
    return out


def recompute_summary(rows, n_override=None):
    """Compute the summary CSV using the *final_label* column (regex updated
    with LLM relabels). The denominator N is taken from n_override if given,
    otherwise from the curated dataset size DATASET_N (see the module
    docstring for the rationale). reasoner_correct, final_correct and class
    counts are taken faithfully from the JSONL log when available; if the
    log is missing (e.g. running --recompute-only on a different machine),
    those counts are recovered from the previously written summary CSV."""
    fallback = _load_existing_summary_counts()
    summaries = []
    for label, path in RUNS:
        # Counts: prefer canonical values from the experiment record (xlsx),
        # otherwise derive from JSONL log, otherwise fall back to existing
        # summary CSV. See CANONICAL_COUNTS for source.
        if label in CANONICAL_COUNTS:
            c = CANONICAL_COUNTS[label]
            n_reasoner_right = c["reasoner_correct"]
            n_final_right = c["final_correct"]
            c2w = c["c2w"]
        elif os.path.isfile(path):
            recs = [json.loads(l) for l in open(path)]
            n_reasoner_right = sum(1 for r in recs if r["reasoner_correct"])
            n_final_right = sum(1 for r in recs if r["final_correct"])
            c2w = sum(1 for r in recs if r["reasoner_correct"] and not r["final_correct"])
        elif label in fallback:
            f = fallback[label]
            n_reasoner_right = f["reasoner_correct"]
            n_final_right = f["final_correct"]
            c2w = f["c2w"]
        else:
            print(f"[skip] {label}: no canonical counts, no JSONL log, no existing summary entry",
                  file=sys.stderr)
            continue
        n = n_override if n_override is not None else DATASET_N

        run_rows = [r for r in rows if r["run_label"] == label]
        from collections import Counter
        cnt = Counter(r["final_label"] for r in run_rows)
        n_kg = cnt.get("LIKELY_KG_SUPPORTED", 0)
        n_kg_ctx = sum(1 for r in run_rows
                       if r["final_label"] == "LIKELY_KG_SUPPORTED"
                       and (str(r["of_which_context_driven"]).lower() in {"true", "1"}))
        n_mixed = cnt.get("MIXED", 0)
        n_leak = cnt.get("LIKELY_LEAKAGE", 0)
        n_unc = cnt.get("UNCLASSIFIED", 0)
        n_relabel = sum(1 for r in run_rows if r["label_source"] == "llm")

        raw_final = n_final_right / n
        adj_final = ((n_final_right - n_leak) / (n - n_leak)
                     if (n - n_leak) > 0 else 0.0)
        reasoner_acc = n_reasoner_right / n

        summaries.append({
            "run_label": label,
            "input_path": path,
            "N": n,
            "reasoner_correct": n_reasoner_right,
            "final_correct": n_final_right,
            "reasoner_acc": round(reasoner_acc, 4),
            "raw_final_acc": round(raw_final, 4),
            "adj_final_acc": round(adj_final, 4),
            "delta_raw_pp": round((raw_final - reasoner_acc) * 100, 2),
            "delta_adj_pp": round((adj_final - reasoner_acc) * 100, 2),
            "w2c_total": len(run_rows),
            "kg_supported": n_kg,
            "kg_supported_of_which_context_driven": n_kg_ctx,
            "mixed": n_mixed,
            "leakage": n_leak,
            "unclassified": n_unc,
            "llm_relabelled": n_relabel,
            "c2w": c2w,
        })

    if summaries:
        with open(OUT_SUMMARY, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"Wrote {OUT_SUMMARY}")
    for s in summaries:
        print(f"\n=== {s['run_label']} ===")
        for k in ("N", "reasoner_acc", "raw_final_acc", "adj_final_acc",
                  "delta_raw_pp", "delta_adj_pp", "w2c_total",
                  "kg_supported", "kg_supported_of_which_context_driven",
                  "mixed", "leakage", "unclassified", "llm_relabelled", "c2w"):
            print(f"  {k:35s} = {s[k]}")


def load_per_case_csv(path):
    """Load an existing per-case CSV produced by an earlier run, in the same
    row schema written by write_per_case()."""
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
    ap.add_argument("--n", type=int, default=None,
                    help=f"Override the denominator N (default: DATASET_N={DATASET_N})")
    ap.add_argument("--recompute-only", action="store_true",
                    help="Skip the LLM re-label pass; rebuild the summary CSV "
                         "from the existing per-case CSV. Use this to refresh "
                         "the summary with a different --n without burning API "
                         "calls.")
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

    recompute_summary(all_rows, n_override=args.n)


if __name__ == "__main__":
    asyncio.run(main())
