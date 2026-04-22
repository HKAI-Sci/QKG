"""Classify Qwen-validator correct-to-wrong (C->W) regressions into:
likely KG-supported / mixed / likely leakage.

Mirrors ``classify_leakage.py`` (which handles W->C) but operates on C->W
records and selects decisive claims accordingly. A C->W record has
``reasoner_correct=True`` and ``final_correct=False``. Analogous to the W->C
case, a CONTRADICTED claim is decisive if it either:

  - contradicts a ``supports=True`` claim about the Reasoner's original
    (correct) answer, pushing the Reasoner away from gold, or
  - contradicts a ``supports=False`` claim about the option that eventually
    became ``final_answer``, un-eliminating the wrong option.

Interpretation of the resulting labels:

  - LIKELY_KG_SUPPORTED C->W: decisive evidence cites a KG entity/relation
    (or a QKG ConstraintItem applicability token). Consistent with the
    benchmark-ceiling reading of Sec 5.2: patient-context-conditioned
    reasoning correctly eliminates an option whose underlying fact is
    nonetheless the MCQ gold.
  - LIKELY_LEAKAGE C->W: decisive evidence is parametric clinical knowledge
    with no KG support. Consistent with a strong-validator-hallucination
    reading: validator-supplied prior knowledge pushed the Reasoner off
    gold without graph grounding.

Outputs (written next to this script):

    leakage_classification_c2w_per_case.csv  -- one row per C->W case
    leakage_classification_c2w_summary.csv   -- per-run counts
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from classify_leakage import label_evidence  # noqa: E402


def is_decisive_c2w(item, reasoner_answer: str, final_answer: str) -> bool:
    if item.get("status") != "CONTRADICTED":
        return False
    opt = (item.get("option") or "").strip()
    if item.get("supports") is True and opt == reasoner_answer:
        return True
    if item.get("supports") is False and opt == final_answer:
        return True
    return False


def case_label_c2w(record):
    decisive = [
        it for it in record["validation_report"]
        if is_decisive_c2w(it, record["reasoner_answer"], record["final_answer"])
    ]
    if not decisive:
        decisive = [
            it for it in record["validation_report"]
            if it.get("status") == "CONTRADICTED"
        ]

    ev_labels = [label_evidence(it.get("evidence", "")) for it in decisive]
    if not ev_labels:
        return "UNCLASSIFIED", False

    has_context = "EV_CONTEXT" in ev_labels
    has_kg = "EV_KG_GROUNDED" in ev_labels
    has_leak = "EV_LEAKAGE" in ev_labels
    has_supp = has_context or has_kg

    if has_supp and has_leak:
        return "MIXED", has_context
    if has_supp:
        return "LIKELY_KG_SUPPORTED", has_context
    if has_leak:
        return "LIKELY_LEAKAGE", has_context
    return "UNCLASSIFIED", has_context


def analyse_run(label: str, path: str):
    if not os.path.isfile(path):
        print(f"[skip] {label}: missing file {path}", file=sys.stderr)
        return None
    recs = [json.loads(line) for line in open(path)]
    n = len(recs)
    c2w_records = [
        r for r in recs if r["reasoner_correct"] and not r["final_correct"]
    ]

    per_case = []
    for r in c2w_records:
        lab, of_which_ctx = case_label_c2w(r)
        per_case.append({
            "run_label": label,
            "sample_key": r["sample_key"],
            "reasoner_answer": r["reasoner_answer"],
            "final_answer": r["final_answer"],
            "gold_answer": r["gold_answer"],
            "case_label": lab,
            "of_which_context_driven": of_which_ctx,
        })

    classes = Counter(c["case_label"] for c in per_case)
    n_kg_supp_ctx = sum(
        1 for c in per_case
        if c["case_label"] == "LIKELY_KG_SUPPORTED" and c["of_which_context_driven"]
    )
    return {
        "run_label": label,
        "input_path": path,
        "N": n,
        "c2w_total": len(c2w_records),
        "kg_supported": classes.get("LIKELY_KG_SUPPORTED", 0),
        "kg_supported_of_which_context_driven": n_kg_supp_ctx,
        "mixed": classes.get("MIXED", 0),
        "leakage": classes.get("LIKELY_LEAKAGE", 0),
        "unclassified": classes.get("UNCLASSIFIED", 0),
        "per_case": per_case,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-pc", default=os.environ.get("QKG_NO_PC_LOG"),
        help="JSONL log for the no-patient-context Qwen-validator run "
             "(or set env var QKG_NO_PC_LOG)")
    ap.add_argument("--with-pc", default=os.environ.get("QKG_WITH_PC_LOG"),
        help="JSONL log for the with-patient-context Qwen-validator run "
             "(or set env var QKG_WITH_PC_LOG)")
    ap.add_argument("--outdir", default=str(Path(__file__).parent))
    args = ap.parse_args()

    if not args.no_pc or not args.with_pc:
        ap.error("--no-pc and --with-pc (or env vars QKG_NO_PC_LOG / "
                 "QKG_WITH_PC_LOG) must point at the validator-run JSONL logs")

    runs = [
        ("KG w/o context (Qwen V, no patient context)", args.no_pc),
        ("QKG w/ context (Qwen V, with patient context)", args.with_pc),
    ]
    summaries = []
    per_case_all = []
    for label, path in runs:
        s = analyse_run(label, path)
        if s is None:
            continue
        summaries.append({k: v for k, v in s.items() if k != "per_case"})
        per_case_all.extend(s["per_case"])
        print(f"\n=== {label} ===")
        for k in ("N", "c2w_total", "kg_supported",
                  "kg_supported_of_which_context_driven",
                  "mixed", "leakage", "unclassified"):
            print(f"  {k:40s} = {s[k]}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summaries:
        sum_path = out_dir / "leakage_classification_c2w_summary.csv"
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nWrote {sum_path}")
    if per_case_all:
        pc_path = out_dir / "leakage_classification_c2w_per_case.csv"
        with open(pc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case_all[0].keys()))
            w.writeheader()
            w.writerows(per_case_all)
        print(f"Wrote {pc_path}")


if __name__ == "__main__":
    main()
