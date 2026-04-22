"""Classify Qwen-validator wrong-to-correct (W->C) revisions into:
likely KG-supported / mixed / likely leakage,
and compute a leakage-adjusted final accuracy.

The classification operates only on the validator's *decisive* CONTRADICTED
evidence strings (those whose flipping the Reasoner reconsidered against). The
rules below are intentionally transparent regex patterns so that any reviewer
can audit the per-case labels and the resulting summary numbers.

Inputs
------
Two JSONL log files produced by the reasoner-validator pipeline, where each
record has at least the fields:
    sample_key, gold_answer, reasoner_answer, final_answer,
    reasoner_correct, final_correct, validation_report (list of items
    {option, claim, supports, status, evidence}).

Outputs (written next to this script)
-------------------------------------
    leakage_classification_per_case.csv    one row per W->C case with its label
    leakage_classification_summary.csv     per-run counts and adjusted accuracy

Heuristic
---------
For each decisive evidence string we detect four signals via regex:
    KG_SUPPORT  : evidence cites a KG entity/relation/edge as the basis of the
                  contradiction (e.g., "KG confirms ... indication relation",
                  "KG entity 30494 has direct positive phenotype relations to
                  Acute kidney injury").
    KG_GAP      : evidence concedes that the KG had no relevant edge for the
                  question (e.g., "KG lacks", "returned no", "empty list").
    PARAMETRIC  : evidence asserts external clinical/guideline knowledge
                  (e.g., "Medically", "Clinically", "AHA guidelines",
                  "standard of care").
    CONTEXT     : evidence uses a QKG-specific applicability token. We match
                  these tokens narrowly so that generic patient mentions in
                  free text (e.g., "this patient's") do NOT count. The
                  matched tokens are: case-sensitive AVOID / RECOMMENDED /
                  CAUTION (which appear as uppercase ConstraintItem labels in
                  the validator output) and case-insensitive "ConstraintItem"
                  and "safety judgment". The looser token "applicability" was
                  deliberately dropped because it appears in ordinary
                  clinical-trial prose without implying a QKG ConstraintItem
                  hit.

Each decisive evidence string is then labelled:
    EV_CONTEXT      : CONTEXT present (treated here as a KG-supported decision
                      because the QKG token only exists when a ConstraintItem
                      record was retrieved)
    EV_KG_GROUNDED  : KG_SUPPORT present, CONTEXT absent, and not dominated by
                      a KG_GAP+PARAMETRIC pivot
    EV_LEAKAGE      : the contradiction's stated basis is parametric clinical
                      knowledge with no KG support, captured as either
                      (KG_GAP and PARAMETRIC and not CONTEXT) or
                      (PARAMETRIC and not KG_SUPPORT and not CONTEXT and
                       not KG_GAP)  [pure-parametric variant]
    EV_UNCLASSIFIED : none of the above

Per-case label (three buckets, plus UNCLASSIFIED):
    LIKELY_KG_SUPPORTED : any decisive evidence is EV_CONTEXT or
                          EV_KG_GROUNDED, and no decisive evidence is
                          EV_LEAKAGE
    MIXED               : at least one EV_CONTEXT/EV_KG_GROUNDED AND at least
                          one EV_LEAKAGE
    LIKELY_LEAKAGE      : at least one EV_LEAKAGE and no EV_CONTEXT and no
                          EV_KG_GROUNDED
    UNCLASSIFIED        : none of the above (no signal in any decisive evidence)

The per-case CSV also retains a finer "of_which_context_driven" flag so that
the proportion of patient-context-conditioned wins inside the KG-supported
bucket can be inspected without re-running the classifier.

Leakage-adjusted final accuracy
-------------------------------
    raw_final_acc    = (# final_correct) / N
    adj_final_acc    = (# final_correct - # likely_leakage_W->C)
                       / (N - # likely_leakage_W->C)
This drops the W->C cases labelled LIKELY_LEAKAGE from both numerator and
denominator, reporting accuracy on the subset of samples whose final answer
is not suspected of resting on validator-knowledge leakage. (An alternative
that keeps the original denominator and only subtracts from the numerator
gives a lower bound on accuracy under leakage; we report only the
subset-removal form for clarity, but the per-case CSV makes the alternative
trivial to compute.)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# --- regex patterns ---------------------------------------------------------

KG_SUPPORT_PATTERNS = [
    r"\bkg\s+(confirms|shows|lists|states|indicates|annotates|reports|reveals|provides|maps|associates|links|connects|flags|marks|explicitly)\b",
    r"\bkg\s+(?:explicitly\s+)?(supports|contradicts|associates|links|targets|annotates)\b",
    r"\b(entity|entities|node)\s+\d+\b",
    r"\bindex\s+\d+\b",
    r"\b(direct|explicit)\s+(indication|contraindication|drug_effect|drug\s+effect|phenotype|protein|relation|edge|link|annotation)\b",
    r"\b(indication|contraindication|drug_effect|drug\s+effect|phenotype)\s+(relation|annotation|link|edge|data)\b",
    r"\bin\s+the\s+(knowledge\s+graph|graph|kg)\b",
    r"\baccording\s+to\s+(the\s+)?kg\b",
    r"\bkg\s+(?:explicitly\s+)?(?:has|contains|includes|stores)\b",
    r"\bkg\s+(?:explicitly\s+)?(?:identifies|identified)\b",
    r"\bkg\s+search\s+(confirmed|identified|returned)\b",
    r"\bkg\s+query\b",
    r"\bprimekg\b",
    r"\bdrugbank\b",
]

KG_GAP_PATTERNS = [
    r"\bkg\s+lacks\b",
    r"\bkg\s+does\s+not\b",
    r"\bkg\s+contains\s+no\b",
    r"\bkg\s+has\s+no\b",
    r"\bno\s+edges?\s+(linking|linked|to|between|connecting)\b",
    r"\bno\s+relations?\b",
    r"\b(returned|return)\s+no\b",
    r"\blacks?\s+specific\b",
    r"\bno\s+clinical\s+guideline\b",
    r"\bdid\s+not\s+yield\b",
    r"\bdid\s+not\s+return\b",
    r"\bno\s+relevant\s+(relation|edge|data|annotation|guideline)\b",
    r"\bno\s+specific\s+(data|relation|edge|annotation)\b",
    r"\bno\s+direct\s+relation\b",
    r"\bempty\s+list\b",
    r"\bnot\s+in\s+the\s+kg\b",
    r"\bno\s+(disease\s+|drug\s+|phenotype\s+)?entities?\b",
    r"\bno\s+(supporting|contradicting)\s+evidence\b",
    r"\bno\s+(scheduling|dosing|guideline)\s+(data|information)\b",
    r"\blacks?\s+(coverage|guidelines?|annotations?)\b",
    r"\bnot\s+contain\s+(specific\s+)?(entities|data|relations|annotations|edges|guidelines?)\b",
]

PARAMETRIC_PATTERNS = [
    r"\bmedically\b",
    r"\bclinically\b",
    r"\bhowever[,\s]",
    r"\bguidelines?\b",
    r"\bstandard\s+practice\b",
    r"\bstandard\s+of\s+care\b",
    r"\bin\s+clinical\s+practice\b",
    r"\bclinical\s+guideline\b",
    r"\bper\s+(?:current|standard|aha|cdc|acip|uspstf|nice|esc|acc|asco|usp|asgct|atp|kdigo)\b",
    r"\baccording\s+to\b",
    r"\b(aha|cdc|acip|uspstf|nice|esc|acc|asco|usp|asgct|atp|kdigo|ada|asco|asn)\b",
    r"\bestablished\s+(medical|clinical|psychiatric|pharmacological|guideline)\b",
    r"\bclinical\s+(consensus|teaching|practice)\b",
    r"\b(first[-\s]line|second[-\s]line|gold[-\s]standard)\s+(treatment|therapy|management|recommendation)\b",
]

# CONTEXT is intentionally narrow: only tokens that are emitted by the
# QKG ConstraintItem layer when a patient-conditioned applicability annotation
# is retrieved. The first three are matched case-sensitively because they
# appear as uppercase ConstraintItem labels in the validator output;
# lowercase "avoid"/"recommended"/"caution" in free clinical prose does not
# count.
CONTEXT_PATTERNS_CASE_SENSITIVE = [
    r"\bAVOID\b",
    r"\bRECOMMENDED\b",
    r"\bCAUTION\b",
]
CONTEXT_PATTERNS_CASE_INSENSITIVE = [
    r"\bconstraintitem\b",
    r"\bsafety\s+judgment\b",
]
# Note: "applicability" was previously included but was dropped because it
# appears in ordinary clinical-trial prose (e.g., "evidence-based applicability
# for this trial") and produced one false positive in the no-context run
# (qa_2554) where no QKG ConstraintItem layer was active.

KG_SUPPORT_RE = re.compile("|".join(KG_SUPPORT_PATTERNS), re.IGNORECASE)
KG_GAP_RE = re.compile("|".join(KG_GAP_PATTERNS), re.IGNORECASE)
PARAMETRIC_RE = re.compile("|".join(PARAMETRIC_PATTERNS), re.IGNORECASE)
CONTEXT_RE_CS = re.compile("|".join(CONTEXT_PATTERNS_CASE_SENSITIVE))
CONTEXT_RE_CI = re.compile("|".join(CONTEXT_PATTERNS_CASE_INSENSITIVE), re.IGNORECASE)


# --- per-evidence label -----------------------------------------------------

def label_evidence(evidence: str) -> str:
    """Return EV_CONTEXT / EV_KG_GROUNDED / EV_LEAKAGE / EV_UNCLASSIFIED."""
    ev = evidence or ""
    has_context = bool(CONTEXT_RE_CS.search(ev) or CONTEXT_RE_CI.search(ev))
    has_kg_support = bool(KG_SUPPORT_RE.search(ev))
    has_kg_gap = bool(KG_GAP_RE.search(ev))
    has_param = bool(PARAMETRIC_RE.search(ev))

    if has_context:
        return "EV_CONTEXT"
    if has_kg_gap and has_param:
        return "EV_LEAKAGE"
    if has_param and not has_kg_support and not has_kg_gap:
        return "EV_LEAKAGE"
    if has_kg_support:
        return "EV_KG_GROUNDED"
    return "EV_UNCLASSIFIED"


# --- decisive-claim selection ----------------------------------------------

def is_decisive(item, reasoner_answer: str, gold_answer: str) -> bool:
    """A CONTRADICTED claim is decisive if it contradicts the option the
    Reasoner originally chose, or contradicts the Reasoner's elimination of
    the gold option. These are the claims that drive the reconsider step."""
    if item.get("status") != "CONTRADICTED":
        return False
    opt = (item.get("option") or "").strip()
    if item.get("supports") is True and opt == reasoner_answer:
        return True
    if item.get("supports") is False and opt == gold_answer:
        return True
    return False


def case_label(record):
    """Aggregate decisive evidence labels into a per-case label.

    Returns (case_label, of_which_context_driven), where case_label is one of
    LIKELY_KG_SUPPORTED / MIXED / LIKELY_LEAKAGE / UNCLASSIFIED, and
    of_which_context_driven is True iff at least one decisive evidence string
    matched the QKG-specific CONTEXT tokens (AVOID/RECOMMENDED/CAUTION/
    applicability/ConstraintItem/safety judgment).
    """
    decisive = [it for it in record["validation_report"]
                if is_decisive(it, record["reasoner_answer"], record["gold_answer"])]
    if not decisive:
        decisive = [it for it in record["validation_report"]
                    if it.get("status") == "CONTRADICTED"]

    ev_labels = [label_evidence(it.get("evidence", "")) for it in decisive]
    if not ev_labels:
        return "UNCLASSIFIED", False

    has_context = "EV_CONTEXT" in ev_labels
    has_kg = "EV_KG_GROUNDED" in ev_labels
    has_leak = "EV_LEAKAGE" in ev_labels
    has_supp = has_context or has_kg

    if has_supp and has_leak:
        label = "MIXED"
    elif has_supp:
        label = "LIKELY_KG_SUPPORTED"
    elif has_leak:
        label = "LIKELY_LEAKAGE"
    else:
        label = "UNCLASSIFIED"
    return label, has_context


# --- driver -----------------------------------------------------------------

def analyse_run(label: str, path: str):
    if not os.path.isfile(path):
        print(f"[skip] {label}: missing file {path}", file=sys.stderr)
        return None
    recs = [json.loads(line) for line in open(path)]
    n = len(recs)
    n_reasoner_right = sum(1 for r in recs if r["reasoner_correct"])
    n_final_right = sum(1 for r in recs if r["final_correct"])
    w2c_records = [r for r in recs if (not r["reasoner_correct"]) and r["final_correct"]]
    c2w = sum(1 for r in recs if r["reasoner_correct"] and not r["final_correct"])

    per_case = []
    for r in w2c_records:
        lab, of_which_ctx = case_label(r)
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
    n_leak = classes["LIKELY_LEAKAGE"]
    n_kg_supp = classes.get("LIKELY_KG_SUPPORTED", 0)
    n_mixed = classes.get("MIXED", 0)
    n_unc = classes.get("UNCLASSIFIED", 0)
    n_kg_supp_ctx = sum(1 for c in per_case
                        if c["case_label"] == "LIKELY_KG_SUPPORTED"
                        and c["of_which_context_driven"])
    raw_final = n_final_right / n
    adj_final = (n_final_right - n_leak) / (n - n_leak) if (n - n_leak) > 0 else 0.0
    reasoner_acc = n_reasoner_right / n

    return {
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
        "w2c_total": len(w2c_records),
        "kg_supported": n_kg_supp,
        "kg_supported_of_which_context_driven": n_kg_supp_ctx,
        "mixed": n_mixed,
        "leakage": n_leak,
        "unclassified": n_unc,
        "c2w": c2w,
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
    ap.add_argument("--outdir", default=str(Path(__file__).parent),
        help="Where to write the CSV outputs")
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
        for k in ("N", "reasoner_acc", "raw_final_acc", "adj_final_acc",
                  "delta_raw_pp", "delta_adj_pp", "w2c_total",
                  "kg_supported", "kg_supported_of_which_context_driven",
                  "mixed", "leakage", "unclassified", "c2w"):
            print(f"  {k:35s} = {s[k]}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_path = out_dir / "leakage_classification_summary.csv"
    pc_path = out_dir / "leakage_classification_per_case.csv"
    if summaries:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"\nWrote {sum_path}")
    if per_case_all:
        with open(pc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case_all[0].keys()))
            w.writeheader()
            w.writerows(per_case_all)
        print(f"Wrote {pc_path}")


if __name__ == "__main__":
    main()
