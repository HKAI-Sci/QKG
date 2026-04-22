"""
Paired McNemar significance tests for:
  - Figure 2 (fig:main_patient_results, Haiku Reasoner + Haiku Validator)
  - Figure 3 (fig:qwen-validator-results, Haiku Reasoner + Qwen Validator)
  - Table 2 (tab:leakage-classification, leakage-adjusted Qwen comparison)

Inputs (per-sample, in paper/data_result/per_sample/):
  haiku_wpc.csv, haiku_nopc.csv, qwen_wpc.csv, qwen_nopc.csv
  (extracted from the four validator-run JSONL logs; see
   per_sample/README.md for the extraction snippet)

Outputs:
  paper/data_result/per_sample/paired_haiku.csv
  paper/data_result/per_sample/paired_qwen.csv
  paper/data_result/per_sample/paired_qwen_leakage_adjusted.csv
  paper/data_result/significance_results.csv  (machine-readable summary)
"""
import csv
import os
from math import comb
from statistics import NormalDist

ROOT = os.path.dirname(os.path.abspath(__file__))
PER = os.path.join(ROOT, "per_sample")
FIG = os.path.join(os.path.dirname(ROOT), "figures")


def load_run(name):
    with open(os.path.join(PER, f"{name}.csv")) as f:
        return {r["sample_key"]: r for r in csv.DictReader(f)}


def mcnemar_exact_two_sided(b, c):
    """Exact binomial two-sided p-value for McNemar.

    Under H0 that each discordant flip is equally likely to go either way,
    the count of flips in one direction is Binomial(b+c, 0.5).
    Two-sided p = 2 * P(X >= max(b,c)) (clipped at 1).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    # P(X >= k) under Binomial(n, 0.5) = sum_{i=k..n} C(n,i) / 2**n
    tail = sum(comb(n, i) for i in range(k, n + 1))
    p = 2 * tail / (2 ** n)
    return min(p, 1.0)


def mcnemar_cc_chi2(b, c):
    """Continuity-corrected chi-square (1 df) p-value."""
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # survival of chi-square(1) at x equals 2 * (1 - Phi(sqrt(x))) for x >= 0
    from math import sqrt
    return 2 * (1 - NormalDist().cdf(sqrt(chi2)))


def paired_discordant(records_a, records_b, key_a="final_correct", key_b="final_correct"):
    """For the intersection of sample_keys, return (b, c) where
    b = a correct & b wrong, c = a wrong & b correct.  (Standard McNemar off-diag.)
    """
    keys = set(records_a) & set(records_b)
    b = c = both_correct = both_wrong = 0
    for k in keys:
        a_ok = int(records_a[k][key_a])
        b_ok = int(records_b[k][key_b])
        if a_ok and not b_ok:
            b += 1
        elif (not a_ok) and b_ok:
            c += 1
        elif a_ok and b_ok:
            both_correct += 1
        else:
            both_wrong += 1
    return b, c, both_correct, both_wrong, len(keys)


def report(label, b, c, n, acc_a, acc_b):
    p_exact = mcnemar_exact_two_sided(b, c)
    p_cc = mcnemar_cc_chi2(b, c)
    delta = (acc_b - acc_a) * 100
    print(f"{label}")
    print(f"  N (paired)      = {n}")
    print(f"  acc_A / acc_B   = {acc_a*100:.2f}% / {acc_b*100:.2f}%  (delta = {delta:+.2f} pp)")
    print(f"  discordant b/c  = {b} / {c}   (b = A-correct & B-wrong; c = A-wrong & B-correct)")
    print(f"  McNemar exact 2-sided p = {p_exact:.4g}")
    print(f"  McNemar CC chi2    p    = {p_cc:.4g}")
    print()
    return {
        "comparison": label,
        "N_paired": n,
        "acc_A_pct": round(acc_a * 100, 2),
        "acc_B_pct": round(acc_b * 100, 2),
        "delta_pp": round(delta, 2),
        "b_A_correct_B_wrong": b,
        "c_A_wrong_B_correct": c,
        "p_exact_two_sided": p_exact,
        "p_cc_chi2": p_cc,
    }


def write_paired_csv(path, run_a, run_b, name_a, name_b):
    keys = sorted(set(run_a) & set(run_b))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_key", "gold_answer",
            f"reasoner_correct_{name_a}", f"final_correct_{name_a}", f"final_answer_{name_a}",
            f"reasoner_correct_{name_b}", f"final_correct_{name_b}", f"final_answer_{name_b}",
        ])
        for k in keys:
            a = run_a[k]
            b = run_b[k]
            w.writerow([
                k, a["gold_answer"],
                a["reasoner_correct"], a["final_correct"], a["final_answer"],
                b["reasoner_correct"], b["final_correct"], b["final_answer"],
            ])
    print(f"wrote {path}  ({len(keys)} paired rows)")


def main():
    h_wpc = load_run("haiku_wpc")
    h_nopc = load_run("haiku_nopc")
    q_wpc = load_run("qwen_wpc")
    q_nopc = load_run("qwen_nopc")

    write_paired_csv(os.path.join(PER, "paired_haiku.csv"), h_nopc, h_wpc, "haiku_nopc", "haiku_wpc")
    write_paired_csv(os.path.join(PER, "paired_qwen.csv"),  q_nopc, q_wpc, "qwen_nopc",  "qwen_wpc")
    print()

    results = []
    print("=" * 72)
    print("Figure 2 — Haiku Reasoner + Haiku Validator  (fig:main_patient_results)")
    print("=" * 72)

    # A. No-validator vs KG-no-pc: each sample's reasoner_correct vs final_correct in haiku_nopc
    acc_r = sum(int(r["reasoner_correct"]) for r in h_nopc.values()) / len(h_nopc)
    acc_f = sum(int(r["final_correct"]) for r in h_nopc.values()) / len(h_nopc)
    b, c, _, _, n = paired_discordant(h_nopc, h_nopc, "reasoner_correct", "final_correct")
    results.append(report("A. No validator vs. KG w/o context (Haiku V)", b, c, n, acc_r, acc_f))

    # B. No-validator vs QKG-w/-pc
    acc_r = sum(int(r["reasoner_correct"]) for r in h_wpc.values()) / len(h_wpc)
    acc_f = sum(int(r["final_correct"]) for r in h_wpc.values()) / len(h_wpc)
    b, c, _, _, n = paired_discordant(h_wpc, h_wpc, "reasoner_correct", "final_correct")
    results.append(report("B. No validator vs. QKG w/ context (Haiku V)", b, c, n, acc_r, acc_f))

    # C. KG-no-pc vs QKG-w/-pc (validator-to-validator, paired on sample_key)
    keys = set(h_nopc) & set(h_wpc)
    acc_a = sum(int(h_nopc[k]["final_correct"]) for k in keys) / len(keys)
    acc_b = sum(int(h_wpc[k]["final_correct"]) for k in keys) / len(keys)
    b, c, _, _, n = paired_discordant(h_nopc, h_wpc)
    results.append(report("C. KG w/o context vs. QKG w/ context (Haiku V)", b, c, n, acc_a, acc_b))

    print("=" * 72)
    print("Figure 3 — Haiku Reasoner + Qwen Validator  (fig:qwen-validator-results)")
    print("=" * 72)

    acc_r = sum(int(r["reasoner_correct"]) for r in q_nopc.values()) / len(q_nopc)
    acc_f = sum(int(r["final_correct"]) for r in q_nopc.values()) / len(q_nopc)
    b, c, _, _, n = paired_discordant(q_nopc, q_nopc, "reasoner_correct", "final_correct")
    results.append(report("D. No validator vs. KG w/o context (Qwen V)", b, c, n, acc_r, acc_f))

    acc_r = sum(int(r["reasoner_correct"]) for r in q_wpc.values()) / len(q_wpc)
    acc_f = sum(int(r["final_correct"]) for r in q_wpc.values()) / len(q_wpc)
    b, c, _, _, n = paired_discordant(q_wpc, q_wpc, "reasoner_correct", "final_correct")
    results.append(report("E. No validator vs. QKG w/ context (Qwen V)", b, c, n, acc_r, acc_f))

    keys = set(q_nopc) & set(q_wpc)
    acc_a = sum(int(q_nopc[k]["final_correct"]) for k in keys) / len(keys)
    acc_b = sum(int(q_wpc[k]["final_correct"]) for k in keys) / len(keys)
    b, c, _, _, n = paired_discordant(q_nopc, q_wpc)
    results.append(report("F. KG w/o context vs. QKG w/ context (Qwen V)", b, c, n, acc_a, acc_b))

    # Table 2 leakage adjustment (Eq. 1 in Appendix A.3): drop
    #   (i)  W->C revisions labelled LIKELY_LEAKAGE, and
    #   (ii) C->W regressions whose decisive evidence cites a QKG
    #        applicability token (the ctx-driven subset of LIKELY_KG_SUPPORTED).
    # Then run the paired comparison on sample_keys kept by BOTH runs.
    print("=" * 72)
    print("Table 2 — Leakage-adjusted comparison  (tab:leakage-classification)")
    print("=" * 72)

    # (i) W->C LIKELY_LEAKAGE (from the LLM-augmented per-case CSV)
    leak = {"qwen_nopc": set(), "qwen_wpc": set()}
    lbl_path = os.path.join(FIG, "leakage_classification_per_case_llm.csv")
    with open(lbl_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row["final_label"] != "LIKELY_LEAKAGE":
                continue
            rl = row["run_label"]
            if "no patient context" in rl:
                leak["qwen_nopc"].add(row["sample_key"])
            elif "with patient context" in rl:
                leak["qwen_wpc"].add(row["sample_key"])

    # (ii) C->W ctx-driven LIKELY_KG_SUPPORTED (the subset whose decisive evidence
    # cites a QKG applicability token; §5.2 argues these reflect weak MCQ gold,
    # not QKG failure).
    c2w_ctx = {"qwen_nopc": set(), "qwen_wpc": set()}
    c2w_path = os.path.join(FIG, "leakage_classification_c2w_per_case.csv")
    with open(c2w_path) as f:
        rdr = csv.DictReader(f)
        # Column name is "final_label" after the LLM re-label pass, falling
        # back to "case_label" for the regex-only legacy CSV.
        for row in rdr:
            label = row.get("final_label") or row.get("case_label")
            if label != "LIKELY_KG_SUPPORTED":
                continue
            if row["of_which_context_driven"].strip().lower() != "true":
                continue
            rl = row["run_label"]
            if "no patient context" in rl:
                c2w_ctx["qwen_nopc"].add(row["sample_key"])
            elif "with patient context" in rl:
                c2w_ctx["qwen_wpc"].add(row["sample_key"])

    print(f"likely-leakage W->C keys:       qwen_nopc={len(leak['qwen_nopc'])}  qwen_wpc={len(leak['qwen_wpc'])}")
    print(f"ctx-driven KG-supp C->W keys:   qwen_nopc={len(c2w_ctx['qwen_nopc'])}  qwen_wpc={len(c2w_ctx['qwen_wpc'])}")

    # Paper's adjustment drops these from numerator AND denominator.  For a paired
    # test, drop keys flagged in EITHER run (otherwise we would compare different
    # per-run sample sets).
    drop = (leak["qwen_nopc"] | leak["qwen_wpc"]
            | c2w_ctx["qwen_nopc"] | c2w_ctx["qwen_wpc"])
    keep_keys = (set(q_nopc) & set(q_wpc)) - drop
    print(f"paired-intersection before drop = {len(set(q_nopc) & set(q_wpc))}")
    print(f"paired-intersection after drop  = {len(keep_keys)}  (dropped {len(drop & (set(q_nopc) & set(q_wpc)))})")

    # save the adjusted paired CSV
    out = os.path.join(PER, "paired_qwen_leakage_adjusted.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_key", "gold_answer",
            "final_correct_qwen_nopc", "final_correct_qwen_wpc",
        ])
        for k in sorted(keep_keys):
            w.writerow([
                k, q_nopc[k]["gold_answer"],
                q_nopc[k]["final_correct"], q_wpc[k]["final_correct"],
            ])
    print(f"wrote {out}")

    acc_a = sum(int(q_nopc[k]["final_correct"]) for k in keep_keys) / len(keep_keys)
    acc_b = sum(int(q_wpc[k]["final_correct"]) for k in keep_keys) / len(keep_keys)
    b = sum(1 for k in keep_keys if int(q_nopc[k]["final_correct"]) == 1 and int(q_wpc[k]["final_correct"]) == 0)
    c = sum(1 for k in keep_keys if int(q_nopc[k]["final_correct"]) == 0 and int(q_wpc[k]["final_correct"]) == 1)
    results.append(report("G. KG w/o vs. QKG w/ (Qwen V, leakage-adjusted)", b, c, len(keep_keys), acc_a, acc_b))

    # write summary CSV
    out = os.path.join(os.path.dirname(ROOT), "data_result", "significance_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            r["p_exact_two_sided"] = f"{r['p_exact_two_sided']:.4g}"
            r["p_cc_chi2"] = f"{r['p_cc_chi2']:.4g}"
            w.writerow(r)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
