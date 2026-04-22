# Paper data results

Per-sample correctness tables, paired McNemar significance tests, and the
released summary numbers cited in Results (Figures 2 and 3, Table 2) and
Method (Section 3.3, `Statistical Testing`).

## Files

- `significance_tests.py` — driver for the paired McNemar tests, including
  the leakage adjustment that drops W->C likely-leakage revisions and
  C->W ctx-driven KG-supported regressions.
- `significance_results.csv` — machine-readable summary of all paired
  McNemar tests cited in the paper.
- `top_samples_2788.README.md` — documents how the N=2,788 curated
  evaluation set is derived from the larger 2,789-sample source file
  (one reserved case-study sample `medqa_839` is held out).
- `per_sample/` — per-sample correctness CSVs and their paired joins.
  See `per_sample/README.md`.

## How significance results are computed

```bash
python3 paper/data_result/significance_tests.py
```

This reads `per_sample/haiku_{wpc,nopc}.csv` and `per_sample/qwen_{wpc,nopc}.csv`,
writes `per_sample/paired_haiku.csv`, `per_sample/paired_qwen.csv`, and the
leakage-adjusted `per_sample/paired_qwen_leakage_adjusted.csv`, and refreshes
`significance_results.csv` with seven paired McNemar tests (A–G) matching the
numbers in the paper.

The leakage-adjustment subset draws from:

- `../figures/leakage_classification_per_case_llm.csv` (W->C
  `LIKELY_LEAKAGE` samples), and
- `../figures/leakage_classification_c2w_per_case.csv` (C->W
  `LIKELY_KG_SUPPORTED` samples whose decisive evidence cites a QKG
  applicability token, i.e., `of_which_context_driven=True`).

See Appendix A.3 of the paper for the rules and Eq. 1 for the
leakage-adjusted accuracy formula.
