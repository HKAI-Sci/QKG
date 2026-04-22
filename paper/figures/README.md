# Figures and analysis artefacts

This directory holds the figures cited in the paper (as compiled PDFs), the
scripts that produce them, and the leakage-classification artefacts cited in
Section 5.3 and Appendix A.3.

## Figures used by the paper

- `figure1_qkg_overview_chrome.pdf`
  - Figure 1, rendered via Chrome from `figure1_qkg_overview_print.html`
  - Source: `generate_figure1_qkg_overview.py` writes the SVG, Chrome prints it to PDF
- `main_patient_results_plot.pdf`
  - Figure 2 (Haiku-validator ablation; fact/patient-context)
  - Built from `main_patient_results.csv` by `generate_main_patient_results_plot.py`
- `qwen_validator_results_plot.pdf`
  - Figure 3 (Qwen-validator ablation)
  - Built from `qwen_validator_results.csv` by `generate_qwen_validator_results_plot.py`
- `case_studies_combined.pdf`
  - Figure 4 (two case studies)
  - Built from `case_study_1.png` + `case_study_2.png` by `combine_case_studies.py`
  - The source PNGs can be regenerated with `generate_case_studies.py`

## Leakage classification (Table 2 / Table 3 / Appendix A.3)

Two passes produce the leakage-classification numbers cited in Tables 2 and 3
and in Section 5.3:

1. **Regex pass**: `classify_leakage.py` (W->C revisions) and
   `classify_leakage_c2w.py` (C->W regressions). Labels each decisive
   `CONTRADICTED` claim with one of `EV_CONTEXT` / `EV_KG_GROUNDED` /
   `EV_LEAKAGE` / `EV_UNCLASSIFIED`, then rolls up to a per-case label of
   `LIKELY_KG_SUPPORTED` / `MIXED` / `LIKELY_LEAKAGE` / `UNCLASSIFIED` using
   the rules in Appendix A.3.
2. **LLM re-label pass**: `../../classify_unclassified_with_llm.py` (W->C)
   and `../../classify_unclassified_c2w_with_llm.py` (C->W). Prompt the
   Haiku-4.5 LLM configured under `patient-context-llm` in
   `../../conf/config.yaml` to re-label only the residual `UNCLASSIFIED`
   cases (29 + 27 for W->C; 3 + 3 for C->W), using the same decisive
   evidence strings.

The two drivers live at the **repository root** because they pick up
`conf/config.yaml` and the `f1` package from the repo root.

### Files

- `classify_leakage.py`, `classify_leakage_c2w.py`
  - Regex classifier. Reads a validator-run JSONL log; writes
    `leakage_classification_*.csv` in this directory.
  - Input JSONL logs are user-supplied at run time; defaults in the scripts
    point at the development host path, override via `--no-pc` / `--with-pc`
    or env vars `QKG_NO_PC_LOG` / `QKG_WITH_PC_LOG`.
- `leakage_classification_summary.csv`, `leakage_classification_per_case.csv`
  - Regex-pass output for W->C. Kept for transparency; re-running
    `classify_leakage.py` regenerates them.
- `leakage_classification_summary_llm.csv`, `leakage_classification_per_case_llm.csv`
  - **Final W->C numbers used in Tables 2 and 3.** Merged regex + LLM labels.
    The per-case CSV includes both the regex label, the LLM label, and the
    source of the final label.
- `leakage_classification_c2w_summary.csv`, `leakage_classification_c2w_per_case.csv`
  - **Final C->W numbers used in Table 3.** Same schema as the W->C per-case
    CSV with LLM relabels.

## Regeneration

From the repository root:

```bash
python3 paper/figures/generate_figure1_qkg_overview.py
'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --headless --disable-gpu --allow-file-access-from-files \
  --print-to-pdf-no-header \
  --print-to-pdf='paper/figures/figure1_qkg_overview_chrome.pdf' \
  "file://$(pwd)/paper/figures/figure1_qkg_overview_print.html"

python3 paper/figures/generate_main_patient_results_plot.py
python3 paper/figures/generate_qwen_validator_results_plot.py
python3 paper/figures/generate_case_studies.py
python3 paper/figures/combine_case_studies.py
```

The leakage-classification CSVs can be regenerated once the user has their
own validator-run JSONL logs (Table 1 and Section 4.2 describe the pipeline):

```bash
# regex pass — point --no-pc / --with-pc at your own JSONL logs
python3 paper/figures/classify_leakage.py --no-pc /path/to/nopc.jsonl --with-pc /path/to/wpc.jsonl
python3 paper/figures/classify_leakage_c2w.py --no-pc /path/to/nopc.jsonl --with-pc /path/to/wpc.jsonl

# LLM re-label pass (requires f1 package and conf/config.yaml)
python3 classify_unclassified_with_llm.py
python3 classify_unclassified_c2w_with_llm.py
```

## Notes

- The PDF outputs are committed alongside their sources; PNG case-study
  images are kept because `combine_case_studies.py` depends on them.
- `generate_figure1_qkg_overview.py` writes `figure1_qkg_overview.svg` as an
  intermediate before Chrome exports the final PDF.
