# Per-sample correctness and paired significance artefacts

Inputs and outputs for the McNemar paired significance tests cited in
Results (Figures 2 and 3, Table 2) and described in Method (Section 3.3,
`Statistical Testing`).

## Raw per-sample files

One row per sample. Columns:

```
sample_key, gold_answer, reasoner_answer, final_answer,
reasoner_correct, final_correct
```

| File | Reasoner | Validator | Patient context |
|---|---|---|---|
| `haiku_wpc.csv`  | Haiku-4.5 | Haiku-4.5     | with    |
| `haiku_nopc.csv` | Haiku-4.5 | Haiku-4.5     | without |
| `qwen_wpc.csv`   | Haiku-4.5 | Qwen-3.6-Plus | with    |
| `qwen_nopc.csv`  | Haiku-4.5 | Qwen-3.6-Plus | without |

These are extracted from the validator-run JSONL logs produced by
`conditionKgTestAgentic.py` (one JSONL per setting). See the snippet at
the bottom of this file for the extraction script.

## Derived paired files

Produced by `../significance_tests.py` from the four raw files:

- `paired_haiku.csv` — joined on `sample_key`, Haiku-validator runs.
- `paired_qwen.csv` — joined on `sample_key`, Qwen-validator runs.
- `paired_qwen_leakage_adjusted.csv` — Qwen-validator pair with samples
  flagged in *either* run removed, where "flagged" means:
  - a W->C revision labelled `LIKELY_LEAKAGE`
    (`../../figures/leakage_classification_per_case_llm.csv`), or
  - a C->W regression labelled `LIKELY_KG_SUPPORTED` whose decisive evidence
    cites a QKG applicability token (the ctx-driven subset; see
    `../../figures/leakage_classification_c2w_per_case.csv`).

This is the subset used for the leakage-adjusted paired McNemar test
reported alongside Table 2 (Results section) and described in
`Statistical Testing` (Method).

## Regenerating

```bash
# 1) dump per-sample correctness from your validator-run JSONL logs:
python3 - <<'PY'
import json, csv
runs = {
    'haiku_wpc':  '/path/to/haiku_wpc.jsonl',   # Haiku R + Haiku V, with patient context
    'haiku_nopc': '/path/to/haiku_nopc.jsonl',  # Haiku R + Haiku V, no patient context
    'qwen_wpc':   '/path/to/qwen_wpc.jsonl',    # Haiku R + Qwen V,  with patient context
    'qwen_nopc':  '/path/to/qwen_nopc.jsonl',   # Haiku R + Qwen V,  no patient context
}
for name, path in runs.items():
    with open(path) as f, open(f'{name}.csv', 'w', newline='') as g:
        w = csv.writer(g)
        w.writerow(['sample_key','gold_answer','reasoner_answer','final_answer',
                    'reasoner_correct','final_correct'])
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            w.writerow([d['sample_key'], d['gold_answer'], d['reasoner_answer'], d['final_answer'],
                        int(bool(d['reasoner_correct'])), int(bool(d['final_correct']))])
PY

# 2) produce the paired tables and the significance summary:
python3 paper/data_result/significance_tests.py
```
