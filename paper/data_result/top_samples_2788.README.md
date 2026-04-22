# top_samples_2788.jsonl

Clean release version of the KG-grounded MedReason evaluation set used in the QKG paper (N = 2,788).

## Provenance

- **Source file**: `top_samples.jsonl` (2,789 records) in the same directory.
- **Derived by**: removing a single reserved case-study sample.

## Filter rule

One record is excluded:

| `sample_key` | Reason |
|---|---|
| `medqa_839` | Reserved as the canonical debug / case-study sample across the evaluation scripts. Heavily iterated on during development, so it is held out of aggregate metrics to avoid contaminating reported numbers. |

`medqa_839` itself is a well-formed sample --- no field errors, patient context extracts cleanly, has KG paths --- but it is hardcoded into `EXCLUDE_KEYS = {"medqa_839"}` in:

- `conditionKgTestValidation.py:54`
- `conditionKgTestAgentic.py:51`
- `tools/inspect_bad_case.py:120` (random-pick guard)

and is the example sample cited in the CLI help strings (`--run medqa_839`).

## How to reproduce

```bash
python3 - <<'PY'
import json
EXCLUDE = {"medqa_839"}
with open("top_samples.jsonl") as fin, open("top_samples_2788.jsonl", "w") as fout:
    for ln in fin:
        if json.loads(ln)["sample_key"] not in EXCLUDE:
            fout.write(ln)
PY
```

Verification:

```
$ wc -l top_samples_2788.jsonl
2788 top_samples_2788.jsonl
```

## Schema

Each line is a JSON object with four top-level fields:

| Field | Type | Description |
|---|---|---|
| `sample_key` | str | Unique sample id, e.g. `qa_4657`, `medqa_1234` |
| `original_sample` | dict | Source QA record and KG-grounding metadata (question, options, answer_key, question_entities, choice_entities, primekg_subgraph_evidence, patient_character, coverage, bfs_path, etc.) |
| `v1_eval` | dict | Pure-LLM (no-KG) baseline evaluation: `parsed`, `raw_response`, `error` |
| `v3_eval` | dict | BFS-path KG-augmented evaluation: `parsed`, `raw_response`, `error` |

## Alignment with the paper

This file matches the N = 2,788 curated evaluation set reported in the QKG paper (Experimental Setup, Section 4.1). All evaluation output files in the paper use the same `sample_key` space as this file.
