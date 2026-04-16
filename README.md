# Quantum Knowledge Graph (QKG): Modeling Context-Dependent Triplet Validity

QKG is the public codebase for reproducing the paper’s question-guided knowledge graph inference workflow on MedReason-derived medical QA data.

The supported reproduction path is artifact-first:

- load the published QKG Mongo-ready artifacts
- load upstream PrimeKG relations
- configure the `reasoner` and `validator` LLMs
- run `conditionKgTestAgentic.py`

This repo does not require rebuilding the historical internal preprocessing pipeline.

## What You Need

There are two groups of required data.

### 1. Official Datasets From Other Sources

1. Official `PrimeKg.csv`
   - upstream dependency used to populate `primeKG.relations`
   - example: the `PrimeKg.csv` file from the official PrimeKG release
   - actual Mongo example used by this project: `{relation: "protein_protein", display_relation: "ppi", x_index: 0, x_id: "9796", x_type: "gene/protein", x_name: "PHYHIP", x_source: "NCBI", y_index: 8889, y_id: "56992", y_type: "gene/protein", y_name: "KIF15", y_source: "NCBI"}`
2. Official UMLS data loaded into MongoDB
   - required for exact reproduction of the `search_entity` UMLS synonym fallback
   - example: official UMLS string data from `MRCONSO.RRF`, imported into `umls_test.umls_strings_raw_test`
   - actual Mongo example used by this project: `{aui: "A26634265", cui: "C0000005", language: "ENG", source: "MSH", source_code: "D012711", source_name: "(131)I-Macroaggregated Albumin", source_term_type: "PEP", string_type: "PF", term_status: "P"}`

### 2. Datasets Published By QKG

3. `qkg-primekg-entities-with-cui`
   - published by us at https://huggingface.co/datasets/wangyaobupt/qkg-primekg-entities-with-cui
   - available as JSONL
4. `qkg-relation-with-facts`
   - published by us at https://huggingface.co/datasets/wangyaobupt/qkg-relation-with-facts
   - available as JSONL
5. `qkg-medreason-eval.jsonl`
   - TODO: published by us
   - evaluation dataset consumed by `conditionKgTestAgentic.py`

## Quickstart

### 1. Create an environment

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create local config

```bash
mkdir -p conf
cp conf/config.example.yaml conf/config.yaml
```

Edit `conf/config.yaml` with:

- your MongoDB URIs
- your artifact file paths
- your `reasoner` LLM backend
- your `validator` LLM backend

The current config supports two LLM roles:

- `reasoner`
- `validator`

Each can use either:

- OpenRouter / OpenAI-compatible chat API
- AWS Bedrock

## Required Config

At minimum, set these fields:

```yaml
mongo:
  primekg_uri: mongodb://localhost:27017/primeKG
  umls_uri: mongodb://localhost:27017/umls_test

paths:
  primekg_csv: /absolute/path/to/PrimeKg.csv
  umls_mrconso_rrf: /absolute/path/to/MRCONSO.RRF
  primekg_entities_jsonl: /absolute/path/to/qkg-primekg-entities-with-cui.jsonl
  relation_with_facts_jsonl: /absolute/path/to/qkg-relation-with-facts.jsonl
  qa_eval_jsonl: /absolute/path/to/qkg-medreason-eval.jsonl
```


## Load Data Into MongoDB

The runtime expects four Mongo collections:

- `primeKG.relations`
- `primeKG.entities`
- `primeKG.relation_with_facts`
- `umls_test.umls_strings_raw_test`

Collection purpose and field summaries are in [docs/resource-mongo.md](docs/resource-mongo.md).

### Load QKG JSONL Artifacts

For the `mongoimport` command, export your PrimeKG Mongo URI first:

```bash
export QKG_PRIMEKG_MONGO_URI="mongodb://localhost:27017/primeKG"
```

```bash
mongoimport --uri "$QKG_PRIMEKG_MONGO_URI" --db primeKG --collection entities --file /path/to/qkg-primekg-entities-with-cui.jsonl
python tools/import_relation_facts.py
```

The first command loads the published `qkg-primekg-entities-with-cui.jsonl` artifact into `primeKG.entities`.

The second command loads `paths.relation_with_facts_jsonl` into `primeKG.relation_with_facts`.

Then load the two official upstream dependencies:

```bash
python tools/import_primekg_relations.py
python tools/import_umls_strings.py
```

These scripts read from:

- `paths.primekg_csv`
- `paths.umls_mrconso_rrf`
- `mongo.primekg_uri`
- `mongo.umls_uri`

The UMLS import is required for exact reproduction because the public runtime uses `umls_test.umls_strings_raw_test` for UMLS-backed synonym fallback in `search_entity`.

## Run Evaluation

The main public entrypoint is:

```bash
python conditionKgTestAgentic.py
```

Useful variants:

```bash
python conditionKgTestAgentic.py --patient-context
python conditionKgTestAgentic.py --run medqa_839
python conditionKgTestAgentic.py --run medqa_839 --patient-context
```

The input evaluation file is read from:

- `paths.qa_eval_jsonl`

The output file path is controlled by command-line flags:

- `--output /path/to/results.jsonl`

If `--output` is omitted, the script writes to an auto-named file under `output/`.

## Output Format

`conditionKgTestAgentic.py` writes one JSON object per line.

Typical top-level fields include:

- `sample_key`: stable sample identifier from the evaluation dataset
- `gold_answer`: gold multiple-choice answer
- `agentic_answer`: final answer chosen by the agentic pipeline
- `agentic_correct`: whether `agentic_answer` matches `gold_answer`
- `agentic_reasoning`: final natural-language reasoning returned by the agent
- `num_turns`: total ReAct turns used in the run
- `num_tool_calls`: number of KG tool invocations made by the agent
- `tool_calls`: structured log of tool actions, including turn number and observation length
- `conversation`: full assistant / observation trace for the run
- `llm_stats`: aggregated LLM usage stats such as call count and total model time
- `elapsed_s`: end-to-end wall-clock runtime in seconds

When `--patient-context` is enabled, the output also includes:

- `patient_context`: extracted patient summary used for patient-aware validation
- `hook_log`: log of post-processing applied to KG results in patient-context mode
- `compression_log`: summaries of any observation-compression steps used to keep context size manageable

Example structure:

```json
{
  "sample_key": "qa_4055",
  "gold_answer": "D",
  "agentic_answer": "D",
  "agentic_correct": true,
  "agentic_reasoning": "...",
  "num_turns": 11,
  "num_tool_calls": 10,
  "tool_calls": [...],
  "conversation": [...],
  "llm_stats": {
    "calls": 22,
    "wait_s": 0.0,
    "llm_s": 276.4
  },
  "patient_context": "...",
  "hook_log": [...],
  "compression_log": [...],
  "elapsed_s": 320.8
}
```
