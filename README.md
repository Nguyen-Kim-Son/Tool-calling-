# Closed-set API-Bank Benchmark for Compact LLMs

This repository contains a cleaned, script-based version of the experimental notebook used to evaluate compact and provider-accessible language models on a closed-set API-Bank tool-calling benchmark. The refactor preserves the core benchmark logic while removing Colab-specific cells, exploratory baseline discovery code, and notebook-only output formatting.

## What this repository includes

The main script, `benchmark_closed_set.py`, supports the full benchmark workflow needed for reproducibility and paper-table generation. It can load API-Bank samples, build a cleaned executable subset, retrieve tool schemas with BM25, call OpenAI-compatible model endpoints, execute predicted API calls against the local API-Bank implementation, and export detailed as well as summary CSV files. It also supports rebuilding the paper-ready tables used in the manuscript, including the main multiseed table, the ablation table, the extended table, and the deployability table.

## Repository structure

```text
.
├── benchmark_closed_set.py
├── configs
│   ├── model_roles.yaml
│   └── models.example.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

The cleaned script assumes that the `DeepAgent` repository is already available locally and contains the API-Bank data folder at:

```text
DeepAgent/data/API-Bank/
```

The required subfolders are:

- `data/API-Bank/apis`
- `data/API-Bank/init_database`
- `data/API-Bank/lv1-lv2-samples/level-1-given-desc-e2e`

## Installation

Create a virtual environment if desired, then install the required packages:

```bash
pip install -r requirements.txt
```

## Model configuration

The script expects a YAML file with one or more model specifications. A template is provided at `configs/models.example.yaml`. The file supports OpenAI-compatible APIs and can resolve environment variables such as `${SILICONFLOW_API_KEY}`.

Example:

```yaml
- enabled: true
  model_id: qwen25_7b_instruct_sf
  label: Qwen2.5-7B-Instruct
  provider: SiliconFlow
  base_url: https://api.siliconflow.com/v1
  api_key: ${SILICONFLOW_API_KEY}
  model_name: Qwen/Qwen2.5-7B-Instruct
  api_mode: chat
  timeout_seconds: 45
  max_tokens: 256
  input_cost_per_1m: 0.0
  output_cost_per_1m: 0.0
```

## Run the benchmark

A typical main run with a balanced subset of 50 samples looks like this:

```bash
python benchmark_closed_set.py run \
  --deepagent-root /path/to/DeepAgent \
  --runs-root ./runs \
  --models-yaml ./configs/models.example.yaml \
  --run-name main_closed_set_api_bank_singlecall_n50_seed42 \
  --prompt-variants json_only \
  --schema-modes bm25_topk \
  --seed 42 \
  --max-samples 50
```

An ablation run with XML-wrapped JSON and full schema exposure looks like this:

```bash
python benchmark_closed_set.py run \
  --deepagent-root /path/to/DeepAgent \
  --runs-root ./runs \
  --models-yaml ./configs/models.example.yaml \
  --run-name ablation_closed_set_api_bank_singlecall_n50_seed42_xml_json__full \
  --prompt-variants xml_json \
  --schema-modes full \
  --seed 42 \
  --max-samples 50
```

Each run creates a dedicated folder under `runs/` containing:

- `detailed_results.csv`
- `summary_results.csv`
- `summary_results.md`
- `run_config.json`
- `paper_subset_inventory.csv`

## Rebuild paper tables

### Main multiseed table

```bash
python benchmark_closed_set.py aggregate-main \
  --runs-root ./runs \
  --run-names main_closed_set_api_bank_singlecall_n50_seed42,main_closed_set_api_bank_singlecall_n50_seed7,main_closed_set_api_bank_singlecall_n50_seed123 \
  --model-roles-yaml ./configs/model_roles.yaml \
  --output-csv ./runs/paper_tables_final/paper_table_main_multiseed.csv
```

### Ablation table

```bash
python benchmark_closed_set.py aggregate-ablation \
  --runs-root ./runs \
  --run-names ablation_closed_set_api_bank_singlecall_n50_seed42_json_only__bm25_topk,ablation_closed_set_api_bank_singlecall_n50_seed42_xml_json__bm25_topk,ablation_closed_set_api_bank_singlecall_n50_seed42_json_only__full,ablation_closed_set_api_bank_singlecall_n50_seed42_xml_json__full \
  --model-roles-yaml ./configs/model_roles.yaml \
  --output-csv ./runs/paper_tables_final/paper_table_ablation.csv
```

### Extended multiseed table

```bash
python benchmark_closed_set.py aggregate-extended \
  --runs-root ./runs \
  --run-names main_closed_set_api_bank_singlecall_n50_seed42,main_closed_set_api_bank_singlecall_n50_seed7,main_closed_set_api_bank_singlecall_n50_seed123,extended_main_closed_set_api_bank_singlecall_n50_seed42,extended_main_closed_set_api_bank_singlecall_n50_seed7,extended_main_closed_set_api_bank_singlecall_n50_seed123 \
  --model-roles-yaml ./configs/model_roles.yaml \
  --output-csv ./runs/paper_tables_final/paper_table_main_multiseed_extended.csv
```

### Deployability / operational reliability table

```bash
python benchmark_closed_set.py export-deployability \
  --paper-main-extended-csv ./runs/paper_tables_final/paper_table_main_multiseed_extended.csv \
  --output-csv ./runs/paper_tables_final/paper_table_deployability_extended.csv
```

## Notes on reproducibility

The original notebook contained additional exploratory cells for baseline discovery, smoke testing, and compatibility wrappers. Those cells were intentionally not copied into the cleaned release because they are not necessary for the core benchmark pipeline or for reproducing the paper tables. The cleaned script focuses on the benchmark logic that matters for the final reported results.

If you want to publish this repository together with experimental outputs, it is recommended to upload the generated CSV files under a separate folder such as `paper_tables_final/` or `released_results/`, while keeping API keys and any personal runtime paths out of version control.

## Suggested citation note for the repository

If you release this code publicly, you may add a short note in the README or repository description stating that the code is a cleaned research release derived from the paper’s experimental notebook and intended for reproducibility of the closed-set API-Bank benchmark.
