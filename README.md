#  A Structural-Validity Audit Framework for Large Language Model-Based Depression-Language Inference

This repository contains the artifact package for a reproducible structural-validity audit of large language models (LLMs) for depression-language inference. The code reconstructs annotation-derived readouts from two public datasets, evaluates prompted and adapted LLM routes, and analyzes whether model outputs preserve endpoint labels, ordinal severity, symptom-language profiles, coupled symptom states, annotation uncertainty, and lexical visibility.

The artifact is intended for scientific auditing and reproducibility. It is not a diagnostic system and must not be used for autonomous clinical diagnosis, triage, intervention, or individual-level decision making.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `Deptweet/` | DEPTWEET split files for binary screening and four-level severity evaluation. |
| `DepressionEmo-main/` | Original DepressionEmo resources and baseline scripts for multilabel symptom-language analysis. |
| `DepressionEmo-main/Dataset/` | DepressionEmo train/validation/test files and label definitions. |
| `DEPTWEET_Split.ipynb` | Reconstructs the DEPTWEET train/validation/test splits used by the audit. |
| `SFT_data_create.ipynb` | Builds instruction-tuning and preference-training files for binary, ordinal, symptom, and joint readouts. |
| `Evaluate_LLMs/` | Prompted evaluation notebooks for open and closed LLM routes across binary, ordinal, and symptom readouts. |
| `interpretability9B.py` | Generates route-level structural and interpretability analyses from saved prediction files. |
| `LlamaFactory/` | Fine-tuning framework with registered audit datasets and training configurations. |
| `LlamaFactory/data/` | Converted instruction datasets for SFT, joint SFT, and SFT-to-KTO routes. |

## Audit Readouts

The artifact supports four nested annotation-derived readouts:

| Readout | Dataset Source | Output Space |
| --- | --- | --- |
| Binary screening | DEPTWEET | non-depressed vs. depressed |
| Ordinal severity | DEPTWEET | non-depressed, mild, moderate, severe |
| Symptom-language profiling | DepressionEmo | 8-dimensional multilabel symptom vector |
| Coupled states | DEPTWEET + DepressionEmo | joint severity and symptom-state readouts |

The eight symptom labels follow `DepressionEmo-main/Dataset/label_names.json` and are used to construct symptom-vector and coupled-state metrics.

## Environment

The main analysis notebooks use standard Python scientific packages. The fine-tuning route uses the included LlamaFactory codebase.

Recommended environment:

```bash
conda create -n kdd2027-audit python=3.11 -y
conda activate kdd2027-audit

pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm openai jupyter

cd LlamaFactory
pip install -e .
```

For GPU fine-tuning, install a PyTorch build matching the local CUDA driver before installing LlamaFactory dependencies. The original experiments used Qwen-family base models with LoRA-style adaptation through LlamaFactory.

## Data Files for LlamaFactory

The converted training/evaluation files are registered in `LlamaFactory/data/dataset_info.json`.

| Dataset Key | File | Use |
| --- | --- | --- |
| `Deptweet_train_binary` | `Deptweet_train_binary.json` | binary SFT training |
| `Deptweet_test_binary` | `Deptweet_test_binary.json` | binary evaluation |
| `Deptweet_train_4` | `Deptweet_train_4.json` | ordinal severity SFT training |
| `Deptweet_test_4` | `Deptweet_test_4.json` | ordinal severity evaluation |
| `DepressionEmo_train_8` | `DepressionEmo_train_8.json` | symptom multilabel SFT training |
| `DepressionEmo_test_8` | `DepressionEmo_test_8.json` | symptom multilabel evaluation |
| `Multi_train_2_8` | `Multi_train_2_8.json` | binary-to-symptom joint training |
| `Multi_train_4_8` | `Multi_train_4_8.json` | severity-to-symptom joint training |

Each converted item follows the instruction format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

## Reproduction Workflow

### 1. Reconstruct Dataset Splits

Run:

```bash
jupyter notebook DEPTWEET_Split.ipynb
```

This notebook reconstructs DEPTWEET binary and four-class splits. If running in a new environment, replace any old absolute paths in the notebook with the repository root.

### 2. Build SFT and KTO Data

Run:

```bash
jupyter notebook SFT_data_create.ipynb
```

This notebook converts the public datasets into instruction-tuning files for binary screening, ordinal severity, symptom-language profiling, and joint readouts. The generated files should be placed under `LlamaFactory/data/` and registered in `dataset_info.json`.

### 3. Prompted LLM Evaluation

Prompted native-model evaluation is organized by task:

```bash
jupyter notebook Evaluate_LLMs/Open_LLMs.ipynb
jupyter notebook Evaluate_LLMs/Open_LLMs_4.ipynb
jupyter notebook Evaluate_LLMs/Open_LLMs_8.ipynb
jupyter notebook Evaluate_LLMs/Close_LLMs.ipynb
jupyter notebook Evaluate_LLMs/Close_LLMs_4.ipynb
jupyter notebook Evaluate_LLMs/Close_LLMs_8.ipynb
```

The notebooks leave API keys blank by default. Set keys through environment variables or a local private configuration file. Do not commit credentials.

### 4. Supervised Fine-Tuning

From the `LlamaFactory/` directory, run LlamaFactory training with the dataset keys above. A typical LoRA SFT command is:

```bash
cd LlamaFactory
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

Update the YAML fields for the target route:

```yaml
model_name_or_path: <base_model_path_or_hub_id>
stage: sft
dataset: Deptweet_train_4
template: qwen3_nothink
output_dir: saves/<model_name>/lora/<route_name>
```

For joint SFT, set `dataset` to `Multi_train_2_8` or `Multi_train_4_8`. For symptom profiling, set `dataset` to `DepressionEmo_train_8`.

### 5. Preference Refinement

The SFT-to-KTO route uses supervised alignment first, followed by preference refinement. Use the KTO example as the template:

```bash
cd LlamaFactory
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml
```

Set the adapter path to the corresponding SFT checkpoint and set the dataset to the KTO-formatted data generated from `SFT_data_create.ipynb`.

### 6. Prediction and Structural Metrics

After inference, keep each route's predictions as JSONL files with prediction and label fields. The analysis scripts parse common keys such as:

```text
predict, prediction, pred, output, response
label, labels, target, answer, gold
```

Run:

```bash
jupyter notebook analysis.ipynb
python interpretability9B.py
```

Before running `interpretability9B.py`, update `PROJECT_ROOT` or refactor it to the current repository path. The script expects route-level `generated_predictions.jsonl` files under the LlamaFactory `saves/` directory and writes supporting CSV and figure outputs to `result_IMG/`.

## Reported Metric Families

The analysis computes:

- Endpoint metrics: accuracy, precision, recall, F1, exact match, and Hamming loss.
- Ordinal structure: severity-boundary behavior and ordinal repair.
- Symptom structure: symptom-level recovery and symptom-vector consistency.
- Coupled-state topology: recovery over joint severity-symptom states.
- Uncertainty analysis: annotation agreement and ambiguous-state behavior.
- Lexical visibility analysis: performance under explicit and weak lexical cues.
- High-risk failures: severe-state misses and suicide-intent-related errors.


## Ethics and Intended Use

This artifact audits model behavior on public depression-language datasets. Labels are annotation-derived and dataset-specific; they are not clinical diagnoses. The code and outputs should be used only for research on structural validity, failure analysis, and reproducible evaluation of mental-health language models. Any real-world mental-health application would require calibrated abstention, human review, escalation protocols, external validation, and population-specific oversight.

