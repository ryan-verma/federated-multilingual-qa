# Fair Federated Learning for Multilingual Question Answering

A federated multilingual Question Answering system built using **XLM-RoBERTa** that learns from geographically distributed language clients while preserving linguistic diversity and data privacy.

The project investigates trade-offs between centralized and federated training across five languages using **MLQA** as the primary benchmark, comparing standard FedAvg and fairness-aware aggregation (q-FFL). A key finding is that federated training with language-partitioned clients consistently **outperforms** the centralized baseline across all languages on both MLQA and XQuAD — attributed to the elimination of cross-language gradient interference during local training.

---

## Objectives

- Fine-tune a multilingual QA model using XLM-RoBERTa-base
- Simulate federated learning across multiple language-specific clients
- Study the effects of federated averaging on cross-lingual transfer
- Implement diversity-preserving aggregation to address language imbalance
- Evaluate cross-dataset generalization via zero-shot transfer to XQuAD
- Compare all approaches on per-language EM/F1 metrics

---

## Technologies Used

- Python 3.10
- PyTorch 2.11.0
- Hugging Face Transformers 4.52.4
- Hugging Face Datasets 3.6.0
- Flower (flwr 1.31.0)
- Pandas 3.0.3
- Matplotlib 3.11.0
- Scikit-learn 1.9.0
- Git & GitHub
- VS Code

---

## Model

**XLM-RoBERTa-base** — a 12-layer multilingual transformer pretrained on 100 languages (278M parameters), fine-tuned as an extractive QA model. A linear span-prediction head is added on top of the encoder, producing start and end logits per token.

---

## Datasets

**MLQA** (facebook/mlqa) — a multilingual extractive QA benchmark with aligned answer spans across 7 languages. Used for training and primary evaluation.

| Language | Train Source | Eval Config |
|---|---|---|
| English | SQuAD (real English QA) | mlqa.en.en |
| Hindi | mlqa-translate-train.hi | mlqa.hi.hi |
| Spanish | mlqa-translate-train.es | mlqa.es.es |
| German | mlqa-translate-train.de | mlqa.de.de |
| Chinese | mlqa-translate-train.zh | mlqa.zh.zh |

**XQuAD** (google/xquad) — 1,190 SQuAD v1.1 questions professionally translated into 10 languages. Used for zero-shot cross-dataset transfer evaluation — no XQuAD data was seen during training.

**XTREME** (xtreme) — Multilingual multitask benchmark. Planned for broader cross-lingual generalization evaluation.

---

## Federated Setup

| Setting | Value |
|---|---|
| Framework | Flower (flwr 1.31.0) |
| Clients | 5 (one per language) |
| Rounds | 3 |
| Local epochs per round | 1 |
| Aggregation (standard) | FedAvg |
| Aggregation (weighted) | q-FFL (q=2) |
| Starting checkpoint | Raw xlm-roberta-base (all variants) |

All federated variants start from the same raw pretrained checkpoint as the centralized baseline so that performance differences are attributable to the training procedure alone, not initialization.

---

## Approaches Compared

| Approach | Description |
|---|---|
| **Centralized** | All language data pooled, standard fine-tuning |
| **FedAvg** | Standard federated averaging weighted by dataset size |
| **Weighted FedAvg** | q-FFL: clients weighted by loss^q × size, upweighting underperforming languages |

---

## Results

### MLQA — Primary Evaluation

| Language | Central EM | Central F1 | FedAvg EM | FedAvg F1 | Weighted EM | Weighted F1 |
|---|---|---|---|---|---|---|
| English | 67.16 | 79.44 | 67.16 | 80.41 | 66.38 | 79.65 |
| Hindi | 45.36 | 64.00 | 49.70 | 67.63 | 47.14 | 66.42 |
| Spanish | 44.00 | 69.25 | 48.40 | 70.14 | 49.60 | 71.27 |
| German | 42.58 | 60.64 | 46.48 | 63.02 | 46.48 | 63.69 |
| Chinese | 18.25 | 60.46 | 43.85 | 67.29 | 43.06 | 67.49 |
| **Average** | 43.47 | 66.76 | 51.14 | 69.70 | 50.53 | 69.70 |
| **Max-Min Gap (F1)** | — | 18.98 | — | 17.39 | — | 17.56 |

> Chinese F1 uses character-level evaluation. The standard whitespace-tokenized SQuAD metric collapses F1 toward EM for Chinese, understating performance. Character-level comparison is applied specifically for zh.

### XQuAD — Zero-Shot Cross-Dataset Transfer

| Language | Central F1 | FedAvg F1 | Weighted F1 |
|---|---|---|---|
| English | 83.55 | 83.64 | 83.47 |
| Hindi | 71.11 | 73.85 | 73.86 |
| Spanish | 75.68 | 79.03 | 78.62 |
| German | 73.02 | 76.10 | 76.57 |
| Chinese | 63.72 | 76.96 | 77.32 |
| **Average** | 73.41 | 77.92 | 77.97 |
| **Max-Min Gap (F1)** | 19.83 | 9.80 | 9.61 |

> Models were not trained on any XQuAD data. Results demonstrate zero-shot cross-dataset generalization.

### Federated Convergence (Eval Loss per Round)

| Round | FedAvg | Weighted FedAvg |
|---|---|---|
| 1 | 1.967 | 2.166 |
| 2 | 1.534 | 1.538 |
| 3 | 1.512 | 1.620 |

---

## Key Findings

- **FedAvg outperforms centralized on every language** on both MLQA and XQuAD — attributed to the elimination of cross-language gradient interference in centralized pooled training.
- **The cross-lingual performance gap narrows under federation** even without fairness-aware aggregation. The max-min F1 gap halves from 19.83 to 9.80 on XQuAD under plain FedAvg.
- **q-FFL provides marginal additional fairness improvement** (XQuAD gap: 9.80 → 9.61) while matching FedAvg's average F1 exactly on MLQA (69.70), indicating that language partitioning itself accounts for most of the improvement.
- **Results generalize across datasets**: the federated advantage persists on XQuAD zero-shot transfer, confirming that federated training produces more robust multilingual representations rather than benchmark-specific gains.
- **Chinese benefits most from federation**: +6.83 F1 on MLQA, +13.24 F1 on XQuAD under FedAvg vs centralized.

---

## Limitations and Future Work

Integration of Differential Privacy via DP-SGD (Opacus) was explored but not completed within the scope of this work due to hardware constraints — full fine-tuning DP-SGD on a model of XLM-RoBERTa's scale requires per-sample gradient computation that is prohibitively memory-intensive without parameter-efficient methods such as LoRA. Future work should investigate DP-FedAvg with LoRA adapters, following DP-DyLoRA (Kerimi et al. 2024), which has not yet been tested on multilingual extractive QA. Additionally, whether DP noise disproportionately harms low-resource-language clients — compounding the language imbalance problem — remains an open and unstudied question in the federated multilingual setting.

XTREME evaluation across all trained models is also identified as a direction for future work, as no existing study has tested cross-task transfer from QA fine-tuning under federated conditions.

---

## Installation

```bash
pip install -r requirements.txt
```

```
# requirements.txt
torch==2.11.0
transformers==4.52.4
datasets==3.6.0
huggingface_hub==0.34.4
tokenizers==0.21.4
accelerate==1.10.1
evaluate==0.4.5
flwr==1.31.0
opacus==1.6.0
numpy==2.3.2
pandas==3.0.3
scikit-learn==1.9.0
matplotlib==3.11.0
jupyter==1.1.1
ipykernel==6.30.1
```
