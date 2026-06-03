# Sexism Detection in Spanish Tweets — Two-Stage Pipeline

> Research conducted in collaboration with the **United Nations International Computing Centre (UNICC)** | Submitted to SIMBIG 2026

---

## Overview

This repository contains the code for a two-stage transformer-based pipeline for detecting and categorizing sexist content in Spanish-language tweets. The pipeline was benchmarked across four Spanish and Twitter-specific pre-trained language models: **BETO**, **RoBERTuito**, **XLM-T**, and **BERTIN**.

**Stage 1** performs binary classification (sexist / not sexist) on a unified corpus of 44,352 tweets.  
**Stage 2** categorizes confirmed sexist tweets into one of four semantic classes.

Fine-tuned open-source models proved competitive with prompt-engineered LLM approaches (GPT-4o) at zero inference cost, supporting their viability for resource-constrained moderation systems such as the UNICC.

---

## Pipeline Architecture

```
Incoming tweet
      │
      ▼
┌─────────────────────────┐
│  Stage 1 — Binary       │  44,352 tweets · BETO · RoBERTuito · XLM-T · BERTIN
│  Classifier             │
└─────────────────────────┘
      │
      ├── Not sexist → Pipeline ends
      │
      ▼ Sexist
┌─────────────────────────┐
│  Stage 2 — Multi-class  │  Sexist tweets only · 4 semantic categories
│  Classifier             │
└─────────────────────────┘
      │
      ├── Sexual Insults & Objectification
      ├── Gendered Stereotypes & Insults
      ├── General Hostile Language
      └── Threats & Physical Harm
```

---

## Dataset

The unified corpus of **44,352 Spanish-language tweets** was assembled from four publicly available datasets:

| Dataset | Language | Size | Labels |
|---|---|---|---|
| EXIST 2021 | EN + ES | 6,930 | Binary |
| HatEval (SemEval 2019) | EN + ES | 15,126 | Binary + fine-grained |
| ManRosSexism Twitter MeToo | ES | 2,282 | Binary |
| EDOS | EN | 20,016 | Binary + multi-class |

English tweets were translated into Spanish using Google Translate (sourced from IE University's prior work). The multi-class subset used for Stage 2 contains **8,612 labeled sexist tweets** consolidated from EDOS's original 11 subcategories into 4 semantically distinct classes.

**Stage 2 class distribution:**

| Category | Count | % |
|---|---|---|
| General Hostile Language | 6,651 | 77% |
| Gendered Stereotypes & Insults | 1,071 | 12% |
| Sexual Insults & Objectification | 447 | 5% |
| Threats & Physical Harm | 443 | 5% |

---

## Models

| Model | Architecture | Pretraining Corpus | Domain |
|---|---|---|---|
| [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) | BERT | Large Spanish corpus | General |
| [BERTIN](https://huggingface.co/bertin-project/bertin-roberta-base-spanish) | RoBERTa | Spanish web + books | General |
| [XLM-T](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base) | XLM-RoBERTa | 198M multilingual tweets | Twitter |
| [RoBERTuito](https://huggingface.co/pysentimiento/robertuito-base-uncased) | RoBERTa | 500M Spanish tweets | Twitter |

---

## Results

### Stage 1 — Binary Classification (n = 6,653 test tweets)

| Model | Accuracy | Macro F1 | Weighted F1 | Precision | Recall |
|---|---|---|---|---|---|
| **BETO ★** | **0.8258** | **0.7856** | **0.8251** | 0.7880 | 0.7833 |
| RoBERTuito | 0.8247 | 0.7830 | 0.8235 | 0.7873 | 0.7791 |
| XLM-T | 0.8234 | 0.7829 | 0.8228 | 0.7849 | 0.7810 |
| BERTIN | 0.8130 | 0.7789 | 0.8159 | 0.7716 | **0.7887** |

★ Best overall · Primary metric: Macro F1

### Stage 2 — Multi-class Classification (n = 729 test tweets)

| Model | Accuracy | Macro F1 | Weighted F1 | Precision | Recall |
|---|---|---|---|---|---|
| **BETO ★** | **0.7933** | **0.5852** | **0.8045** | 0.5697 | 0.6128 |
| RoBERTuito | 0.7717 | 0.5762 | 0.7889 | 0.5464 | **0.6324** |
| XLM-T | 0.7693 | 0.5633 | 0.7842 | 0.5367 | 0.6115 |
| BERTIN | 0.7918 | 0.5526 | 0.7940 | **0.5738** | 0.5485 |

★ Best overall · Primary metric: Macro F1

**Key findings:**
- All Stage 1 models cluster within 0.007 Macro F1 points, pointing to a **data-quality ceiling** rather than an architectural problem
- The ~0.20 point F1 drop from Stage 1 to Stage 2 reflects **structural class imbalance** (15:1 ratio), not model failure
- **BETO** is the strongest overall model; **BERTIN** and **RoBERTuito** are better suited for recall-sensitive deployments where missing a sexist tweet carries a higher cost than a false alarm
- Fine-tuned open-source models are **competitive with GPT-4o prompt engineering** at zero inference cost

---

## Notebooks

| Notebook | Model | Description |
|---|---|---|
| [01_BERTIN_two_stage_pipeline.ipynb](notebooks/01_BERTIN_two_stage_pipeline.ipynb) | BERTIN | Full two-stage pipeline |
| [02_BETO_two_stage_pipeline.ipynb](notebooks/02_BETO_two_stage_pipeline.ipynb) | BETO | Full two-stage pipeline |
| [03_RoBERTuito_two_stage_pipeline.ipynb](notebooks/03_RoBERTuito_two_stage_pipeline.ipynb) | RoBERTuito | Full two-stage pipeline |
| [04_mDeBERTa_v3_two_stage_pipeline.ipynb](notebooks/04_mDeBERTa_v3_two_stage_pipeline.ipynb) | mDeBERTa-v3 | Incomplete — hardware constraints |
| [05_XLM_RoBERTa_two_stage_pipeline.ipynb](notebooks/05_XLM_RoBERTa_two_stage_pipeline.ipynb) | XLM-RoBERTa | Full two-stage pipeline |
| [06_error_analysis_RoBERTuito.ipynb](notebooks/error_analysis/06_error_analysis_RoBERTuito.ipynb) | RoBERTuito | Error analysis |
| [07_error_analysis_BETO.ipynb](notebooks/error_analysis/07_error_analysis_BETO.ipynb) | BETO | Error analysis |

---

## Training Setup

All models were fine-tuned using the HuggingFace Transformers library with PyTorch on an NVIDIA Tesla T4 GPU (Google Colab).

| Hyperparameter | Value |
|---|---|
| Max sequence length | 128 tokens |
| Epochs | Up to 4 (early stopping, patience=2) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Loss function | Weighted cross-entropy |
| Dataset split | 70 / 15 / 15 (train / val / test), stratified |
| Optimization criterion | Macro F1 on validation set |

---

## Limitations

- **Class imbalance:** General Hostile Language accounts for 77% of Stage 2 training data, placing a ceiling on Macro F1 regardless of model choice
- **Machine translation:** English tweets were translated with Google Translate, which may introduce artifacts with informal social media language
- **Hardware constraints:** XLM-RoBERTa-large and mDeBERTa-v3 could not be fully evaluated due to T4 GPU memory limits
- **Dialect mismatch:** Training data skews toward Latin American Spanish; BETO and BERTIN were pretrained on Iberian Spanish corpora
- **Single seed:** Variance is not reported; small differences between models may not be statistically significant
- **Positionality:** The authors do not have formal training in gender or race studies, which shaped label interpretation during dataset construction

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{thomas2026sexism,
  title     = {Two-Stage Sexism Detection Pipeline for Spanish Twitter},
  author    = {Thomas, Gizela},
  booktitle = {SIMBIG 2026},
  year      = {2026}
}
```

---

## Acknowledgements

This research was conducted in collaboration with the **United Nations International Computing Centre (UNICC)**. Dataset sources include EXIST 2021, HatEval (SemEval 2019), ManRosSexism Twitter MeToo, and EDOS. The multi-class dataset was sourced from IE University's research team.
