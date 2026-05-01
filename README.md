# LLM-Augmented Sequence Modeling for Explainable ICU Diagnosis Prediction

A two-stage pipeline for early ICU diagnosis prediction that combines a sequence model with a MedGemma LLM reasoning layer. Built on MIMIC-IV critical care data.

---

## Overview

ICU patients are classified into diagnostic groups using the first 12 hours of EHR time-series data (vital signs, labs, medications, procedures). A bidirectional LSTM or LightGBM model generates a shortlist of candidate diagnoses, which a MedGemma 27B reasoning layer then reviews alongside a structured clinical summary to select the single most appropriate grouping.

Two label spaces were explored:

- **21-class**: A broad set of clinically meaningful groupings spanning six organ-system domains (Infectious, Cardiovascular, Respiratory, Neurological, Renal/GI/Hepatic, Toxic/Metabolic & Surgical), trained on 14,968 ICU stays.
- **5-class**: A focused subset of five high-signal conditions — Sepsis, Stroke/Cerebrovascular, Acute MI, Coronary Artery Disease, and Heart Failure — trained on the same cohort but evaluated on a 2,484-case held-out test set.

---

## Repository Structure

```
├── base model training/
│   ├── 21_LSTM_training.ipynb         # Bidirectional LSTM, 21-class label space
│   ├── 21_lightgbm_training.ipynb     # LightGBM, 21-class label space
│   ├── 5_LSTM_training.ipynb          # Bidirectional LSTM, 5-class label space
│   └── 5_LightGBM_training.ipynb      # LightGBM, 5-class label space
│
├── LLM layer/
│   ├── BLINDED_single_medgemma_final.ipynb             # MedGemma on 21-class LSTM (blind)
│   ├── NONBLINDED_single_medgemma_final.ipynb          # MedGemma on 21-class LSTM (non-blind)
│   ├── 5_LSTM_BLINDED_single_medgemma_final.ipynb      # MedGemma on 5-class LSTM (blind)
│   └── 5_LIGHTGBM_BLINDED_single_medgemma_final.ipynb # MedGemma on 5-class LightGBM (blind)
│
├── LLM results/
│   ├── LLM_results_21_LSTM_BLIND.csv
│   ├── LLM_results_5_LSTM.csv
│   └── LLM_results_5_lightGBM.csv
│
└── data preprocessing/
    └── v6_MIMIC_PREPROCESSING_final.ipynb
```

---

## Models

### Bidirectional LSTM
Trained end-to-end on 12-hour hourly time-series windows, represented as `(12, F)` tensors with interleaved value and observation-mask columns. Architecture: 2-layer BiLSTM (hidden dim 256) with 4-head self-attention, residual connection, LayerNorm, global average pooling, and a two-layer classification head. Trained with Focal Loss, AdamW, and cosine annealing with warm restarts.

### LightGBM
Trained on 408 aggregate features derived from the same 68 variables used in the LSTM: mean, min, max, last observed value, linear slope, and missing rate per variable across the 12-hour window. This matches exactly what the LLM reasoning layer sees in its structured clinical summary prompt.

### MedGemma 27B Reasoning Layer
`google/medgemma-27b-text-it`, loaded in 4-bit NF4 quantization (~16–18 GB VRAM). Receives the base model's top-3 candidate list alongside a structured clinical summary (vital sign trends, labs, medications, procedures) stripped of any diagnosis-leaking fields. Constrained to select exactly one diagnosis from the candidate shortlist.

Two conditions were evaluated:
- **Blind**: candidates presented in random order with no LSTM rank or confidence scores
- **Non-blind**: candidates presented in LSTM rank order with associated probabilities

---

## Results

### Base Model Performance

| Metric | 21-class LSTM | 5-class LSTM | 5-class LightGBM |
|---|---|---|---|
| Test samples | 4,336 | 2,484 | 2,484 |
| Top-1 accuracy | 44.23% | 67.15% | 73.87% |
| Top-3 accuracy | 72.19% | 93.88% | 96.01% |
| Top-5 accuracy | 81.71% | — | — |
| F1 (macro) | 23.99% | 59.87% | 66.17% |
| F1 (weighted) | 39.97% | 66.39% | 72.63% |

### LLM Reasoning Layer (100-case pilot)

All four conditions were evaluated on 100-case stratified samples drawn from the held-out test set.

| | 21-class LSTM (blind) | 21-class LSTM (non-blind) | 5-class LSTM (blind) | 5-class LightGBM (blind) |
|---|---|---|---|---|
| Base model Top-1 | 40.0% | 41.0% | 65.0% | 72.0% |
| Agent Top-1 | 36.0% | 40.0% | 39.0% | 41.0% |
| Net change | −4.0% | −1.0% | −26.0% | −31.0% |
| True in top-3 (ceiling) | 80.0% | 78.0% | 96.0% | 98.0% |
| Agent disagreed | 48.0% | 4.0% | 61.0% | 63.0% |
| Both correct | 26 | 40 | 28 | 30 |
| Agent improved | 10 | 0 | 11 | 11 |
| Agent hurt | 14 | 1 | 37 | 42 |
| Both wrong | 50 | 59 | 24 | 17 |

The non-blind condition largely deferred to the LSTM (4% disagreement rate), functioning as a near-passthrough. The blind condition disagreed frequently but overrode the correct base model prediction more often than it rescued an incorrect one, resulting in net accuracy loss across all setups.

### Sepsis Subgroup

The LLM layer showed consistent gains on sepsis specifically, where the clinical presentation in vital signs and labs is distinctive enough for the reasoning layer to contribute positively.

| Setup | Base accuracy | Agent accuracy | Gain |
|---|---|---|---|
| 21-class LSTM (blind) | 62% (13/21) | 71% (15/21) | +9 ppts |
| 5-class LightGBM (blind) | 73% (24/33) | 94% (31/33) | +21 ppts |
| 5-class LSTM (blind) | 64% (21/33) | 97% (32/33) | +33 ppts |

---

## Data

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/), a freely available de-identified critical care database from Beth Israel Deaconess Medical Center. Access requires completion of a data use agreement and CITI training through PhysioNet. Data cannot be redistributed and is not included in this repository.

Four MIMIC-IV tables are used: `chartevents`, `labevents`, `inputevents`, and `procedureevents`. See `data preprocessing/v6_MIMIC_PREPROCESSING_final.ipynb` for the full preprocessing pipeline.

---

## Dependencies

**Core ML:** `torch`, `lightgbm`, `scikit-learn`, `transformers`, `accelerate`, `bitsandbytes`

**Data/utils:** `pandas`, `numpy`, `tqdm`, `joblib`, `huggingface_hub`

**Visualization:** `matplotlib`, `seaborn`

**Hardware:** LSTM and LightGBM training run on a standard GPU (tested on Google Colab). MedGemma 27B inference requires a GPU with at least 24 GB VRAM (A100 through Google Colab PRO used in this project).

---

## Acknowledgements

MIMIC-IV data was accessed via PhysioNet under a data use agreement. Use of this data requires independent credentialing — see [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/) for access instructions.

MedGemma is developed by Google DeepMind. Model weights are available via the [Hugging Face Hub](https://huggingface.co/google/medgemma-27b-text-it).
