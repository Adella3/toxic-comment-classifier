# Toxic Comment Classifier

> **"Which online conversations are becoming harmful — and how do we catch them at scale?"**

A binary text classification system for detecting toxic comments, comparing Logistic Regression and XGBoost on 223,000+ online comments. Built for UTS 31005 Machine Learning (2025).

---

## Results

| Model | F1 | Precision | Recall | AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.748** | 0.724 | 0.773 | **0.968** |
| XGBoost (final) | 0.716 | 0.769 | 0.670 | 0.948 |

Logistic Regression outperformed XGBoost overall — a finding that reflects the importance of model-problem fit over algorithmic complexity.

---

## The Problem

Toxic content moderation at scale is a classification problem with real consequences: miss too many toxic comments (high false negatives) and harmful content spreads; flag too many normal comments (high false positives) and users lose trust. This project focused on understanding that trade-off — not just maximizing accuracy.

---

## Dataset

**Jigsaw Multilingual Toxic Comment Classification** (Kaggle)

| Split | Total | Toxic | Non-toxic | Toxic % |
|---|---|---|---|---|
| Train | 178,839 | 17,107 | 161,732 | 9.57% |
| Validation | 22,355 | 2,139 | 20,216 | 9.57% |
| Test | 22,355 | 2,138 | 20,217 | 9.56% |

Class imbalance: ~90% non-toxic vs ~10% toxic. Addressed using `class_weight="balanced"` (LR) and `scale_pos_weight` (XGBoost).

---

## Approach

### Feature Engineering
- **TF-IDF vectorizer**: unigrams + bigrams, min_df=3, max 200,000 features
- Fitted on train only — no data leakage into validation/test

### Models
**Logistic Regression** (baseline)
- L2 regularization, C=0.5
- `liblinear` solver (well-suited for sparse TF-IDF features)
- Balanced class weights

**XGBoost** (advanced)
- Histogram-based splits (`tree_method=hist`)
- Early stopping (50 rounds patience)
- `scale_pos_weight` for class imbalance

### Threshold Tuning
Default threshold (0.5) was not optimal for imbalanced data. Both models were tuned on validation data to find the F1-maximizing threshold:
- LR optimal threshold: **0.64**
- XGBoost optimal threshold: **0.71**

---

## Key Findings

**Where LR beats XGBoost:** AUC (0.968 vs 0.948) — LR's probability estimates are better calibrated for ranking toxic content.

**Where both models struggle:** Long comments (200+ words). F1 drops from ~0.81 (short) to ~0.66 (long) — longer context introduces ambiguity that n-gram models miss.

**Top toxic signals (LR weights):** `stupid`, `fuck`, `shit`, `idiot`, `ass` — the model correctly learned that certain words are highly predictive, while also surfacing subtler patterns.

**False Negative examples worth noting:**
```
"ur vaginal area is loose"           → predicted non-toxic (borderline phrasing)
"LISTEN YA HUMP! QUIT VANDALIZING!!" → predicted non-toxic (domain-specific jargon)
```
These cases reveal that toxicity is contextual — a finding that matters when deciding where human review is still needed.

---

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `TF-IDF` · `pandas` · `NumPy` · `Matplotlib` · Google Colab

---

## Files

| File | Description |
|---|---|
| `Toxic_Comment_Classification.ipynb` | Full pipeline: preprocessing, training, evaluation, error analysis |

---

## Context

Individual project. UTS Bachelor of IT — Data Analytics (2025).
