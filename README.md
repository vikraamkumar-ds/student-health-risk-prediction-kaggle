# 🩺 Predicting Student Health Risk — Kaggle Playground Series S6E7

> **A 3-class classification problem (`at-risk` / `unhealthy` / `fit`) on 690,088 students, scored by balanced accuracy — with a heavily imbalanced target, informative missingness, and a full honest record of what worked and what didn't.**

<br>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-2E8B57?logo=leaflet)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Model-FFCC00)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-6A5ACD)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 The Central Finding

**Best result: 0.95020 balanced accuracy on the public leaderboard** — from the feature-engineering notebook (v6). The tuned LightGBM baseline (v3, 0.95011) is a very close second, and both are selected as final submissions to hedge against the private leaderboard (the other 80% of test data) ranking them differently.

| Version | Approach | CV Balanced Accuracy | Public LB | Outcome |
|---|---|---|---|---|
| v1 | RandomForest baseline | 0.86212 | — | Got something working end-to-end |
| v2 | LightGBM, native categorical handling | 0.94907 | 0.94979 | Big jump over RF — model choice mattered most |
| v3 | LightGBM + Optuna tuning (20 trials) | 0.94965 | 0.95011 | Strong, repeatable result — selected as a final submission |
| v4 | CatBoost (GPU) | 0.94902 | 0.94945 | Comparable to LightGBM, slightly behind |
| v5 | Ensemble (v3 + v4, OOF-validated blend weight) | 0.94961 | 0.94995 | Small gain over v4 alone, still below v3 |
| **v6** | **Feature engineering (ratio/interaction + missingness flags)** | **0.94966** | **0.95020** | **✅ Best result — selected as a final submission** |
| v7 | Pseudo-labeling (confident test predictions added to training) | 0.94991 vs 0.95012 (held-out check) | 0.95000 | Scored well on LB despite a locally-measured drop — see note below |
| v8 | Native missing-value handling + prior correction | 0.94985 | 0.94966 | Small CV gain didn't hold up on LB |

> **Note on CV vs. leaderboard:** local out-of-fold validation predicted v5/v6/v7 wouldn't beat v3 — the leaderboard disagreed for all three, and v6 ended up on top. All scores here sit within a ~0.0007 band, which is inside the noise range for this competition. This is less "the local testing was wrong" and more a reminder that the public leaderboard is itself computed on only 20% of the test set — close results like these aren't fully resolved by either signal alone.

---

## 🔍 Key Findings

### What actually moved the needle

| Lever | Effect |
|---|---|
| **Model choice** (RandomForest → LightGBM) | +~8.7 points balanced accuracy — by far the largest single improvement |
| **Hyperparameter tuning** (Optuna, 20 trials) | +~0.0005–0.0006 over baseline LightGBM — small, real, and repeatable |
| **Feature engineering** (v6) | Ended up the best submission (0.95020), despite a flat result in local OOF testing |
| Ensembling, pseudo-labeling, missing-value strategy | Scored competitively (0.94966–0.95000) but did not clearly beat v3/v6 — all differences here are small enough to sit within normal CV/LB noise for this competition |

### What the data itself revealed

| Insight | Detail |
|---|---|
| Class imbalance dominates the metric | Target is ~86% at-risk / 8% unhealthy / 6% fit — a majority-only model scores just **0.33** balanced accuracy |
| Missingness is informative (MNAR) | *Which* values are missing correlates with the target — especially `sleep_duration`, `stress_level`, `physical_activity_level` |
| CV tracked the leaderboard closely | Gaps mostly within ±0.001 across every submission — local CV could be trusted for decisions instead of "spending" submissions to check |

### Feature Importance Snapshot

| Rank | Feature | Note |
|---|---|---|
| 1 | `bmi` | Strongest single numeric predictor |
| 2 | `sleep_duration` | Interacts heavily with `stress_level` |
| 3 | `exercise_duration` | Conditions BMI and heart-rate signals |
| 4 | `heart_rate` | Moderate on its own; stronger in ratio features |
| 5 | `calorie_expenditure` | Correlated with `step_count` |
| — | `gender` | Weakest categorical predictor |

---

## 📁 Repo Structure

```
student-health-risk-kaggle/
│
├── README.md
├── Student_Health_Risk_Concept_Book.docx     # Plain-language glossary: what we did & why
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── v1_randomforest_baseline.ipynb
│   ├── v2_lightgbm_baseline.ipynb
│   ├── v3_lightgbm_optuna_tuned.ipynb          # final submitted model (0.95011 LB)
│   ├── v4_catboost.ipynb
│   ├── v5_ensemble_lightgbm_catboost.ipynb
│   ├── v6_feature_engineering.ipynb
│   ├── v7_pseudo_labeling.ipynb
│   └── v8_native_missing_prior_correction.ipynb
│
└── outputs/
    ├── v1_submission.csv
    ├── v2_submission.csv
    ├── v3_submission.csv                       # best submission (0.95011 LB)
    ├── v4_submission.csv
    ├── v5_submission.csv
    ├── v6_submission.csv
    ├── v7_submission.csv
    └── v8_submission.csv
```

> Competition data (`train.csv`, `test.csv`, `sample_submission.csv`) is not included here — download it from the [competition's Data page](https://www.kaggle.com/competitions/playground-series-s6e7/data) and place it wherever your notebook's `DATA_DIR` points to.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `v1_randomforest_baseline.ipynb` | Beginner-friendly starter pipeline — EDA, preprocessing, RandomForest baseline |
| `v2_lightgbm_baseline.ipynb` | Swaps RandomForest for LightGBM with native categorical handling |
| `v3_lightgbm_optuna_tuned.ipynb` | Optuna hyperparameter search + 5-fold CV — **best result** |
| `v4_catboost.ipynb` | CatBoost (GPU), saves OOF + test probabilities for ensembling |
| `v5_ensemble_lightgbm_catboost.ipynb` | Blends v3 + v4 via OOF-validated weight search — LB 0.94995 |
| `v6_feature_engineering.ipynb` | Adds missingness flags + ratio/interaction features — **LB 0.95020, best result** |
| `v7_pseudo_labeling.ipynb` | Adds confident test predictions as extra training data — LB 0.95000 |
| `v8_native_missing_prior_correction.ipynb` | Native NaN handling + class-prior probability correction |

> Notebooks are self-contained and were run on Kaggle. Run them in order to reproduce the full progression, or jump straight to `v3` for the best single result.

---

## 🚀 Reproduce Locally

### 1. Clone the repo
```bash
git clone https://github.com/vikraamkumar-ds/student-health-risk-prediction-kaggle.git
cd student-health-risk-prediction-kaggle
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the competition data
Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [competition's Data page](https://www.kaggle.com/competitions/playground-series-s6e7/data) and place them in a `data/` folder.

### 4. Update `DATA_DIR` and run
Each notebook has a `DATA_DIR` variable near the top — point it at your local `data/` folder, then run all cells top to bottom.

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
lightgbm
catboost
optuna
```

---

## 🔗 Related Links

| Resource | Link |
|---|---|
| Kaggle Competition | [Playground Series — Season 6, Episode 7](https://www.kaggle.com/competitions/playground-series-s6e7) |
| Kaggle Dataset (OOF/test probability files) | [health-risk-oof-files](https://kaggle.com/datasets/4ae485da4f5c27c7c43cb1450ccb077d5773dc13297d7ef8295137c011995eae) |

The dataset above contains the LightGBM + CatBoost out-of-fold and test probability arrays used in `v5_ensemble_lightgbm_catboost.ipynb` for the blending step.

---

## 📋 What I'd Try Next

- **Conditional modeling on the "rule" features** — community analysis found balanced accuracy on rows with complete `sleep_duration` / `stress_level` / `physical_activity_level` data reaches ~0.97, collapsing toward chance as those go missing. Modeling complete-key and incomplete-key rows separately is a promising, untested direction.
- **A genuinely different model type for ensembling** — LightGBM and CatBoost turned out to make very similar predictions here; a neural net or linear model might add real diversity.
- **Exact-value target encoding** for the ordinal categoricals (`stress_level`, `sleep_quality`, `physical_activity_level`), fitted safely inside each CV fold.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM, CatBoost |
| Hyperparameter Tuning | Optuna |
| Validation | Stratified K-Fold, Balanced Accuracy |
| Platform | Kaggle Notebooks (GPU-enabled for CatBoost) |

---

## 👤 Author

**Vikram Kumar**
BS Data Science

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/vikram-kumar-204013263/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/vikraamkumar-ds)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Predicting Student Health Risk · Kaggle Playground Series S6E7 · Vikram Kumar · 2026*
