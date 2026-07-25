# 🩺 Predicting Student Health Risk — Kaggle Playground Series S6E7

> **A 3-class classification problem (`at-risk` / `unhealthy` / `fit`) on 690k students, scored by balanced accuracy — with a heavily imbalanced target, informative missingness, and a full honest record of what worked and what didn't.**

<br>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-2E8B57?logo=leaflet)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-Model-FFCC00?logo=data%3Aimage%2Fpng)](https://catboost.ai/)
[![Optuna](https://img.shields.io/badge/Optuna-Tuning-6A5ACD)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 The Central Finding

**Best result: 0.95011 balanced accuracy on the public leaderboard** — achieved with a single tuned LightGBM model. Five follow-up techniques (ensembling, feature engineering, pseudo-labeling, alternative missing-value handling) were tried on top of it, and **none produced a reliable improvement.**

That negative result is the point of this repo: knowing when a technique isn't paying off — and trusting honest out-of-fold validation over wishful thinking — is as much a part of the skill as finding the winning model in the first place.

| Version | Approach | CV Balanced Accuracy | Public LB | Outcome |
|---|---|---|---|---|
| v1 | RandomForest baseline | 0.86212 | — | Got something working end-to-end |
| v2 | LightGBM, native categorical handling | 0.94907 | 0.94979 | Big jump over RF — model choice mattered most |
| **v3** | **LightGBM + Optuna tuning (20 trials)** | **0.94965** | **0.95011** | **✅ Best result — final submission** |
| v4 | CatBoost (GPU) | 0.94902 | 0.94945 | Comparable to LightGBM, slightly behind |
| v5 | Ensemble (v3 + v4, OOF-validated blend weight) | 0.94961 | not submitted | No real gain — best blend was ~80% LightGBM |
| v6 | Feature engineering (ratio/interaction + missingness flags) | 0.94966 | not submitted | No real gain — trees already captured this |
| v7 | Pseudo-labeling (confident test predictions added to training) | 0.94991 vs 0.95012 baseline | not submitted | ❌ Made things worse — reinforced majority class |
| v8 | Native missing-value handling + prior correction | 0.94985 | 0.94966 | Small CV gain didn't transfer to LB — within noise |

---

## 🔍 Key Findings

### What actually moved the needle

1. **Model choice (RF → LightGBM)** — by far the largest single improvement (**+~8.7 points** balanced accuracy).
2. **Hyperparameter tuning (Optuna)** — a small, real, repeatable gain (~+0.0005–0.0006), confirmed on both CV and leaderboard.
3. Everything else — ensembling, feature engineering, pseudo-labeling, alternative missing-value handling — **did not beat the tuned LightGBM baseline**, and each was tested honestly via proper out-of-fold validation rather than eyeballed.

### What the data itself revealed

| Insight | Detail |
|---|---|
| Class imbalance dominates the metric | Target is ~86% / 8% / 6% — a majority-only model scores just **0.33** balanced accuracy |
| Missingness is informative (MNAR) | Which values are missing correlates with the target — especially `sleep_duration`, `stress_level`, `physical_activity_level` |
| CV tracked the leaderboard closely | Gaps mostly within ±0.001 — local CV could be trusted for decisions instead of "spending" submissions |

---

## 📁 Repo Structure

```
student-health-risk-kaggle/
│
├── README.md
├── Student_Health_Risk_Concept_Book.docx
├── gitignore
│
├── notebooks/
│   ├── shr-v1-randomforest-baseline.ipynb
│   ├── shr-v2-lightgbm-baseline.ipynb
│   ├── shr-v3-lightgbm-optuna-tuned.ipynb          # final submitted model (0.95011 LB)
│   ├── shr-v4-catboost.ipynb
│   ├── shr-v5-ensemble-lightgbm-catboost.ipynb
│   ├── shr-v6-feature-engineering.ipynb
│   ├── shr-v7-pseudo-labeling.ipynb
│   └── shr-v8-native-missing-prior-correction.ipynb
│
└── outputs/
    ├── shr-v1-submission.csv
    ├── shr-v2-submission.csv
    ├── shr-v3-submission.csv                       # best submission (0.95011 LB)
    ├── shr-v4-submission.csv
    ├── shr-v5-submission.csv
    ├── shr-v6-submission.csv
    ├── shr-v7-submission.csv
    └── shr-v8-submission.csv
```

Each notebook is self-contained and was run on Kaggle against the competition's `train.csv` / `test.csv` (not included here — see the [competition data page](https://www.kaggle.com/competitions/playground-series-s6e7/data) to download).

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `shr-v1-randomforest-baseline.ipynb` | Beginner-friendly starter pipeline — EDA, preprocessing, RandomForest baseline |
| `shr-v2-lightgbm-baseline.ipynb` | Swaps RF for LightGBM with native categorical handling |
| `shr-v3-lightgbm-optuna-tuned.ipynb` | Optuna hyperparameter search + 5-fold CV — **best result** |
| `shr-v4-catboost.ipynb` | CatBoost (GPU), saves OOF + test probabilities for ensembling |
| `shr-v5-ensemble-lightgbm-catboost.ipynb` | Blends v3 + v4 via OOF-validated weight search |
| `shr-v6-feature-engineering.ipynb` | Adds missingness flags + ratio/interaction features |
| `shr-v7-pseudo-labeling.ipynb` | Adds confident test predictions as extra training data |
| `shr-v8-native-missing-prior-correction.ipynb` | Native NaN handling + class-prior probability correction |

---

## 🔗 Related Links

| Resource | Link |
|---|---|
| Kaggle Competition | [Playground Series — Season 6, Episode 7](https://www.kaggle.com/competitions/playground-series-s6e7) |
| Kaggle Dataset (OOF/test probability files) | [health-risk-oof-files](https://kaggle.com/datasets/4ae485da4f5c27c7c43cb1450ccb077d5773dc13297d7ef8295137c011995eae) |

The dataset above contains the LightGBM + CatBoost out-of-fold and test probability arrays used in `shr-v5-ensemble-lightgbm-catboost.ipynb` for the blending step.

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

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/vikram-kumar-204013263/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github)](https://github.com/vikraamkumar-ds)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Predicting Student Health Risk · Kaggle Playground Series S6E7 · Vikram Kumar · 2026*
