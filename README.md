# Explainable and Fair Recidivism Prediction

A graduate-level applied machine learning project that builds, explains, and audits predictive models for two-year recidivism using the ProPublica COMPAS dataset — with an emphasis on **fairness**, **explainability**, and **responsible AI**.

---

## Overview

In 2016, ProPublica published an investigation arguing that COMPAS — a proprietary recidivism risk-assessment tool used by US courts — was significantly more likely to incorrectly flag Black defendants as high-risk, and more likely to incorrectly clear white defendants as low-risk. This project reproduces and extends that analysis using modern, open-source machine learning tooling.

The goal is not to build a better COMPAS. It is to demonstrate what a *responsible* ML pipeline looks like: one that treats predictive performance, model explainability, and fairness auditing as co-equal objectives rather than treating fairness as an afterthought.

---

## Key Results

### Predictive Performance (XGBoost, held-out test set)

| Metric | Score |
|---|---|
| Accuracy | 68.8% |
| Precision | 71.0% |
| Recall | 66.0% |
| F1 | 68.4% |
| **ROC-AUC** | **0.746** |

The best model achieves **0.746 AUC** using only publicly available, interpretable features — matching or exceeding the proprietary COMPAS tool (~0.71 AUC) without any access to its internal scoring logic.

---

### Fairness Audit (XGBoost, test set)

| Sensitive Attribute | Demographic Parity Difference | Equalized Odds Difference |
|---|---|---|
| Race | **0.721** | **0.565** |
| Sex | 0.280 | 0.256 |

**False Positive and Negative Rates by Race:**

| Group | FPR | FNR | Selection Rate |
|---|---|---|---|
| African-American | 38.1% | 24.0% | 59.1% |
| Caucasian | 22.2% | 46.2% | 38.5% |
| Hispanic | 20.0% | 56.5% | 28.6% |
| Other | 11.5% | 47.1% | 27.9% |

> ⚠️ Asian and Native American groups had fewer than 20 samples in the test set; their metrics are not statistically reliable and are excluded from the summary above.

Despite **race being excluded as a direct model input**, a 16-percentage-point gap in false positive rates between African-American and Caucasian defendants persists. This is proxy bias: features like `priors_count` are statistically correlated with race due to well-documented disparities in policing, so the model learns race indirectly. Removing a protected attribute from model inputs is not sufficient to guarantee equitable outcomes.

The simultaneous presence of a higher FPR for African-American defendants *and* a higher FNR for Caucasian defendants is not a fixable bug — it is a structural consequence of differing base recidivism rates between groups. Chouldechova (2017) proved formally that no single model can simultaneously equalize both error rates when base rates differ, which is the same mathematical result at the heart of the original ProPublica/Northpointe dispute.

---

## Project Structure

```
├── Explainable_and_Fair_Recidivism_Prediction.ipynb   # Main notebook
└── README.md
```

The notebook is structured as follows:

| Section | Content |
|---|---|
| Introduction | Recidivism prediction, COMPAS controversy, responsible AI framing |
| Data Loading | Auto-download from ProPublica GitHub, manual upload fallback |
| Data Cleaning | ProPublica-standard filters, missing values, duplicates, type correction |
| EDA | Target distribution, recidivism rates by race/sex/age, correlation heatmap |
| Feature Engineering | One-hot encoding, 70/15/15 stratified split, leakage-safe scaling |
| Baseline Modeling | Logistic Regression with full evaluation suite |
| Advanced Modeling | Random Forest, XGBoost, LightGBM with RandomizedSearchCV + 5-fold CV |
| Class Imbalance | SMOTE vs class-weighting comparison |
| Explainability | SHAP summary, bar, dependence, and waterfall plots |
| Fairness Analysis | Fairlearn audit across race and sex; FPR/FNR/TPR/selection rate |
| Responsible AI | Historical context, proxy bias, fairness-accuracy tradeoffs |
| Results & Discussion | Final model comparison, interpretation, policy implications |
| Limitations & Future Work | Honest scope boundaries and next steps |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| Imbalance | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Fairness | `fairlearn` |
| Visualization | `matplotlib`, `seaborn` |

---

## How to Run

The notebook is designed to run top-to-bottom in **Google Colab** with no setup required.

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Explainable_and_Fair_Recidivism_Prediction.ipynb`
3. Run all cells (`Runtime → Run all`)

The dataset downloads automatically from ProPublica's public GitHub repository. If that fails, a clearly marked fallback cell prompts for manual upload.

All random seeds are fixed (`SEED = 42`) for full reproducibility.

---

## Responsible AI Note

This project is an **educational and portfolio demonstration**. It is not intended for deployment as a real-world risk-assessment system. The fairness analysis in this notebook illustrates precisely why such systems require ongoing scrutiny, transparent auditing, and human oversight — not why they should be built and deployed uncritically.

Key considerations discussed in the notebook:

- A model can reproduce racial disparities without using race as a direct input (proxy bias)
- Accuracy is a necessary but not sufficient criterion for deploying models in high-stakes legal contexts
- Fairness metrics involve irresolvable tradeoffs when base rates differ across groups — this requires policy judgment, not just better algorithms
- Predictive tools should support, not replace, human judgment in decisions that affect individual liberty

---

## References

- Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing). *ProPublica*.
- Chouldechova, A. (2017). [Fair Prediction with Disparate Impact](https://arxiv.org/abs/1703.00056). *Big Data*.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807). *ITCS 2017*.
- Lundberg, S. & Lee, S.I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). *NeurIPS*.
- Bird, S. et al. (2020). [Fairlearn: A toolkit for assessing and improving fairness in AI](https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/). *Microsoft Research*.

---

## Author

Built as a portfolio project demonstrating end-to-end responsible machine learning practice: predictive modeling, explainability, and quantitative fairness auditing on a real-world, socially consequential dataset.
