# 🏦 Bank Marketing — Term Deposit Prediction

> **Author:** Ioannis Frangistas  
> **Dataset:** UCI Bank Marketing (`bank-additional-full.csv`)  
> **Task:** Binary classification — predict whether a client will subscribe to a term deposit

***

## 📋 Project Overview

This project implements a full end-to-end machine learning workflow on the UCI Bank Marketing dataset. A Portuguese bank ran telephone marketing campaigns to sell term deposits, and the goal is to predict which clients are likely to subscribe — enabling smarter, more cost-effective targeting.

***

## 🗂️ Repository Structure

```
bank-marketing-analysis/
├── data/
│   ├── raw/
│   │   └── bank-additional-full.csv       # Original UCI dataset
│   └── processed/
│       ├── bank_additional_clean_minimal.csv   # Cleaned (for EDA)
│       └── bank_additional_model_ready.csv     # One-hot encoded (for modeling)
├── notebooks/
│   └── complete_analysis.ipynb            # Full workflow notebook
├── README.md
└── requirements.txt
```

***

## 📊 Dataset

| Attribute | Value |
|---|---|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| Records (raw) | 41,188 |
| Records (after cleaning) | 41,170 |
| Features | 20 (after preprocessing) |
| Target | `deposit` — 1 = subscribed, 0 = not subscribed |
| Class balance | ~89% No / ~11% Yes (imbalanced) |

### Feature Groups

- **Demographic:** `age`, `job`, `marital`, `education`
- **Financial:** `housing`, `loan`
- **Campaign:** `contact`, `month`, `day_of_week`, `duration`, `campaign`, `previous`, `poutcome`, `prev_contact`
- **Macroeconomic:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

***

## ⚙️ Workflow

### 1. Data Cleaning & Preprocessing
- Standardized column names and trimmed string values
- Converted target `y` → numeric `deposit` (1/0)
- Replaced `"unknown"` with `NaN` across all categorical columns
- Engineered `prev_contact` flag from `pdays` (999 = no previous contact)
- Collapsed `basic.4y / .6y / .9y` education levels into a single `"basic"` category
- Dropped `default` (>20% missing, near-zero variance)
- Removed duplicate rows
- Applied 1st–99th percentile capping on `campaign`, `previous`, `age`
- One-hot encoding with `drop_first=True` for modeling

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics for numeric variables
- Target distribution and class imbalance visualization
- Univariate distributions: age, duration, campaign intensity
- Bivariate analysis: deposit rate by job, month, contact type, poutcome
- Macroeconomic indicator correlation heatmap

### 3. Modeling & Comparison
Three classifiers were trained and compared on a **70/30 stratified split**:

| Model | ROC AUC | Avg Precision | Best F1 | Threshold |
|---|---|---|---|---|
| **XGBoost** 🏆 | **0.946** | **0.649** | **0.643** | 0.74 |
| Random Forest | 0.941 | 0.630 | 0.626 | 0.66 |
| Logistic Regression | 0.935 | 0.577 | 0.616 | 0.65 |

All models used class imbalance handling:
- Logistic Regression: `class_weight='balanced'`
- Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` set to the negative/positive class ratio

### 4. Model Interpretation (SHAP)
The best model (XGBoost) was interpreted using SHAP TreeExplainer:

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|---|
| 1 | `duration` | 2.519 | Call duration — strongest predictor |
| 2 | `euribor3m` | 1.447 | 3-month Euribor rate |
| 3 | `cons.price.idx` | 0.362 | Consumer price index |
| 4 | `month_may` | 0.287 | Negative effect for May calls |
| 5 | `age` | 0.159 | Customer age |

***

## 📈 Key Findings

- **Call duration** is the single most predictive feature, though it is only observable after the call.
- **Macroeconomic conditions** (`euribor3m`, `cons.price.idx`) strongly influence subscription rates — lower interest rate environments favor deposits.
- **Campaign timing matters:** May calls are associated with lower conversion; October and March show higher success rates.
- **Prior campaign success** (`poutcome_success`) is a strong positive signal — returning successful contacts should be prioritized.
- **XGBoost** achieves ROC AUC ≈ 0.95, excellent performance for highly imbalanced marketing data.


***

## 📄 License

This project is released under ALL RIGHTS RESERVED LICENSE. View (Portfolio) purpose only. Unauthorized copying, modification, or distribution of this project's code and analysis is strictly prohibited.
Dataset source: [Moro et al., 2014] — UCI Machine Learning Repository. **Dataset is open-access under open license**

***

*© 2026 Ioannis Frangistas*
