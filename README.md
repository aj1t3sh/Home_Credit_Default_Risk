# Home Credit Default Risk Prediction

An end-to-end, production-grade Machine Learning pipeline for the Kaggle **Home Credit Default Risk** competition. This project predicts the probability of loan default (`TARGET = 1`) by integrating and aggregating 7 relational tabular data sources into a high-dimensional, high-signal feature matrix and training an optimized Gradient Boosted Decision Tree (LightGBM) model.

---

## 1. Project Overview

Many individuals struggle to access credit due to insufficient or non-existent traditional credit histories. Home Credit strives to broaden financial inclusion for the unbanked population by providing safe borrowing experiences using alternative data sources (telecom, transactional history, and prior repayment patterns).

This project implements a modular, memory-efficient, and production-style classification pipeline designed to execute efficiently on standard hardware while generating an exact, fully validated Kaggle `submission.csv`.

---

## 2. Kaggle Competition

- **Competition Name**: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **Task**: Binary Classification & Probability Estimation
- **Target Variable**: `TARGET` (0 = Repaid on time, 1 = Encountered payment difficulties / defaulted)
- **Primary Identifier**: `SK_ID_CURR`
- **Evaluation Metric**: ROC-AUC (Area Under the Receiver Operating Characteristic Curve)

---

## 3. Problem Statement

Given an applicant's demographic profile, current loan application details, credit bureau history, past credit applications, POS cash balances, credit card records, and installment payment timelines:
$$\hat{p} = P(\text{TARGET} = 1 \mid \mathbf{X})$$
The primary objective is to rank applicants accurately by risk so that lenders can maximize loan approvals for reliable borrowers while mitigating credit default losses.

---

## 4. Dataset Architecture

The dataset comprises 7 relational tables linked via primary and foreign keys:

| Dataset | Records | Description | Primary Key / Join Key |
| :--- | :--- | :--- | :--- |
| `application_train.csv` | 307,511 | Main loan application table with target labels | `SK_ID_CURR` |
| `application_test.csv` | 48,744 | Main test set for Kaggle submission | `SK_ID_CURR` |
| `bureau.csv` | 1,716,428 | Previous credits from other financial institutions | `SK_ID_CURR`, `SK_ID_BUREAU` |
| `bureau_balance.csv` | 27,299,925 | Monthly credit balance history for bureau credits | `SK_ID_BUREAU` |
| `previous_application.csv` | 1,670,214 | Prior loan applications with Home Credit | `SK_ID_CURR`, `SK_ID_PREV` |
| `POS_CASH_balance.csv` | 10,001,358 | Monthly balance snapshots for POS/cash loans | `SK_ID_CURR`, `SK_ID_PREV` |
| `installments_payments.csv` | 13,605,401 | Historical installment payments and delays | `SK_ID_CURR`, `SK_ID_PREV` |
| `credit_card_balance.csv` | 3,840,312 | Monthly credit card balance and drawing records | `SK_ID_CURR`, `SK_ID_PREV` |

---

## 5. Data Preprocessing

1. **Memory Reduction (`reduce_mem_usage`)**: Automatically casts integer and floating-point columns to optimal lower-precision types (`int8`, `int16`, `int32`, `float32`), reducing total memory footprint by over 60%.
2. **Domain Anomaly Handling**:
   - `DAYS_EMPLOYED`: Addressed the anomalous placeholder `365,243` (1,000 years in future) by replacing it with `NaN` and generating a boolean flag `DAYS_EMPLOYED_ANOM`.
   - `CODE_GENDER` & `ORGANIZATION_TYPE`: Handled rare `'XNA'` values.
3. **Categorical Encoding**:
   - **Binary Features (<= 2 categories)**: Label encoded (e.g. `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_CONTRACT_TYPE`, `EMERGENCYSTATE_MODE`).
   - **Multi-class Features (> 2 categories)**: One-Hot Encoded with dummy indicators for missing values.
   - Exact column alignment enforced across training and test feature sets.

---

## 6. Feature Engineering

A total of **847 engineered features** were constructed by combining domain ratios and aggregating relational secondary tables:

### A. Application-Level Domain Features
- **Financial ratios**:
  - `INCOME_CREDIT_PERC` = `AMT_INCOME_TOTAL / AMT_CREDIT`
  - `ANNUITY_INCOME_PERC` = `AMT_ANNUITY / AMT_INCOME_TOTAL`
  - `PAYMENT_RATE` = `AMT_ANNUITY / AMT_CREDIT`
  - `CREDIT_GOODS_PERC` = `AMT_CREDIT / AMT_GOODS_PRICE`
- **Demographic & Employment**:
  - `INCOME_PER_PERSON` = `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
  - `INCOME_PER_CHILD` = `AMT_INCOME_TOTAL / (1 + CNT_CHILDREN)`
  - `AGE_YEARS` = `-DAYS_BIRTH / 365.25`
  - `EMPLOYED_YEARS` = `-DAYS_EMPLOYED / 365.25`
  - `DAYS_EMPLOYED_PERC` = `DAYS_EMPLOYED / DAYS_BIRTH`
- **External Score Interactions**:
  - `EXT_SOURCES_MEAN`, `EXT_SOURCES_STD`, `EXT_SOURCES_MIN`, `EXT_SOURCES_MAX`, `EXT_SOURCES_SUM`, `EXT_SOURCES_PROD`
  - Cross-ratios: `EXT_SOURCE_1_2_RATIO`, `EXT_SOURCE_2_3_RATIO`, `EXT_SOURCE_1_3_RATIO`

### B. Relational Table Aggregations (Grouped by `SK_ID_CURR`)
1. **Bureau & Bureau Balance**:
   - Monthly delinquency status distribution (`STATUS` 0–5, C, X).
   - Counts of active vs closed credits, debt-to-credit ratio (`BURO_DEBT_CREDIT_RATIO`), overdue-to-credit ratio, credit duration statistics (`DAYS_CREDIT` min, max, mean, var).
2. **Previous Applications**:
   - Count of past applications, approval rates, refusal rates, cancellation rates.
   - Application vs credit amount ratio (`APPLICATION_CREDIT_RATIO`), requested vs approved differences, `DAYS_DECISION` statistics.
3. **POS CASH Balance**:
   - Total contracts, contract status distributions, days past due (`SK_DPD`, `SK_DPD_DEF`), remaining installment counts.
4. **Installments Payments**:
   - Days Past Due (`DPD`), Days Before Due (`DBD`).
   - Payment-to-installment ratio (`PAYMENT_PERC`), underpayment/overpayment amount (`PAYMENT_DIFF`), installment payment sums and averages.
5. **Credit Card Balance**:
   - Utilization ratios: `BALANCE_TO_LIMIT`, `DRAWINGS_TO_LIMIT`.
   - ATM and POS drawing counts and amounts, minimum regularity payment ratios, overdue days.

---

## 7. Models Used

1. **Baseline Model: Logistic Regression**
   - Preprocessing Pipeline: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()` $\rightarrow$ `LogisticRegression(C=0.01, class_weight='balanced', solver='lbfgs')`.
   - Serves as an interpretable linear benchmark.
2. **Advanced Model: LightGBM Classifier (Gradient Boosted Decision Trees)**
   - Tuned hyperparameters: `n_estimators=1500`, `learning_rate=0.03`, `num_leaves=31`, `colsample_bytree=0.8`, `subsample=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`, `min_child_samples=30`, `importance_type='gain'`.
   - Early stopping (50 rounds) on validation ROC-AUC.

---

## 8. Evaluation Metric

Because default events constitute an imbalanced dataset (~8.07% positive class), accuracy is an uninformative metric. The primary evaluation metric is **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**, evaluating the model's ability to rank risk probabilities across all thresholds.

---

## 9. Experimental Results

Evaluated on an 80/20 Stratified Validation Split ($N_{\text{val}} = 61,503$, $N_{\text{train}} = 246,008$, `random_state=42`):

| Model | Validation ROC-AUC | Validation PR-AUC | Training Time | Selected |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression Baseline** | `0.77483` | `0.25586` | ~25.0s | No |
| **LightGBM Gradient Boosting** | **`0.78848`** | **`0.29139`** | ~26.8s | **Yes (Best Model)** |

### Top 10 Most Important Features (LightGBM Gain):
1. `EXT_SOURCES_MEAN` (Engineered mean of external credit scores)
2. `EXT_SOURCES_MIN` (Minimum external credit score)
3. `EXT_SOURCES_MAX` (Maximum external credit score)
4. `PAYMENT_RATE` (`AMT_ANNUITY / AMT_CREDIT`)
5. `EXT_SOURCES_SUM` (Sum of external credit scores)
6. `CREDIT_GOODS_PERC` (`AMT_CREDIT / AMT_GOODS_PRICE`)
7. `EXT_SOURCE_3` (Raw normalized external bureau score 3)
8. `AMT_ANNUITY` (Loan annuity amount)
9. `INSTAL_DPD_MEAN` (Historical average days past due on installments)
10. `BURO_DEBT_CREDIT_RATIO` (Debt-to-credit ratio across all bureau loans)

---

## 10. Project Structure

```
home-credit-default-risk/
│
├── data/
│   ├── raw/                           # Raw competition CSV files
│   └── processed/                     # Caching engineered parquet files
│
├── notebooks/
│   └── EDA.ipynb                      # Interactive EDA notebook with plots
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # Memory reduction, logging, timers
│   ├── data_preprocessing.py          # Application data cleaning & encoding
│   ├── feature_engineering.py         # Relational table aggregations & ratios
│   ├── train.py                       # Baseline & LightGBM training & selection
│   ├── evaluate.py                    # ROC-AUC, curves, confusion matrix, FI
│   ├── predict.py                     # Inference & submission validation
│   └── eda_analysis.py                # Standalone EDA figure generator
│
├── models/
│   └── final_model.pkl                # Serialized final retrained LightGBM model
│
├── reports/
│   ├── figures/                       # Generated analysis and model plots
│   │   ├── target_distribution.png
│   │   ├── missing_values.png
│   │   ├── ext_sources_kde.png
│   │   ├── age_vs_default.png
│   │   ├── correlation_matrix.png
│   │   ├── days_employed_anomaly.png
│   │   ├── roc_curve.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   ├── feature_importance.csv         # Full ranked feature importances
│   └── pipeline.log                   # Pipeline execution logs
│
├── run_pipeline.py                    # Master pipeline runner
├── submission.csv                     # Verified Kaggle submission file
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Git tracking rules
└── README.md                          # Project documentation
```

---

## 11. How to Run

### 1. Environment Setup
```bash
# Clone or navigate to the repository
cd /path/to/home-credit-default-risk

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute End-to-End Pipeline
Run the master script to reproduce EDA, preprocessing, feature engineering, model training, validation, full retraining, and submission generation:
```bash
python3 run_pipeline.py
```

### 3. Run Individual Components
- **Generate EDA Figures**:
  ```bash
  python3 -m src.eda_analysis
  ```
- **Preprocess Application Data**:
  ```bash
  python3 -m src.data_preprocessing
  ```

---

## 12. Submission File Verification

The generated `submission.csv` was strictly validated against Kaggle competition specifications:
- **File Name**: `submission.csv`
- **Dimensions**: Exactly `48,744` rows $\times$ `2` columns (`SK_ID_CURR`, `TARGET`).
- **Identifier Check**: `SK_ID_CURR` aligns 1-to-1 with `application_test.csv` and `sample_submission.csv`.
- **Target Values**: Continuous calibrated probabilities in $[0.00336, 0.75209]$ (Mean = $0.07501$, matching the true default base rate).
- **Integrity**: Zero NaN values, zero duplicates.

Preview:
```csv
SK_ID_CURR,TARGET
100001,0.040553
100005,0.154904
100013,0.044965
100028,0.033204
100038,0.165653
```

---

## 13. Future Improvements

1. **5-Fold Stratified Out-of-Fold Ensembling**: Blend LightGBM, CatBoost, and XGBoost models to further improve test ROC-AUC.
2. **Target Encoding**: Out-of-fold target encoding for high-cardinality features such as `ORGANIZATION_TYPE` and `OCCUPATION_TYPE`.
3. **Time-decayed Aggregations**: Apply exponentially weighted moving averages on recent installment payments and bureau records.
