# Financial Risk Intelligence

## Credit Card Default Risk Prediction & Customer Risk Intelligence

A machine-learning based credit-risk decision-support system for predicting credit-card default risk, assigning Probability of Default (PD), identifying behavioral customer segments, explaining individual predictions with SHAP, and monitoring model performance.

The project uses the **UCI Default of Credit Card Clients** dataset and combines supervised machine learning, feature engineering, customer segmentation, model explainability, PostgreSQL, and a Streamlit dashboard into an end-to-end credit-risk analytics workflow.

---

## Project Overview

Credit-card default prediction is a binary classification problem where the objective is to estimate whether a customer is likely to default on their payment obligations.

This project goes beyond a simple default prediction by producing:

* Probability of Default (PD) for each customer
* Low, Medium, and High risk tiers
* Behavioral customer segments using KMeans
* Individual customer risk explanations using SHAP
* Portfolio-level risk analysis
* Model performance and calibration monitoring
* PostgreSQL storage
* A Dockerized Streamlit decision-support dashboard

The final portfolio contains **29,965 customers** after removing 35 exact duplicate records from the original 30,000 observations.

---

## Key Results

The final XGBoost model was evaluated on a held-out test population of **5,993 customers**.

| Metric                   | Result |
| ------------------------ | -----: |
| ROC-AUC                  | 0.7730 |
| PR-AUC                   | 0.5510 |
| Accuracy                 | 0.7909 |
| Precision                | 0.5271 |
| Recall                   | 0.5354 |
| F1 Score                 | 0.5312 |
| Actual Defaults          |  1,326 |
| Predicted Defaults       |  1,347 |
| Classification Threshold |   0.30 |

The model was selected over the Logistic Regression baseline because it provided stronger discrimination and precision on the validation data.

---

## Dashboard

The project includes a four-page Streamlit dashboard:

### 1. Portfolio Overview

Provides a portfolio-level view of:

* Customer count
* Average Probability of Default
* Observed default rate
* High-risk customer share
* Risk-tier distribution
* Behavioral customer-segment distribution
* Risk-tier performance
* Portfolio filtering by risk tier and segment

### 2. Customer Lookup

Allows individual customers to be inspected using:

* Probability of Default
* Risk tier
* Behavioral segment
* Credit limit
* Age and demographic information
* Repayment behavior
* Delay statistics
* Credit utilization
* Payment behavior
* Individual SHAP explanation

The dashboard displays the features that contribute most strongly to an individual customer's model prediction.

### 3. Model Monitoring

Provides model-level monitoring using the held-out test population:

* Average predicted PD
* Observed default rate
* Calibration gap
* ROC-AUC
* Accuracy
* Precision
* Recall
* F1 score
* Risk-tier performance
* Customer-segment performance
* Monitoring health status

### 4. Model Information

Displays:

* Model name and version
* Classification threshold
* XGBoost configuration
* Preprocessing information
* Number of transformed features
* Performance snapshot

---

## Data Source

The project uses the **UCI Default of Credit Card Clients** dataset.

The original dataset contains:

* 30,000 customer records
* 23 explanatory variables
* 1 binary target variable indicating default

The raw dataset is obtained from the UCI Machine Learning Repository.

The project removes **35 exact duplicate records**, resulting in a cleaned dataset containing **29,965 customers**.

---

# Project Workflow

The project follows the following workflow:

```text
UCI Dataset
     │
     ▼
Data Understanding & Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Train / Validation / Test Split
     │
     ▼
Logistic Regression Baseline
     │
     ▼
XGBoost Model
     │
     ▼
Probability of Default
     │
     ├──────────────► Risk Tiers
     │
     ├──────────────► SHAP Explainability
     │
     └──────────────► Model Monitoring

Feature & Behavior Data
     │
     ▼
KMeans Customer Segmentation

All Results
     │
     ▼
PostgreSQL Database
     │
     ▼
Streamlit Dashboard
```

---

# 1. Data Understanding

The first stage focused on understanding the raw dataset and preparing it for downstream analysis.

The work included:

* Loading and inspecting the UCI Default of Credit Card Clients dataset.
* Examining dataset structure, data types, missing values, and target distribution.
* Reviewing categorical variables, repayment-status variables, bill amounts, and payment amounts.
* Identifying and removing 35 exact duplicate records.
* Documenting unusual bill values and the right-skewed distribution of payment amounts.
* Saving the cleaned dataset containing 29,965 customers for downstream feature engineering.

### Categorical Variables

The categorical variables are stored as numeric codes rather than continuous measurements.

#### Gender

| Code | Meaning |
| ---: | ------- |
|    1 | Male    |
|    2 | Female  |

#### Education

| Code | Meaning         |
| ---: | --------------- |
|    0 | Others          |
|    1 | Graduate School |
|    2 | University      |
|    3 | High School     |
|    4 | Others          |
|    5 | Unknown         |
|    6 | Unknown         |

#### Marital Status

| Code | Meaning |
| ---: | ------- |
|    0 | Others  |
|    1 | Married |
|    2 | Single  |
|    3 | Others  |

The original categorical codes are retained in the modeling data and handled as categorical variables during preprocessing.

---

# 2. Feature Engineering

Ten engineered features were created to summarize repayment behavior, credit utilization, and payment behavior.

### Repayment Behavior

* `num_delayed_months`
* `max_delay`
* `avg_delay`
* `recent_delay`

### Credit Utilization

* `utilization_sep`
* `avg_utilization`
* `max_utilization`

### Payment Behavior

* `total_payment_6m`
* `recent_payment_3m`
* `older_payment_3m`

These features summarize information across the six monthly repayment, billing, and payment observations.

Candidate features such as average bill utilization and payment trend ratio were evaluated and removed from the final feature set when they were considered redundant or less useful.

For utilization calculations, negative statement amounts were clipped to zero for the derived utilization features. The original bill variables were not modified.

The final feature-engineered dataset contains:

* 29,965 customers
* 33 model input features
* 1 target variable

---

# 3. Model Development

Two supervised learning models were evaluated.

### Logistic Regression

Logistic Regression was used as an interpretable baseline model.

### XGBoost

XGBoost was selected as the primary model because it provided stronger performance on the validation data, particularly in ROC-AUC, PR-AUC, and precision.

The final model is an XGBoost classifier wrapped inside a preprocessing pipeline.

### Data Split

A stratified three-way split was used:

* 60% training
* 20% validation
* 20% test

The validation set was used for model and classification-threshold decisions. The test set was kept separate for final evaluation.

### Preprocessing

Categorical variables were one-hot encoded.

For Logistic Regression, numerical features were standardized.

For XGBoost, numerical features were passed through without standardization.

### Classification Threshold

A classification threshold of **0.30** was selected using the validation set.

This means:

```text
Predicted default = 1 when PD >= 0.30
Predicted default = 0 when PD < 0.30
```

The classification threshold should not be confused with the risk-tier boundaries described below.

---

# 4. Probability of Default & Risk Tiers

The final XGBoost model produces a continuous **Probability of Default (PD)** for each customer.

PD represents the model-estimated likelihood of default and is different from the binary classification decision.

### Project Risk Tiers

| Risk Tier | PD Range       |
| --------- | -------------- |
| Low       | PD < 10%       |
| Medium    | 10% ≤ PD < 30% |
| High      | PD ≥ 30%       |

These risk-tier boundaries are **project-defined thresholds** used for portfolio analysis. They are not presented as universal banking or regulatory thresholds.

### Test Population Risk-Tier Results

| Risk Tier | Customers | Average PD | Observed Default Rate |
| --------- | --------: | ---------: | --------------------: |
| Low       |     2,149 |      6.01% |                 7.63% |
| Medium    |     2,497 |     17.35% |                18.10% |
| High      |     1,347 |     55.33% |                52.71% |

The observed default rate increases consistently from Low to Medium to High risk.

---

# 5. Model Explainability

**SHAP (SHapley Additive exPlanations)** was used to explain both the global behavior of the XGBoost model and individual customer predictions.

The explainability workflow:

* Uses SHAP TreeExplainer with the trained XGBoost estimator.
* Applies the preprocessing pipeline before generating model explanations.
* Uses mean absolute SHAP values for global feature importance.
* Examines SHAP direction to understand whether a feature pushes model output higher or lower.
* Groups one-hot encoded categorical contributions back to their original feature names for human-readable explanations.
* Provides individual customer-level SHAP explanations through the Streamlit dashboard.

Repayment behavior is a major source of model influence, with features such as delay statistics, recent repayment status, number of delayed months, and maximum delay among important predictors.

SHAP explanations describe how the trained model arrives at its predictions. They should **not be interpreted as causal effects**.

---

# 6. Customer Segmentation

KMeans clustering was used to identify behavioral and financial customer segments independently of the default target and model PD.

The clustering process:

* Excluded `default` and model PD to avoid target leakage.
* Focused on financial and behavioral characteristics.
* Used repayment-delay, utilization, credit-limit, age, and payment-behavior features.
* Applied `log1p` transformation to payment aggregates because of their right-skewed distributions.
* Standardized clustering features using `StandardScaler`.
* Evaluated K values from 2 through 8 using inertia and silhouette score.

### Selected Number of Clusters

**K = 3** was selected because it provided the best balance between clustering quality and interpretability.

The resulting segments are behavioral groups rather than direct replacements for the Low, Medium, and High PD risk tiers.

### Test Population Segments

| Segment | Behavioral Profile             | Observed Default Rate | Average PD |
| ------: | ------------------------------ | --------------------: | ---------: |
|       1 | Lower-risk behavioral group    |                11.90% |     10.84% |
|       0 | Elevated-risk behavioral group |                33.80% |     34.98% |
|       2 | Highest-risk behavioral group  |                44.03% |     45.16% |

The lowest and highest observed default rates differ by approximately **32.13 percentage points**.

KMeans segment labels are arbitrary identifiers. Their interpretation comes from profiling the characteristics and risk behavior of each group.

---

# 7. Model Monitoring

Model monitoring was performed using the held-out test population of **5,993 customers**.

The monitoring process evaluates:

* Average predicted PD versus observed default rate
* Calibration gap
* Risk-tier performance
* Customer-segment performance
* Classification metrics at the 0.30 decision threshold
* ROC-AUC

### Overall Monitoring Results

| Metric                |   Result |
| --------------------- | -------: |
| Average PD            |   21.82% |
| Observed Default Rate |   22.13% |
| Calibration Gap       | -0.31 pp |
| ROC-AUC               |   0.7730 |
| Accuracy              |   0.7909 |
| Precision             |   0.5271 |
| Recall                |   0.5354 |
| F1                    |   0.5312 |

The calibration gap is calculated as:

```text
Average PD - Observed Default Rate
```

The resulting gap is approximately **-0.31 percentage points**.

### Monitoring Health Check

The project marks monitoring status as `HEALTHY` when:

* The absolute calibration gap is within 5 percentage points.
* Observed default rates increase consistently from Low to Medium to High risk.

The current evaluated test population has a monitoring status of:

**HEALTHY**

This monitoring implementation is a **project-level model assessment**. It does not implement production-grade feature drift detection, PSI/CSI monitoring, automated alerts, continuous monitoring, or regulatory model validation.

---

# 8. PostgreSQL Database

PostgreSQL is used to store the customer portfolio, behavioral segmentation results, model predictions, and model metadata.

The database contains four primary tables:

### `customers`

Stores customer-level demographic, credit, repayment, utilization, and payment information.

### `customer_segments`

Stores the KMeans behavioral segment assigned to each customer.

### `risk_predictions`

Stores:

* Probability of Default
* Risk tier
* Actual default
* Model name
* Prediction date

### `model_metadata`

Stores model information including:

* Model name
* Model version
* Classification threshold
* Model description
* Creation timestamp

Foreign-key relationships connect customer segments and predictions to the corresponding customer records.

---

# 9. Streamlit Application

The application is located in:

```text
app/
```

The main entry point is:

```text
app/app.py
```

The application supports PostgreSQL as its primary data source and CSV fallback for portfolio data when PostgreSQL is unavailable.

The dashboard also loads the trained XGBoost pipeline for model information and individual SHAP explanations.

---

# 10. Docker Deployment

The project includes a Dockerized deployment with two services:

```text
Streamlit
    │
    ▼
PostgreSQL
```

### Services

#### Streamlit

* Python 3.12
* Streamlit application
* Port `8501`

#### PostgreSQL

* PostgreSQL 16
* Database: `credit_risk_db`
* Persistent Docker volume
* Healthcheck before Streamlit starts

### Start the Application

After the required generated artifacts are available:

```bash
docker compose up --build
```

The dashboard is then available at:

```text
http://localhost:8501
```

To stop the services:

```bash
docker compose down
```

To stop the services while retaining the PostgreSQL volume:

```bash
docker compose down
```

The PostgreSQL data is stored in the Docker volume:

```text
postgres_data
```

---

# 11. Reproducibility

The repository contains the project source code, notebooks, application code, Docker configuration, database schema, and model configuration.

Generated datasets, processed outputs, and the trained model artifact are excluded from Git.

Important generated artifacts include:

```text
data/raw/
data/processed/
models/credit_risk_xgboost_pipeline.joblib
```

The trained XGBoost pipeline and risk configuration are generated by the model-development notebook:

```text
notebooks/03_credit_risk_modeling.ipynb
```

The notebook saves:

```text
models/credit_risk_xgboost_pipeline.joblib
models/risk_config.json
```

The downstream notebooks use the saved model and configuration for PD/risk analysis, explainability, and segmentation.

The final dashboard/database workflow also expects the generated processed data files to be available locally.

Because generated artifacts are not committed to Git, a completely fresh clone requires the modeling/data-generation workflow to be executed or the generated runtime artifacts to be supplied before launching the full Docker application.

---

# 12. Project Structure

```text
financial-risk-intelligence/
│
├── app/
│   ├── app.py
│   ├── data.py
│   └── shap_utils.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docker/
│   └── init.sql
│
├── models/
│   ├── risk_config.json
│   └── credit_risk_xgboost_pipeline.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_credit_risk_modeling.ipynb
│   ├── 04_pd_and_risk_segmentation.ipynb
│   ├── 05_model_explainability.ipynb
│   └── 06_customer_segmentation.ipynb
│
├── src/
│   ├── database.py
│   ├── download_data.py
│   ├── load_database.py
│   ├── mlflow_tracking.py
│   └── model_monitoring.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

# 13. Technology Stack

### Programming & Analysis

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* XGBoost
* Logistic Regression
* KMeans

### Explainability

* SHAP

### Model Tracking

* MLflow

### Database

* PostgreSQL

### Dashboard

* Streamlit
* Plotly

### Deployment

* Docker
* Docker Compose

---

# 14. Key Design Decisions

### Why XGBoost?

XGBoost was selected after comparison with Logistic Regression because it provided stronger predictive performance for the project.

### Why a 0.30 classification threshold?

The threshold was selected using the validation set rather than the final test set. This allows the test set to remain an unbiased evaluation population.

### Why separate PD and risk tiers?

PD is a continuous model output, while risk tiers provide an easier portfolio-level categorization.

The binary classification threshold of **0.30** is therefore distinct from the risk-tier boundaries of **10% and 30% PD**.

### Why KMeans?

KMeans provides behavioral customer profiles that complement the supervised default-risk model.

The segmentation does not use default or PD during clustering, reducing target leakage and allowing customer behavior to be analyzed separately from model-predicted risk.

### Why SHAP?

SHAP provides a model-specific explanation of which features contribute to individual predictions and overall model behavior.

---

# 15. Limitations

This project is intended as an academic and portfolio-level credit-risk intelligence system.

Important limitations include:

* The dataset is historical and may not represent current credit-card populations.
* Model performance may change on different populations or time periods.
* The project does not implement production-grade drift detection.
* The monitoring process is not continuous.
* The risk-tier boundaries are project-defined rather than regulatory standards.
* SHAP explanations indicate model behavior rather than causality.
* The model should not be treated as an automated lending decision system without additional validation, governance, fairness analysis, and regulatory review.
* Generated data and model artifacts are not stored directly in the Git repository.

---

# 16. Conclusion

Financial Risk Intelligence combines machine learning, customer analytics, explainability, monitoring, database integration, and dashboard development into a single credit-risk decision-support workflow.

The final system can:

1. Prepare and engineer credit-risk data.
2. Predict customer default probability using XGBoost.
3. Convert PD into interpretable risk tiers.
4. Identify behavioral customer segments.
5. Explain individual predictions using SHAP.
6. Monitor model performance and risk-tier consistency.
7. Store results in PostgreSQL.
8. Present portfolio and customer-level insights through Streamlit.
9. Run the application using Docker.

The project demonstrates an end-to-end approach to building a practical financial-risk analytics system rather than focusing only on model training.

---

## Author

**Rishikesh Singh**

B.Sc. Data Science — Final Year Project

GitHub repository:

`https://github.com/RishikeshSingh19/financial-risk-intelligence`
