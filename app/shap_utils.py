from pathlib import Path

import joblib
import pandas as pd
import shap


MODEL_PATH = Path("models/credit_risk_xgboost_pipeline.joblib")
FEATURE_DATA_PATH = Path("data/processed/credit_risk_features.csv")


MODEL_FEATURES = [
    "credit_limit",
    "gender",
    "education",
    "marital_status",
    "age",
    "repayment_sep",
    "repayment_aug",
    "repayment_july",
    "repayment_jun",
    "repayment_may",
    "repayment_apr",
    "bill_sep",
    "bill_aug",
    "bill_july",
    "bill_jun",
    "bill_may",
    "bill_apr",
    "payment_sep",
    "payment_aug",
    "payment_july",
    "payment_jun",
    "payment_may",
    "payment_apr",
    "num_delayed_months",
    "max_delay",
    "avg_delay",
    "recent_delay",
    "utilization_sep",
    "avg_utilization",
    "max_utilization",
    "total_payment_6m",
    "recent_payment_3m",
    "older_payment_3m",
]


def load_model():
    """Load the trained credit-risk pipeline."""
    return joblib.load(MODEL_PATH)


def load_feature_data():
    """Load the original feature dataset used by the model."""
    df = pd.read_csv(FEATURE_DATA_PATH)

    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing from feature dataset: "
            f"{missing_features}"
        )

    return df


def get_customer_model_features(customer_id):
    """
    Retrieve the original model input features for one customer.

    Customer IDs in PostgreSQL start at 1, while the CSV uses
    zero-based row positions.
    """
    feature_df = load_feature_data()

    if customer_id < 1 or customer_id > len(feature_df):
        raise ValueError(
            f"Customer ID {customer_id} is outside the valid range "
            f"1-{len(feature_df)}."
        )

    customer_row = feature_df.iloc[[customer_id - 1]]

    return customer_row[MODEL_FEATURES]


def get_readable_feature_name(feature_name):
    """Convert transformed sklearn feature names into readable names."""
    name = feature_name

    if name.startswith("num__"):
        name = name.replace("num__", "", 1)

    elif name.startswith("cat__"):
        name = name.replace("cat__", "", 1)

        categorical_prefixes = [
            "gender_",
            "education_",
            "marital_status_",
        ]

        for prefix in categorical_prefixes:
            if name.startswith(prefix):
                category = name.replace(prefix, "", 1)
                return f"{prefix[:-1].replace('_', ' ').title()} = {category}"

    return name.replace("_", " ").title()


def calculate_customer_shap(model, customer_row):
    """
    Calculate SHAP contributions for one customer.

    Parameters
    ----------
    model:
        Trained sklearn Pipeline containing the preprocessor and XGBoost model.

    customer_row:
        DataFrame containing the original 33 model features for one customer.

    Returns
    -------
    DataFrame
        SHAP explanation sorted by absolute contribution.
    """
    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in customer_row.columns
    ]

    if missing_features:
        raise ValueError(
            "Required model features are missing: "
            f"{missing_features}"
        )

    customer_features = customer_row[MODEL_FEATURES]

    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    transformed_features = preprocessor.transform(customer_features)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(xgb_model)

    shap_values = explainer.shap_values(transformed_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = shap_values[0]

    explanation = pd.DataFrame(
        {
            "feature": [
                get_readable_feature_name(name)
                for name in feature_names
            ],
            "raw_feature": feature_names,
            "feature_value": transformed_features[0],
            "shap_value": shap_values,
        }
    )

    explanation["abs_shap"] = explanation["shap_value"].abs()

    explanation["direction"] = explanation["shap_value"].apply(
        lambda value: "Increases risk"
        if value > 0
        else "Decreases risk"
    )

    explanation = explanation.sort_values(
        "abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    return explanation


def calculate_customer_shap_by_id(customer_id):
    """
    Calculate SHAP explanations directly from a customer ID.
    """
    model = load_model()

    customer_features = get_customer_model_features(customer_id)

    return calculate_customer_shap(
        model,
        customer_features,
    )