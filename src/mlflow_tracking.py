import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


MODEL_PATH = "models/credit_risk_xgboost_pipeline.joblib"
RISK_CONFIG_PATH = "models/risk_config.json"
RISK_DATA_PATH = "data/processed/customer_risk_output.csv"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "credit-risk-model"


def load_model_artifacts():
    """Load the existing trained model and risk configuration."""
    model = joblib.load(MODEL_PATH)

    with open(RISK_CONFIG_PATH, "r") as f:
        risk_config = json.load(f)

    return model, risk_config


def load_prediction_data():
    """Load existing model prediction results."""
    return pd.read_csv(RISK_DATA_PATH)


def calculate_metrics(df):
    """Calculate key credit-risk model monitoring metrics."""
    average_pd = df["pd"].mean()
    observed_default_rate = df["actual_default"].mean()

    high_risk = df[df["risk_tier"] == "High"]
    low_risk = df[df["risk_tier"] == "Low"]
    medium_risk = df[df["risk_tier"] == "Medium"]

    metrics = {
        "average_pd": average_pd,
        "observed_default_rate": observed_default_rate,
        "overall_calibration_gap": average_pd - observed_default_rate,
        "median_pd": df["pd"].median(),
        "minimum_pd": df["pd"].min(),
        "maximum_pd": df["pd"].max(),
        "high_risk_default_rate": high_risk["actual_default"].mean(),
        "medium_risk_default_rate": medium_risk["actual_default"].mean(),
        "low_risk_default_rate": low_risk["actual_default"].mean(),
        "high_risk_customers": len(high_risk),
        "medium_risk_customers": len(medium_risk),
        "low_risk_customers": len(low_risk),
        "total_customers": len(df),
    }

    return metrics


def main():
    """Log the existing credit-risk model to MLflow."""
    print("Starting MLflow model tracking...")

    model, risk_config = load_model_artifacts()
    prediction_df = load_prediction_data()

    metrics = calculate_metrics(prediction_df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost_credit_risk_baseline"):

        # Model parameters
        mlflow.log_param("model_name", risk_config["model_name"])
        mlflow.log_param(
            "classification_threshold",
            risk_config["classification_threshold"],
        )
        mlflow.log_param(
            "random_state",
            risk_config["random_state"],
        )
        mlflow.log_param(
            "target",
            risk_config["target"],
        )

        # XGBoost parameters
        xgb_model = model.named_steps["model"]

        mlflow.log_param("n_estimators", xgb_model.get_params()["n_estimators"])
        mlflow.log_param("learning_rate", xgb_model.get_params()["learning_rate"])
        mlflow.log_param("max_depth", xgb_model.get_params()["max_depth"])

        # Dataset information
        mlflow.log_param("evaluation_customers", len(prediction_df))

        # Model metrics
        mlflow.log_metrics(metrics)

        # Model artifacts
        mlflow.log_artifact(RISK_CONFIG_PATH, artifact_path="configuration")

        # Log the complete preprocessing + model pipeline
        mlflow.sklearn.log_model(
            model,
            name="credit_risk_xgboost_pipeline",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        print("MLflow run completed successfully.")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print(f"Customers evaluated: {len(prediction_df):,}")
        print(f"Average PD: {metrics['average_pd']:.4f}")
        print(
            "Observed default rate: "
            f"{metrics['observed_default_rate']:.4f}"
        )
        print(
            "Calibration gap: "
            f"{metrics['overall_calibration_gap']:.4f}"
        )


if __name__ == "__main__":
    main()