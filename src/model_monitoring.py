import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


RISK_DATA_PATH = "data/processed/customer_risk_output.csv"
SEGMENT_DATA_PATH = "data/processed/customer_segmentation_output.csv"
OUTPUT_PATH = "data/processed/model_monitoring_output.csv"

CLASSIFICATION_THRESHOLD = 0.30


def load_monitoring_data():
    """Load model predictions and customer segmentation results."""
    risk_df = pd.read_csv(RISK_DATA_PATH)
    segment_df = pd.read_csv(SEGMENT_DATA_PATH)

    monitoring_df = risk_df.merge(
        segment_df,
        on="customer_index",
        how="left",
        validate="one_to_one",
    )

    if monitoring_df["customer_segment"].isna().any():
        raise ValueError("Some prediction records have no customer segment.")

    return monitoring_df


def calculate_overall_metrics(df):
    """Calculate overall model monitoring metrics."""
    average_pd = df["pd"].mean()
    observed_default_rate = df["actual_default"].mean()

    return {
        "metric_group": "overall",
        "metric": "calibration",
        "group": "all_customers",
        "customers": len(df),
        "customer_share": None,
        "average_pd": average_pd,
        "observed_default_rate": observed_default_rate,
        "calibration_gap": average_pd - observed_default_rate,
        "status": None,
    }


def calculate_risk_tier_metrics(df):
    """Calculate monitoring metrics by risk tier."""
    results = []

    for risk_tier, group in df.groupby("risk_tier", observed=True):
        average_pd = group["pd"].mean()
        observed_default_rate = group["actual_default"].mean()

        results.append(
            {
                "metric_group": "risk_tier",
                "metric": "risk_tier_performance",
                "group": risk_tier,
                "customers": len(group),
                "customer_share": len(group) / len(df),
                "average_pd": average_pd,
                "observed_default_rate": observed_default_rate,
                "calibration_gap": average_pd - observed_default_rate,
                "status": None,
            }
        )

    return results


def calculate_segment_metrics(df):
    """Calculate monitoring metrics by customer segment."""
    results = []

    for segment, group in df.groupby("customer_segment", observed=True):
        average_pd = group["pd"].mean()
        observed_default_rate = group["actual_default"].mean()

        results.append(
            {
                "metric_group": "customer_segment",
                "metric": "segment_performance",
                "group": str(segment),
                "customers": len(group),
                "customer_share": len(group) / len(df),
                "average_pd": average_pd,
                "observed_default_rate": observed_default_rate,
                "calibration_gap": average_pd - observed_default_rate,
                "status": None,
            }
        )

    return results


def calculate_prediction_distribution(df):
    """Calculate overall prediction-distribution statistics."""
    average_pd = df["pd"].mean()
    observed_default_rate = df["actual_default"].mean()

    return [
        {
            "metric_group": "prediction_distribution",
            "metric": "average_pd",
            "group": "all_customers",
            "customers": len(df),
            "customer_share": None,
            "average_pd": average_pd,
            "observed_default_rate": observed_default_rate,
            "calibration_gap": average_pd - observed_default_rate,
            "status": None,
        },
        {
            "metric_group": "prediction_distribution",
            "metric": "median_pd",
            "group": "all_customers",
            "customers": len(df),
            "customer_share": None,
            "average_pd": df["pd"].median(),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "prediction_distribution",
            "metric": "minimum_pd",
            "group": "all_customers",
            "customers": len(df),
            "customer_share": None,
            "average_pd": df["pd"].min(),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "prediction_distribution",
            "metric": "maximum_pd",
            "group": "all_customers",
            "customers": len(df),
            "customer_share": None,
            "average_pd": df["pd"].max(),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
    ]


def calculate_classification_metrics(df):
    """Calculate classification metrics at the configured threshold."""
    y_true = df["actual_default"]
    y_pred = (df["pd"] >= CLASSIFICATION_THRESHOLD).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return [
        {
            "metric_group": "classification",
            "metric": "accuracy",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": accuracy_score(y_true, y_pred),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "precision",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "recall",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "f1",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "true_negatives",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": tn,
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "false_positives",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": fp,
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "false_negatives",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": fn,
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
        {
            "metric_group": "classification",
            "metric": "true_positives",
            "group": "threshold_0.30",
            "customers": len(df),
            "customer_share": None,
            "average_pd": tp,
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        },
    ]


def calculate_discrimination_metrics(df):
    """Calculate ranking/discrimination metrics."""
    y_true = df["actual_default"]
    pd_scores = df["pd"]

    return [
        {
            "metric_group": "discrimination",
            "metric": "roc_auc",
            "group": "all_customers",
            "customers": len(df),
            "customer_share": None,
            "average_pd": roc_auc_score(
                y_true,
                pd_scores,
            ),
            "observed_default_rate": None,
            "calibration_gap": None,
            "status": None,
        }
    ]


def calculate_monitoring_status(df):
    """Evaluate basic model monitoring health checks."""
    average_pd = df["pd"].mean()
    observed_default_rate = df["actual_default"].mean()

    calibration_gap = abs(
        average_pd - observed_default_rate
    )

    risk_order = (
        df.groupby("risk_tier", observed=True)["actual_default"]
        .mean()
        .sort_index()
    )

    risk_order_ok = (
        "Low" in risk_order
        and "Medium" in risk_order
        and "High" in risk_order
        and risk_order["Low"]
        < risk_order["Medium"]
        < risk_order["High"]
    )

    calibration_ok = calibration_gap <= 0.05

    overall_status = (
        "HEALTHY"
        if calibration_ok and risk_order_ok
        else "REVIEW"
    )

    return {
        "metric_group": "monitoring_status",
        "metric": "overall_health",
        "group": "model",
        "customers": len(df),
        "customer_share": None,
        "average_pd": average_pd,
        "observed_default_rate": observed_default_rate,
        "calibration_gap": average_pd - observed_default_rate,
        "status": overall_status,
    }


def main():
    """Run model monitoring and save the monitoring results."""
    df = load_monitoring_data()

    results = [
        calculate_overall_metrics(df),
        *calculate_risk_tier_metrics(df),
        *calculate_segment_metrics(df),
        *calculate_prediction_distribution(df),
        *calculate_classification_metrics(df),
        *calculate_discrimination_metrics(df),
        calculate_monitoring_status(df),
    ]

    monitoring_output = pd.DataFrame(results)

    monitoring_output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Model monitoring completed successfully.")
    print(f"Customers monitored: {len(df):,}")
    print(
        f"Monitoring rows generated: "
        f"{len(monitoring_output):,}"
    )
    print(f"Output saved to: {OUTPUT_PATH}")

    print("\nOverall calibration:")
    overall = monitoring_output[
        monitoring_output["metric_group"] == "overall"
    ]
    print(overall.to_string(index=False))

    print("\nRisk-tier monitoring:")
    risk_tier = monitoring_output[
        monitoring_output["metric_group"] == "risk_tier"
    ]
    print(risk_tier.to_string(index=False))

    print("\nSegment monitoring:")
    segments = monitoring_output[
        monitoring_output["metric_group"] == "customer_segment"
    ]
    print(segments.to_string(index=False))

    print("\nClassification metrics:")
    classification = monitoring_output[
        monitoring_output["metric_group"] == "classification"
    ]
    print(classification.to_string(index=False))

    print("\nDiscrimination metrics:")
    discrimination = monitoring_output[
        monitoring_output["metric_group"] == "discrimination"
    ]
    print(discrimination.to_string(index=False))

    print("\nMonitoring status:")
    status = monitoring_output[
        monitoring_output["metric_group"] == "monitoring_status"
    ]
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()