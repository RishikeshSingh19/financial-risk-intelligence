import pandas as pd


RISK_DATA_PATH = "data/processed/customer_risk_output.csv"
SEGMENT_DATA_PATH = "data/processed/customer_segmentation_output.csv"
OUTPUT_PATH = "data/processed/model_monitoring_output.csv"


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


def calculate_monitoring_status(df):
    """Evaluate basic model monitoring health checks."""
    average_pd = df["pd"].mean()
    observed_default_rate = df["actual_default"].mean()

    calibration_gap = abs(average_pd - observed_default_rate)

    risk_order = (
        df.groupby("risk_tier", observed=True)["actual_default"]
        .mean()
        .sort_index()
    )

    risk_order_ok = (
        "Low" in risk_order
        and "Medium" in risk_order
        and "High" in risk_order
        and risk_order["Low"] < risk_order["Medium"] < risk_order["High"]
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
        calculate_monitoring_status(df),
    ]

    monitoring_output = pd.DataFrame(results)

    monitoring_output.to_csv(OUTPUT_PATH, index=False)

    print("Model monitoring completed successfully.")
    print(f"Customers monitored: {len(df):,}")
    print(f"Monitoring rows generated: {len(monitoring_output):,}")
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

    print("\nMonitoring status:")
    status = monitoring_output[
        monitoring_output["metric_group"] == "monitoring_status"
    ]
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()