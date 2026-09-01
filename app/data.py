from pathlib import Path

import pandas as pd

from src.database import get_connection


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DATA_PATH = BASE_DIR / "data/processed/credit_risk_features.csv"
RISK_DATA_PATH = BASE_DIR / "data/processed/customer_risk_output_full.csv"
SEGMENT_DATA_PATH = BASE_DIR / "data/processed/customer_segmentation_output.csv"
MONITORING_DATA_PATH = BASE_DIR / "data/processed/model_monitoring_output.csv"


def load_from_postgres():
    """Load the dashboard dataset from PostgreSQL."""
    conn = get_connection()

    try:
        query = """
            SELECT
                c.customer_id,
                c.credit_limit,
                c.age,
                c.gender,
                c.education,
                c.marital_status,
                c.num_delayed_months,
                c.max_delay,
                c.avg_delay,
                c.recent_delay,
                c.utilization_sep,
                c.avg_utilization,
                c.max_utilization,
                c.total_payment_6m,
                c.recent_payment_3m,
                c.older_payment_3m,
                s.customer_segment,
                r.pd,
                r.risk_tier,
                r.actual_default,
                r.model_name,
                r.prediction_date
            FROM customers c
            LEFT JOIN customer_segments s
                ON c.customer_id = s.customer_id
            LEFT JOIN risk_predictions r
                ON c.customer_id = r.customer_id
            ORDER BY c.customer_id
        """

        # Read through the cursor directly.
        # This avoids pandas' warning about raw DBAPI connections.
        with conn.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    finally:
        conn.close()


def load_from_csv():
    """Load the dashboard dataset from processed CSV files."""
    features = pd.read_csv(FEATURE_DATA_PATH)
    risk = pd.read_csv(RISK_DATA_PATH)
    segments = pd.read_csv(SEGMENT_DATA_PATH)

    features = features.copy()
    features["customer_id"] = range(1, len(features) + 1)

    risk = risk.copy()

    segments = segments.copy()
    segments["customer_id"] = segments["customer_index"] + 1

    risk_columns = [
        "customer_id",
        "pd",
        "risk_tier",
        "actual_default",
    ]

    segment_columns = [
        "customer_id",
        "customer_segment",
    ]

    dashboard_df = features.merge(
        segments[segment_columns],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    dashboard_df = dashboard_df.merge(
        risk[risk_columns],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    return dashboard_df


def load_dashboard_data():
    """
    Load dashboard data using PostgreSQL first.

    If PostgreSQL is unavailable, fall back to processed CSV files.
    """
    try:
        df = load_from_postgres()

        if df.empty:
            raise ValueError("PostgreSQL returned no customer records.")

        return df, "PostgreSQL"

    except Exception as exc:
        print(
            "PostgreSQL unavailable; using CSV fallback: "
            f"{exc}"
        )

        df = load_from_csv()
        return df, "CSV fallback"


def load_monitoring_data():
    """Load model monitoring output."""
    if not MONITORING_DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(MONITORING_DATA_PATH)


def load_feature_data():
    """Load the original feature dataset."""
    return pd.read_csv(FEATURE_DATA_PATH)