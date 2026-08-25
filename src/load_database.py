import pandas as pd

from src.database import get_connection


FEATURE_DATA_PATH = "data/processed/credit_risk_features.csv"
SEGMENT_DATA_PATH = "data/processed/customer_segmentation_output.csv"
RISK_DATA_PATH = "data/processed/customer_risk_output.csv"


def load_customers(conn):
    """Load customer-level features into the customers table."""
    df = pd.read_csv(FEATURE_DATA_PATH)

    customer_columns = [
        "credit_limit",
        "age",
        "gender",
        "education",
        "marital_status",
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

    records = df[customer_columns].copy()
    records.insert(0, "customer_id", range(1, len(records) + 1))

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO customers (
                customer_id,
                credit_limit,
                age,
                gender,
                education,
                marital_status,
                num_delayed_months,
                max_delay,
                avg_delay,
                recent_delay,
                utilization_sep,
                avg_utilization,
                max_utilization,
                total_payment_6m,
                recent_payment_3m,
                older_payment_3m
            )
            VALUES (
                %(customer_id)s,
                %(credit_limit)s,
                %(age)s,
                %(gender)s,
                %(education)s,
                %(marital_status)s,
                %(num_delayed_months)s,
                %(max_delay)s,
                %(avg_delay)s,
                %(recent_delay)s,
                %(utilization_sep)s,
                %(avg_utilization)s,
                %(max_utilization)s,
                %(total_payment_6m)s,
                %(recent_payment_3m)s,
                %(older_payment_3m)s
            )
            """,
            records.to_dict("records"),
        )

    print(f"Loaded customers: {len(records):,}")


def load_segments(conn):
    """Load customer segmentation results."""
    df = pd.read_csv(SEGMENT_DATA_PATH)

    records = pd.DataFrame({
        "customer_id": df["customer_index"] + 1,
        "customer_segment": df["customer_segment"],
    })

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO customer_segments (
                customer_id,
                customer_segment
            )
            VALUES (
                %(customer_id)s,
                %(customer_segment)s
            )
            """,
            records.to_dict("records"),
        )

    print(f"Loaded customer segments: {len(records):,}")


def load_risk_predictions(conn):
    """Load model PD and risk-tier predictions."""
    df = pd.read_csv(RISK_DATA_PATH)

    records = pd.DataFrame({
        "customer_id": df["customer_index"] + 1,
        "pd": df["pd"],
        "risk_tier": df["risk_tier"],
        "actual_default": df["actual_default"],
        "model_name": "XGBoost",
        "prediction_date": pd.Timestamp.today().date(),
    })

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO risk_predictions (
                customer_id,
                pd,
                risk_tier,
                actual_default,
                model_name,
                prediction_date
            )
            VALUES (
                %(customer_id)s,
                %(pd)s,
                %(risk_tier)s,
                %(actual_default)s,
                %(model_name)s,
                %(prediction_date)s
            )
            """,
            records.to_dict("records"),
        )

    print(f"Loaded risk predictions: {len(records):,}")


def load_model_metadata(conn):
    """Load metadata describing the selected credit-risk model."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_metadata (
                model_name,
                model_version,
                classification_threshold,
                description
            )
            VALUES (
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                "XGBoost",
                "1.0",
                0.5,
                "Selected XGBoost credit-risk model used for Probability of Default estimation."
            ),
        )

    print("Loaded model metadata.")


def main():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE TABLE
                    risk_predictions,
                    customer_segments,
                    customers
                CASCADE
                """
            )

        load_customers(conn)
        load_segments(conn)
        load_risk_predictions(conn)
        load_model_metadata(conn)

        conn.commit()

        print("\nDatabase load completed successfully.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
