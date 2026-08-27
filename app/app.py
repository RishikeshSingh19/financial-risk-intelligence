from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from data import load_dashboard_data, load_monitoring_data
from shap_utils import calculate_customer_shap_by_id, load_model


st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
)


# ---------------------------------------------------------------------
# Cached data/model loaders
# ---------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_dashboard_data():
    """
    Load dashboard data and cache the actual load timestamp.

    The timestamp therefore represents when the cached dataset was
    actually loaded rather than changing on every Streamlit rerun.
    """
    df, source = load_dashboard_data()
    loaded_at = datetime.now()

    return df, source, loaded_at


@st.cache_data(ttl=300)
def get_monitoring_data():
    return load_monitoring_data()


@st.cache_resource
def get_model():
    return load_model()


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

RISK_ICONS = {
    "Low": "🟢",
    "Medium": "🟠",
    "High": "🔴",
}


RISK_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#ef6c00",
    "High": "#c62828",
}


GENDER_LABELS = {
    1: "Male",
    2: "Female",
}


EDUCATION_LABELS = {
    0: "Unknown",
    1: "Graduate School",
    2: "University",
    3: "High School",
    4: "Other",
    5: "Unknown",
    6: "Unknown",
}


MARITAL_STATUS_LABELS = {
    0: "Unknown",
    1: "Married",
    2: "Single",
    3: "Other",
}


def risk_color(risk_tier):
    """Return a display icon for a risk tier."""
    return RISK_ICONS.get(str(risk_tier), "⚪")


def format_percentage(value):
    """Format a decimal probability as a percentage."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def format_gender(value):
    """Convert gender code to readable label."""
    try:
        return GENDER_LABELS.get(int(value), f"Code {int(value)}")
    except (TypeError, ValueError):
        return "Unknown"


def format_education(value):
    """Convert education code to readable label."""
    try:
        return EDUCATION_LABELS.get(int(value), f"Code {int(value)}")
    except (TypeError, ValueError):
        return "Unknown"


def format_marital_status(value):
    """Convert marital-status code to readable label."""
    try:
        return MARITAL_STATUS_LABELS.get(
            int(value),
            f"Code {int(value)}",
        )
    except (TypeError, ValueError):
        return "Unknown"


def risk_bar_html(pd_value, risk_tier):
    """Create a risk-tier-aware probability bar."""
    pd_value = min(max(float(pd_value), 0.0), 1.0)
    percentage = pd_value * 100

    color = RISK_COLORS.get(
        str(risk_tier),
        "#607d8b",
    )

    return f"""
    <div style="
        margin-top: 10px;
        margin-bottom: 10px;
    ">
        <div style="
            background-color: #e0e0e0;
            border-radius: 8px;
            height: 18px;
            width: 100%;
            overflow: hidden;
        ">
            <div style="
                background-color: {color};
                width: {percentage:.2f}%;
                height: 100%;
                border-radius: 8px;
            "></div>
        </div>
        <div style="
            display: flex;
            justify-content: space-between;
            margin-top: 6px;
            font-size: 0.85rem;
            color: #666;
        ">
            <span>0%</span>
            <span>{percentage:.2f}% PD</span>
            <span>100%</span>
        </div>
    </div>
    """


def apply_portfolio_filters(df):
    """Render sidebar filters and return the filtered portfolio."""
    filtered_df = df.copy()

    st.sidebar.subheader("Portfolio Filters")

    risk_options = ["All", "Low", "Medium", "High"]

    selected_risk = st.sidebar.selectbox(
        "Risk Tier",
        risk_options,
    )

    if selected_risk != "All":
        filtered_df = filtered_df[
            filtered_df["risk_tier"] == selected_risk
        ]

    segment_values = (
        filtered_df["customer_segment"]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    segment_options = ["All"] + [
        f"Segment {segment}"
        for segment in segment_values
    ]

    selected_segment = st.sidebar.selectbox(
        "Customer Segment",
        segment_options,
    )

    if selected_segment != "All":
        segment_number = int(
            selected_segment.replace("Segment ", "")
        )

        filtered_df = filtered_df[
            filtered_df["customer_segment"] == segment_number
        ]

    st.sidebar.caption(
        f"Showing {len(filtered_df):,} customers"
    )

    return filtered_df


# ---------------------------------------------------------------------
# Portfolio Overview
# ---------------------------------------------------------------------

def portfolio_overview(df):
    """Render the portfolio overview page."""

    st.title("Credit Risk Intelligence")

    st.caption(
        "Portfolio-level credit-risk decision support powered by XGBoost."
    )

    prediction_df = df.dropna(
        subset=["pd"]
    ).copy()

    if prediction_df.empty:
        st.warning("No model predictions are available.")
        return

    total_customers = len(prediction_df)
    average_pd = prediction_df["pd"].mean()
    default_rate = prediction_df["actual_default"].mean()

    high_risk_share = (
        prediction_df["risk_tier"].eq("High").mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            f"{total_customers:,}",
        )

    with col2:
        st.metric(
            "Average PD",
            format_percentage(average_pd),
        )

    with col3:
        st.metric(
            "Observed Default Rate",
            format_percentage(default_rate),
        )

    with col4:
        st.metric(
            "High-Risk Share",
            format_percentage(high_risk_share),
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Risk-Tier Distribution")

        risk_distribution = (
            prediction_df["risk_tier"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High"],
                fill_value=0,
            )
            .rename_axis("risk_tier")
            .reset_index(name="customers")
        )

        fig = px.bar(
            risk_distribution,
            x="risk_tier",
            y="customers",
            text="customers",
            color="risk_tier",
            category_orders={
                "risk_tier": ["Low", "Medium", "High"]
            },
            color_discrete_map={
                "Low": "#2E8B57",
                "Medium": "#F4A261",
                "High": "#D62728",
            },
            title="Customers by Risk Tier",
        )

        fig.update_layout(
            xaxis_title="Risk Tier",
            yaxis_title="Customers",
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:
        st.subheader("Customer Segments")

        segment_distribution = (
            prediction_df["customer_segment"]
            .value_counts()
            .sort_index()
            .rename_axis("customer_segment")
            .reset_index(name="customers")
        )

        segment_distribution["customer_segment"] = (
            "Segment "
            + segment_distribution["customer_segment"].astype(str)
        )

        fig = px.pie(
            segment_distribution,
            names="customer_segment",
            values="customers",
            title="Customer Segment Distribution",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    st.divider()

    st.subheader("Risk-Tier Performance")

    tier_summary = (
        prediction_df
        .groupby("risk_tier", observed=True)
        .agg(
            customers=("customer_id", "count"),
            average_pd=("pd", "mean"),
            observed_default_rate=("actual_default", "mean"),
        )
        .reset_index()
    )

    tier_summary["average_pd"] *= 100
    tier_summary["observed_default_rate"] *= 100

    tier_summary["calibration_gap"] = (
        tier_summary["average_pd"]
        - tier_summary["observed_default_rate"]
    )

    tier_summary = tier_summary.round(
        {
            "average_pd": 2,
            "observed_default_rate": 2,
            "calibration_gap": 2,
        }
    )

    tier_summary = tier_summary.rename(
        columns={
            "risk_tier": "Risk Tier",
            "customers": "Customers",
            "average_pd": "Average PD (%)",
            "observed_default_rate": "Observed Default Rate (%)",
            "calibration_gap": "Calibration Gap (pp)",
        }
    )

    st.dataframe(
        tier_summary,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Risk-Tier Summary",
        data=tier_summary.to_csv(index=False),
        file_name="risk_tier_summary.csv",
        mime="text/csv",
    )

    st.caption(
        "Positive calibration gaps indicate that average predicted PD "
        "is above the observed default rate."
    )


# ---------------------------------------------------------------------
# Customer Lookup
# ---------------------------------------------------------------------

def customer_lookup(df):
    """Render the individual customer lookup page."""

    st.title("Customer Lookup")

    st.caption(
        "Review customer-level risk, profile information, and SHAP explanations."
    )

    prediction_df = df.dropna(
        subset=["pd"]
    ).copy()

    if prediction_df.empty:
        st.warning(
            "No customers with model predictions are available."
        )
        return

    prediction_df = prediction_df.sort_values(
        "customer_id"
    )

    customer_lookup_options = {
        f"{int(row.customer_id)} — {row.risk_tier} risk": int(
            row.customer_id
        )
        for row in prediction_df.itertuples()
    }

    selected_label = st.selectbox(
        "Select Customer",
        list(customer_lookup_options.keys()),
    )

    selected_id = customer_lookup_options[selected_label]

    customer = prediction_df[
        prediction_df["customer_id"] == selected_id
    ].iloc[0]

    st.divider()

    risk_tier = str(customer["risk_tier"])

    st.subheader(
        f"Customer {selected_id} "
        f"{risk_color(risk_tier)} {risk_tier} Risk"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Probability of Default",
            format_percentage(customer["pd"]),
        )

    with col2:
        st.metric(
            "Risk Tier",
            risk_tier,
        )

    with col3:
        segment = customer["customer_segment"]

        if pd.notna(segment):
            st.metric(
                "Customer Segment",
                f"Segment {int(segment)}",
            )
        else:
            st.metric(
                "Customer Segment",
                "N/A",
            )

    with col4:
        st.metric(
            "Credit Limit",
            f"${float(customer['credit_limit']):,.0f}",
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Profile")

        profile = {
            "Age": int(customer["age"]),
            "Gender": format_gender(customer["gender"]),
            "Education": format_education(customer["education"]),
            "Marital Status": format_marital_status(
                customer["marital_status"]
            ),
            "Delayed Months": int(
                customer["num_delayed_months"]
            ),
            "Maximum Delay": float(
                customer["max_delay"]
            ),
            "Average Delay": float(
                customer["avg_delay"]
            ),
            "Recent Delay": float(
                customer["recent_delay"]
            ),
            "Average Utilization": format_percentage(
                customer["avg_utilization"]
            ),
            "Maximum Utilization": format_percentage(
                customer["max_utilization"]
            ),
        }

        for label, value in profile.items():
            st.write(
                f"**{label}:** {value}"
            )

    with right:
        st.subheader("Risk Probability")

        pd_value = float(customer["pd"])

        st.markdown(
            risk_bar_html(
                pd_value,
                risk_tier,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"### {pd_value * 100:.2f}%"
        )

        if risk_tier == "High":
            st.error(
                "High-risk customer: elevated probability of default."
            )

        elif risk_tier == "Medium":
            st.warning(
                "Medium-risk customer: monitor repayment and utilization."
            )

        else:
            st.success(
                "Low-risk customer: comparatively lower probability of default."
            )

    st.divider()

    st.subheader("Individual SHAP Explanation")

    st.caption(
        "Top model contributions for this customer's predicted risk. "
        "Positive values increase predicted risk; negative values decrease it."
    )

    try:
        with st.spinner(
            "Calculating individual SHAP explanation..."
        ):
            explanation = calculate_customer_shap_by_id(
                int(selected_id)
            )

        if explanation.empty:
            st.warning(
                "No SHAP explanation is available for this customer."
            )
            return

        top_features = explanation.head(5).copy()

        display_df = top_features[
            [
                "feature",
                "feature_value",
                "shap_value",
                "direction",
            ]
        ].copy()

        display_df["shap_value"] = (
            display_df["shap_value"]
            .round(4)
        )

        display_df = display_df.rename(
            columns={
                "feature": "Feature",
                "feature_value": "Feature Value",
                "shap_value": "SHAP Value",
                "direction": "Impact",
            }
        )

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download SHAP Explanation",
            data=display_df.to_csv(index=False),
            file_name=f"customer_{selected_id}_shap.csv",
            mime="text/csv",
        )

        chart_df = top_features.sort_values(
            "shap_value"
        )

        fig = px.bar(
            chart_df,
            x="shap_value",
            y="feature",
            orientation="h",
            title="Top 5 Individual SHAP Contributions",
            labels={
                "shap_value": "SHAP Value",
                "feature": "Feature",
            },
            text="shap_value",
        )

        fig.add_vline(
            x=0,
            line_width=1,
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.caption(
            "Positive SHAP values push the model toward higher predicted "
            "default risk. Negative values push it toward lower predicted risk."
        )

    except Exception as exc:
        st.error(
            f"Unable to generate the SHAP explanation: {exc}"
        )


# ---------------------------------------------------------------------
# Model Monitoring
# ---------------------------------------------------------------------

def model_monitoring():
    """Render the model monitoring page."""

    st.title("Model Monitoring")

    st.caption(
        "Calibration and portfolio monitoring for the deployed baseline model."
    )

    monitoring_df = get_monitoring_data()

    if monitoring_df.empty:
        st.warning(
            "No monitoring data is available. "
            "Run the model monitoring pipeline first."
        )
        return

    overall = monitoring_df[
        monitoring_df["metric_group"] == "overall"
    ]

    if not overall.empty:
        row = overall.iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average PD",
                format_percentage(row["average_pd"]),
            )

        with col2:
            st.metric(
                "Observed Default Rate",
                format_percentage(
                    row["observed_default_rate"]
                ),
            )

        with col3:
            st.metric(
                "Calibration Gap",
                f"{row['calibration_gap'] * 100:.2f} pp",
            )

    st.divider()

    st.subheader("Risk-Tier Monitoring")

    risk_df = monitoring_df[
        monitoring_df["metric_group"] == "risk_tier"
    ].copy()

    if not risk_df.empty:
        risk_display = risk_df[
            [
                "group",
                "customers",
                "customer_share",
                "average_pd",
                "observed_default_rate",
                "calibration_gap",
            ]
        ].copy()

        risk_display["customer_share"] *= 100
        risk_display["average_pd"] *= 100
        risk_display["observed_default_rate"] *= 100
        risk_display["calibration_gap"] *= 100

        risk_display = risk_display.round(2)

        risk_display = risk_display.rename(
            columns={
                "group": "Risk Tier",
                "customers": "Customers",
                "customer_share": "Customer Share (%)",
                "average_pd": "Average PD (%)",
                "observed_default_rate": "Observed Default Rate (%)",
                "calibration_gap": "Calibration Gap (pp)",
            }
        )

        st.dataframe(
            risk_display,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Risk Monitoring",
            data=risk_display.to_csv(index=False),
            file_name="risk_tier_monitoring.csv",
            mime="text/csv",
        )

    st.divider()

    st.subheader("Customer Segment Monitoring")

    segment_df = monitoring_df[
        monitoring_df["metric_group"] == "customer_segment"
    ].copy()

    if not segment_df.empty:
        segment_display = segment_df[
            [
                "group",
                "customers",
                "customer_share",
                "average_pd",
                "observed_default_rate",
                "calibration_gap",
            ]
        ].copy()

        segment_display["customer_share"] *= 100
        segment_display["average_pd"] *= 100
        segment_display["observed_default_rate"] *= 100
        segment_display["calibration_gap"] *= 100

        segment_display = segment_display.round(2)

        segment_display = segment_display.rename(
            columns={
                "group": "Segment",
                "customers": "Customers",
                "customer_share": "Customer Share (%)",
                "average_pd": "Average PD (%)",
                "observed_default_rate": "Observed Default Rate (%)",
                "calibration_gap": "Calibration Gap (pp)",
            }
        )

        st.dataframe(
            segment_display,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Segment Monitoring",
            data=segment_display.to_csv(index=False),
            file_name="customer_segment_monitoring.csv",
            mime="text/csv",
        )

    st.divider()

    st.subheader("Monitoring Health")

    health = monitoring_df[
        monitoring_df["metric_group"] == "monitoring_status"
    ]

    if not health.empty:
        status = str(
            health.iloc[0]["status"]
        )

        if status == "HEALTHY":
            st.success(
                "Model monitoring status: HEALTHY"
            )
        else:
            st.warning(
                f"Model monitoring status: {status}"
            )


# ---------------------------------------------------------------------
# Model Information
# ---------------------------------------------------------------------

def model_info():
    """Render model information."""

    st.title("Model Information")

    st.caption(
        "Technical information about the deployed credit-risk model."
    )

    st.subheader("Credit-Risk Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            "XGBoost",
        )

    with col2:
        st.metric(
            "Model Version",
            "1.0",
        )

    with col3:
        st.metric(
            "Classification Threshold",
            "0.30",
        )

    st.divider()

    st.subheader("Model Configuration")

    st.write(
        """
        The production-style baseline model is an XGBoost classifier
        wrapped inside a preprocessing pipeline.

        The pipeline performs preprocessing before passing the transformed
        features to the XGBoost estimator.
        """
    )

    model = get_model()

    xgb_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Estimator parameters**")

        params = xgb_model.get_params()

        selected_params = {
            "n_estimators": params.get("n_estimators"),
            "learning_rate": params.get("learning_rate"),
            "max_depth": params.get("max_depth"),
            "random_state": params.get("random_state"),
        }

        st.json(selected_params)

    with col2:
        st.write("**Preprocessing**")

        feature_names = preprocessor.get_feature_names_out()

        st.metric(
            "Transformed Features",
            len(feature_names),
        )

        st.write(
            "Numeric features are passed through while categorical "
            "features are one-hot encoded."
        )

    st.divider()

    st.subheader("Model Performance Snapshot")

    monitoring_df = get_monitoring_data()

    overall = monitoring_df[
        monitoring_df["metric_group"] == "overall"
    ]

    if not overall.empty:
        row = overall.iloc[0]

        performance = {
            "Average PD": format_percentage(
                row["average_pd"]
            ),
            "Observed Default Rate": format_percentage(
                row["observed_default_rate"]
            ),
            "Calibration Gap": (
                f"{row['calibration_gap'] * 100:.2f} pp"
            ),
        }

        for label, value in performance.items():
            st.write(
                f"**{label}:** {value}"
            )

    st.info(
        "Individual SHAP explanations are available on the Customer Lookup page."
    )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def main():
    """Run the Streamlit application."""

    df, data_source, loaded_at = get_dashboard_data()

    with st.sidebar:
        st.title("💳 Credit Risk Intelligence")

        st.caption(
            f"Data source: **{data_source}**"
        )

        st.caption(
            "Data loaded: "
            f"{loaded_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        st.divider()

        page = st.radio(
            "Navigation",
            [
                "📊 Portfolio Overview",
                "👤 Customer Lookup",
                "📈 Model Monitoring",
                "⚙️ Model Info",
            ],
        )

        st.divider()

        st.caption(
            "XGBoost credit-risk decision-support dashboard"
        )

    if page == "📊 Portfolio Overview":
        filtered_df = apply_portfolio_filters(df)
        portfolio_overview(filtered_df)

    elif page == "👤 Customer Lookup":
        customer_lookup(df)

    elif page == "📈 Model Monitoring":
        model_monitoring()

    elif page == "⚙️ Model Info":
        model_info()


if __name__ == "__main__":
    main()