
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Melbourne Pedestrian Intelligence",
    page_icon="🚶",
    layout="wide"
)

# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 750;
    margin-bottom: 0rem;
}

.subtitle {
    font-size: 1rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
}

div[data-testid="stMetric"] {
    background-color: #F7F9FA;
    border: 1px solid #E5E7EB;
    padding: 18px;
    border-radius: 12px;
}

.insight-box {
    background: #F2F8F7;
    border-left: 5px solid #159A8C;
    padding: 15px 18px;
    border-radius: 6px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    sensors = pd.read_csv(
        "reliable_sensor_catalog.csv"
    )

    forecasts = pd.read_csv(
        "melbourne_timesfm_forecasts.csv"
    )

    performance = pd.read_csv(
        "model_performance.csv"
    )

    anomalies = pd.read_csv(
        "pedestrian_unusual_activity.csv"
    )

    test_results = pd.read_csv(
        "timesfm_final_test_results.csv"
    )

    forecasts["datetime"] = pd.to_datetime(
        forecasts["datetime"]
    )

    anomalies["datetime"] = pd.to_datetime(
        anomalies["datetime"]
    )

    test_results["datetime"] = pd.to_datetime(
        test_results["datetime"]
    )

    return (
        sensors,
        forecasts,
        performance,
        anomalies,
        test_results
    )


(
    sensors,
    forecasts,
    performance,
    anomalies,
    test_results
) = load_data()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚶 Melbourne Pedestrian Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Interactive pedestrian analytics and AI forecasting using Melbourne sensor data
    and the TimesFM 3 time-series foundation model.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

available_sensors = (
    forecasts[
        ["sensor_id", "sensor_name"]
    ]
    .drop_duplicates()
    .sort_values("sensor_name")
)

sensor_options = dict(
    zip(
        available_sensors["sensor_name"],
        available_sensors["sensor_id"]
    )
)

selected_name = st.sidebar.selectbox(
    "Select pedestrian sensor",
    list(sensor_options.keys())
)

selected_sensor = sensor_options[selected_name]

selected_forecast = forecasts[
    forecasts["sensor_id"] == selected_sensor
].copy()

st.sidebar.markdown("---")

st.sidebar.caption(
    "Forecast horizon: 168 hours (7 days)"
)

st.sidebar.caption(
    "Model: TimesFM 3"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🏙️ Overview",
    "🔮 AI Forecast",
    "🏆 Model Performance",
    "⚠️ Unusual Activity"
])


# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab1:

    st.subheader("Project Overview")

    timesfm_wape = performance.loc[
        performance["Model"] == "TimesFM 3",
        "WAPE"
    ].iloc[0]

    baseline_wape = performance.loc[
        performance["Model"] == "Weekly Seasonal Baseline",
        "WAPE"
    ].iloc[0]

    improvement = (
        (baseline_wape - timesfm_wape)
        / baseline_wape
        * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Reliable Sensors",
        len(sensors)
    )

    c2.metric(
        "AI Forecast Locations",
        forecasts["sensor_id"].nunique()
    )

    c3.metric(
        "TimesFM Test WAPE",
        f"{timesfm_wape:.2f}%"
    )

    c4.metric(
        "Error Reduction",
        f"{improvement:.1f}%"
    )

    st.markdown("### What this project does")

    st.markdown("""
    This project combines historical pedestrian analysis with modern
    time-series AI forecasting.

    **Main capabilities**

    - Explore pedestrian activity patterns across Melbourne
    - Compare changes across locations and time
    - Forecast hourly pedestrian activity
    - Quantify prediction uncertainty
    - Detect unusually high or low activity
    - Compare a foundation model against a seasonal benchmark
    """)

    st.markdown(
        """
        <div class="insight-box">
        <b>Key modelling result:</b><br>
        On 456 previously unseen hourly observations,
        TimesFM achieved a WAPE of <b>8.49%</b>,
        compared with <b>11.69%</b> for the weekly seasonal baseline.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TAB 2 — FORECAST
# =========================================================

with tab2:

    st.subheader(
        f"7-Day Forecast — {selected_name}"
    )

    selected_forecast = (
        selected_forecast
        .sort_values("datetime")
    )

    # -------------------------
    # KPI calculations
    # -------------------------

    peak = selected_forecast.loc[
        selected_forecast["forecast"].idxmax()
    ]

    total = selected_forecast["forecast"].sum()

    hourly_average = (
        selected_forecast["forecast"].mean()
    )

    selected_forecast["date"] = (
        selected_forecast["datetime"].dt.date
    )

    daily = (
        selected_forecast
        .groupby("date")
        .agg(
            hours=("forecast", "size"),
            predicted_total=("forecast", "sum")
        )
        .reset_index()
    )

    complete_days = daily[
        daily["hours"] == 24
    ]

    if len(complete_days) > 0:

        busiest_day = complete_days.loc[
            complete_days[
                "predicted_total"
            ].idxmax()
        ]

        busiest_day_text = (
            f"{busiest_day['date']}"
        )

    else:
        busiest_day_text = "N/A"

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Peak Hour",
        f"{peak['forecast']:,.0f}/hr"
    )

    k2.metric(
        "Peak Time",
        peak["datetime"].strftime(
            "%a %d %b, %I %p"
        )
    )

    k3.metric(
        "7-Day Total",
        f"{total:,.0f}"
    )

    k4.metric(
        "Hourly Average",
        f"{hourly_average:,.0f}"
    )


    # -------------------------
    # Forecast chart
    # -------------------------

    fig = go.Figure()

    # Upper interval
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["datetime"],
            y=selected_forecast["upper_80"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    # Lower interval
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["datetime"],
            y=selected_forecast["lower_80"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(21,154,140,0.17)",
            line=dict(width=0),
            name="80% Prediction Interval",
            hoverinfo="skip"
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=selected_forecast["datetime"],
            y=selected_forecast["forecast"],
            mode="lines",
            name="TimesFM Forecast",
            line=dict(
                color="#159A8C",
                width=3
            ),
            hovertemplate=(
                "<b>%{x|%A %d %b, %I %p}</b>"
                "<br>Forecast: %{y:,.0f}/hour"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Predicted Pedestrian Activity",
        xaxis_title="Date and Time",
        yaxis_title="Pedestrian Detections per Hour",
        template="plotly_white",
        height=520,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.08,
            x=0.5,
            xanchor="center"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -------------------------
    # Daily totals
    # -------------------------

    daily_chart = (
        selected_forecast
        .groupby(
            selected_forecast[
                "datetime"
            ].dt.strftime("%a %d %b")
        )["forecast"]
        .sum()
        .reset_index()
    )

    daily_chart.columns = [
        "Day",
        "Predicted Activity"
    ]

    fig_daily = px.bar(
        daily_chart,
        x="Day",
        y="Predicted Activity",
        title="Predicted Activity by Day"
    )

    fig_daily.update_traces(
        marker_color="#D99A53"
    )

    fig_daily.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="",
        yaxis_title="Predicted Detections"
    )

    st.plotly_chart(
        fig_daily,
        use_container_width=True
    )


# =========================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================

with tab3:

    st.subheader(
        "TimesFM 3 vs Weekly Seasonal Baseline"
    )

    tfm = performance[
        performance["Model"] == "TimesFM 3"
    ].iloc[0]

    baseline = performance[
        performance["Model"]
        == "Weekly Seasonal Baseline"
    ].iloc[0]

    p1, p2, p3 = st.columns(3)

    mae_improvement = (
        (baseline["MAE"] - tfm["MAE"])
        / baseline["MAE"]
        * 100
    )

    rmse_improvement = (
        (baseline["RMSE"] - tfm["RMSE"])
        / baseline["RMSE"]
        * 100
    )

    wape_improvement = (
        (baseline["WAPE"] - tfm["WAPE"])
        / baseline["WAPE"]
        * 100
    )

    p1.metric(
        "WAPE Reduction",
        f"{wape_improvement:.1f}%"
    )

    p2.metric(
        "MAE Reduction",
        f"{mae_improvement:.1f}%"
    )

    p3.metric(
        "RMSE Reduction",
        f"{rmse_improvement:.1f}%"
    )

    improvement_df = pd.DataFrame({
        "Metric": [
            "WAPE",
            "MAE",
            "RMSE"
        ],
        "Error Reduction": [
            wape_improvement,
            mae_improvement,
            rmse_improvement
        ]
    })

    fig_perf = px.bar(
        improvement_df,
        x="Error Reduction",
        y="Metric",
        orientation="h",
        text="Error Reduction",
        title=(
            "Error Reduction Compared "
            "with Weekly Baseline"
        )
    )

    fig_perf.update_traces(
        marker_color="#159A8C",
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_perf.update_layout(
        template="plotly_white",
        height=400,
        xaxis_title="Error Reduction (%)",
        yaxis_title="",
        xaxis_ticksuffix="%"
    )

    st.plotly_chart(
        fig_perf,
        use_container_width=True
    )

    st.markdown("### Final Holdout Test")

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Evaluation used 456 unseen hourly observations "
        "from July–September 2026."
    )


# =========================================================
# TAB 4 — UNUSUAL ACTIVITY
# =========================================================

with tab4:

    st.subheader(
        "Forecast-Based Unusual Activity Detection"
    )

    st.markdown("""
    These periods contain some of the largest differences between
    actual pedestrian activity and the TimesFM expectation.

    They should be interpreted as **unusual activity candidates**,
    not confirmed real-world events.
    """)

    anomalies = anomalies.sort_values(
        "absolute_error",
        ascending=False
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Unusual Hours",
        len(anomalies)
    )

    a2.metric(
        "Largest Difference",
        f"{anomalies['absolute_error'].max():,.0f}/hr"
    )

    largest = anomalies.iloc[0]

    a3.metric(
        "Largest Event",
        largest["datetime"].strftime(
            "%d %b %Y"
        )
    )

    # --------------------------------------
    # Scatter plot
    # --------------------------------------

    fig_anomaly = go.Figure()

    fig_anomaly.add_trace(
        go.Scatter(
            x=anomalies["datetime"],
            y=anomalies["actual"],
            mode="markers",
            name="Actual",
            marker=dict(
                size=11,
                color="#D46A5D"
            )
        )
    )

    fig_anomaly.add_trace(
        go.Scatter(
            x=anomalies["datetime"],
            y=anomalies["timesfm"],
            mode="markers",
            name="TimesFM Expected",
            marker=dict(
                size=9,
                color="#159A8C",
                symbol="diamond"
            )
        )
    )

    fig_anomaly.update_layout(
        title=(
            "Observed vs Expected Activity "
            "During Unusual Periods"
        ),
        xaxis_title="Date",
        yaxis_title="Pedestrian Detections / Hour",
        template="plotly_white",
        height=470,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.08
        )
    )

    st.plotly_chart(
        fig_anomaly,
        use_container_width=True
    )

    st.markdown("### Highest-Deviation Periods")

    display_anomalies = anomalies[
        [
            "datetime",
            "actual",
            "timesfm",
            "residual",
            "absolute_error"
        ]
    ].copy()

    display_anomalies.columns = [
        "Date & Time",
        "Actual",
        "Expected",
        "Difference",
        "Absolute Difference"
    ]

    st.dataframe(
        display_anomalies.head(15),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Melbourne Pedestrian Intelligence • "
    "TimesFM 3 forecasting • "
    "Pedestrian counts represent sensor detections, "
    "not unique individuals."
)
