import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(
    page_title="Reddit Fashion Forecast",
    layout="wide",
)

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("fashion_model.pkl", "rb") as f:
        return pickle.load(f)

art = load_artifacts()

clf = art["clf"]
reg = art["reg"]
surge_threshold = art["surge_threshold"]
FEATURE_COLS = art["feature_cols"]
latest_features = art["latest_features"]        # DataFrame-like, already safe_X processed
latest_df = art["latest_df"].copy()             # microtopic-level latest week data
ts_agg = art.get("ts_agg")
ts_feat = art.get("ts_feat")


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("Filters")

# Always build filter options from ALL time-series
all_brands = sorted(ts_agg["brand"].dropna().unique())
brand = st.sidebar.selectbox("Brand", ["(All Brands)"] + all_brands)

# Items depend on brand selection
if brand != "(All Brands)":
    items_list = sorted(ts_agg[ts_agg["brand"] == brand]["item"].dropna().unique())
else:
    items_list = sorted(ts_agg["item"].dropna().unique())

item = st.sidebar.selectbox("Item", ["(All Items)"] + items_list)


# ---------------------------------------------------------
# UNIVERSAL FILTER FUNCTION
# ---------------------------------------------------------
def apply_filters(df):
    out = df.copy()
    if brand != "(All Brands)":
        out = out[out["brand"] == brand]
    if item != "(All Items)":
        out = out[out["item"] == item]
    return out

# Apply filters universally
filtered_latest = apply_filters(latest_df)
filtered_agg = apply_filters(ts_agg)
filtered_ts  = apply_filters(ts_feat)


# ---------------------------------------------------------
# MODEL PREDICTIONS (safe merge on microtopic)
# ---------------------------------------------------------
if not filtered_latest.empty:
    # Build a features lookup table by microtopic
    features_lookup = latest_features.copy()
    features_lookup["microtopic"] = latest_df["microtopic"].values

    # Merge filtered_latest with its matching features row
    merged = filtered_latest.merge(
        features_lookup,
        on="microtopic",
        how="left",
        suffixes=("", "_feat")
    )

    # Extract only the feature columns used in training
    X = merged[FEATURE_COLS].astype(float)

    # Predict
    filtered_latest["surge_prob"] = clf.predict_proba(X)[:, 1]
    filtered_latest["pred_next_engagement"] = reg.predict(X)

    # Weighted surge
    filtered_latest["weighted_surge"] = (
        filtered_latest["surge_prob"] * np.log1p(filtered_latest["engagement_sum"])
    )

# ---------------------------------------------------------
# SAFETY CHECK: No rows after filtering
# ---------------------------------------------------------
if filtered_latest.empty:
    st.warning("No microtopics match this brand + item selection.")
    st.stop()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Historical Trends", "Forecast", "Surge Analysis", "Explore Raw Data"]
)

# ---------------------------------------------------------
# TAB 1 — OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.title("Fashion Forecast Overview")

    st.write("""
    This tool forecasts microtopic-level trends in Reddit fashion communities,
    including next-week engagement predictions and surge probability alerts.
    """)

    colA, colB = st.columns(2)
    colA.metric("Microtopics (filtered)", f"{len(filtered_latest):,}")
    colB.metric("Surge Threshold", f"{surge_threshold:.3f}")

    st.subheader("Top microtopics (filtered)")
    st.dataframe(
        filtered_latest[[
            "microtopic", "brand", "item",
            "engagement_sum", "sentiment_mean",
            "surge_prob", "pred_next_engagement"
        ]].sort_values("pred_next_engagement", ascending=False).head(20)
    )

    # -----------------------------------------------------
    # Top 10 items by predicted next-week engagement
    # -----------------------------------------------------
    st.subheader("Top 10 items next week (predicted)")

    if "item" in filtered_latest.columns and "pred_next_engagement" in filtered_latest.columns:
        # Aggregate microtopic-level forecasts up to item-level
        item_forecast = (
            filtered_latest
            .dropna(subset=["item", "pred_next_engagement"])
            .groupby(["brand", "item"])
            .agg(
                total_pred_next_engagement=("pred_next_engagement", "sum"),
                avg_pred_next_engagement=("pred_next_engagement", "mean"),
                n_microtopics=("microtopic", "nunique"),
            )
            .reset_index()
        )

        # Top 10 by TOTAL predicted engagement for next week
        top10_items = (
            item_forecast
            .sort_values("total_pred_next_engagement", ascending=False)
            .head(10)
        )

        # Table view
        st.dataframe(
            top10_items[
                ["brand", "item", "total_pred_next_engagement",
                 "avg_pred_next_engagement", "n_microtopics"]
            ].rename(
                columns={
                    "brand": "Brand",
                    "item": "Item",
                    "total_pred_next_engagement": "Predicted engagement (total)",
                    "avg_pred_next_engagement": "Predicted engagement (avg per microtopic)",
                    "n_microtopics": "# microtopics",
                }
            )
        )

        # Bar chart for a quick visual
        fig_items = px.bar(
            top10_items,
            x="total_pred_next_engagement",
            y="item",
            color="brand",
            orientation="h",
            title="Top 10 items by predicted engagement next week",
        )
        st.plotly_chart(fig_items, use_container_width=True)

    else:
        st.info("Item-level data not available; showing only microtopic-level forecasts.")

# ---------------------------------------------------------
# TAB 2 — HISTORICAL TRENDS (Optional Future Expansion)
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Historical Trends")

    if filtered_agg.empty:
        st.info("No time-series data available for this selection.")
    else:
        hist = filtered_agg.groupby("week_start", as_index=False).agg(
            engagement=("engagement_sum", "sum"),
            posts=("posts", "sum"),
            sentiment=("sentiment_mean", "mean"),
        )

        fig = px.line(
            hist,
            x="week_start",
            y="engagement",
            title="Engagement Over Time",
            labels={"week_start": "Week", "engagement": "Engagement"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Historical Table")
        st.dataframe(hist)

# ---------------------------------------------------------
# TAB 3 — FORECAST
# ---------------------------------------------------------
with tab3:
    st.title("Forecast")

    st.write("Predicted engagement for next week:")

    top_pred = filtered_latest.sort_values("pred_next_engagement", ascending=False).head(15)

    fig = px.bar(
        top_pred,
        x="microtopic",
        y="pred_next_engagement",
        color="pred_next_engagement",
        title="Next Week Engagement Forecast",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(top_pred)

# ---------------------------------------------------------
# TAB 4 — SURGE ANALYSIS
# ---------------------------------------------------------
with tab4:
    st.title("Surge Analysis")

    st.write(f"Microtopics with surge probability above {surge_threshold:.3f}:")

    alerts = filtered_latest[filtered_latest["surge_prob"] >= surge_threshold].copy()

    if alerts.empty:
        st.info("No microtopics exceed the surge probability threshold for this selection.")
    else:
        alerts = alerts.sort_values("surge_prob", ascending=False)

        fig = px.bar(
            alerts.head(30),
            x="microtopic",
            y="surge_prob",
            color="surge_prob",
            title="Microtopics with Highest Surge Probability",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Surge Alerts Table")
        st.dataframe(alerts)

    st.subheader("Top Surging Microtopics (Weighted)")
    top_surge = filtered_latest.sort_values("weighted_surge", ascending=False).head(20)

    fig2 = px.bar(
        top_surge,
        x="microtopic",
        y="weighted_surge",
        color="surge_prob",
        title="Top Surging Microtopics"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(top_surge)

# ---------------------------------------------------------
# TAB 5 — RAW DATA
# ---------------------------------------------------------
with tab5:
    st.title("Explore Raw Latest-Week Microtopics")
    st.dataframe(filtered_latest)
