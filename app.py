import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import joblib

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = joblib.load("mmx_sales_model.pkl")

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Marketing Mix Simulator",
    layout="wide"
)

# --------------------------------------------------
# Sidebar controls (with metadata)
# --------------------------------------------------
st.sidebar.title("Scenario Controls")

distribution = st.sidebar.slider(
    "Weighted Distribution (%)",
    40, 60, 50,
    help="Percentage of category sales coming from outlets where the brand is present"
)

price_ratio = st.sidebar.slider(
    "Price Ratio vs Competition",
    0.8, 1.2, 1.0, 0.01,
    help="Brand price divided by competitor price (above 1 = premium pricing)"
)

adstock = st.sidebar.slider(
    "Advertising Adstock",
    0, 4000, 1000,
    help="Cumulative and lagged impact of advertising spend"
)

# --------------------------------------------------
# Prediction logic
# --------------------------------------------------
X_input = np.array([[distribution, price_ratio, np.log1p(adstock)]])
log_sales_pred = model.predict(X_input)[0]
sales_pred = np.expm1(log_sales_pred)

# Base scenario
X_base = np.array([[50, 1.0, np.log1p(1000)]])
base_sales = np.expm1(model.predict(X_base)[0])

# --------------------------------------------------
# Title & overview
# --------------------------------------------------
st.title("Marketing Mix Decision Simulator")
st.caption("Interactive what-if analysis based on a trained regression model")

st.subheader("Model Overview")
st.markdown("""
This simulator is based on a **log-linear Marketing Mix Model (MMX)** trained on
historical monthly data.

### Key Drivers
- **Weighted Distribution (%)** – Availability in high-value outlets  
- **Price Ratio vs Competition** – Relative price positioning  
- **Advertising Adstock** – Lagged and cumulative media impact with diminishing returns
""")

st.divider()

# --------------------------------------------------
# KPI section
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric(
    "Predicted Monthly Sales",
    f"{int(sales_pred):,}",
    f"{((sales_pred / base_sales - 1) * 100):.1f}%"
)

c2.metric("Base Scenario Sales", f"{int(base_sales):,}")
c3.metric("Sales Change vs Base", f"{int(sales_pred - base_sales):,}")

st.divider()

# --------------------------------------------------
# Explainability & calculation logic
# --------------------------------------------------
st.subheader("Explainability (Elasticities)")
st.markdown("""
- **Distribution elasticity:** +2.1%  
- **Price elasticity:** −0.57  
- **Advertising elasticity:** +0.01 (diminishing)

**Interpretation:**  
Distribution is the strongest long-term growth lever.  
Premium pricing reduces demand.  
Advertising supports sales but with diminishing marginal returns.
""")

st.subheader("How Predictions Are Calculated")
st.markdown("""
The model predicts **log(sales)** using regression coefficients.
The result is then converted back to actual monthly sales volume.
""")

st.divider()

# --------------------------------------------------
# Side-by-side charts (KEY CHANGE)
# --------------------------------------------------
left_col, right_col = st.columns(2)

# -------- Left: Scenario Comparison
with left_col:
    st.subheader("Scenario Comparison")

    compare_df = pd.DataFrame({
        "Scenario": ["Base", "Current"],
        "Sales": [base_sales, sales_pred]
    })

    scenario_chart = alt.Chart(compare_df).mark_bar(size=40).encode(
        x=alt.X(
            "Scenario:N",
            title=None,
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("Sales:Q", title="Sales Volume"),
        tooltip=["Scenario", "Sales"]
    ).properties(height=280)

    st.altair_chart(scenario_chart, use_container_width=True)

# -------- Right: Driver Contribution
with right_col:
    st.subheader("Driver Contribution (Directional Impact)")

    contrib_df = pd.DataFrame({
        "Driver": ["Distribution", "Price", "Advertising"],
        "Impact": [
            0.021 * (distribution - 50),
            -0.569 * (price_ratio - 1.0),
            0.012 * (np.log1p(adstock) - np.log1p(1000))
        ]
    })

    driver_chart = alt.Chart(contrib_df).mark_bar(size=28).encode(
        y=alt.Y("Driver:N", sort="-x", title=None),
        x=alt.X("Impact:Q", title="Directional Impact (log scale)"),
        tooltip=["Driver", "Impact"]
    ).properties(height=280)

    st.altair_chart(driver_chart, use_container_width=True)
    st.caption("Directional indicators only, not exact attribution.")

st.divider()

# --------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------
st.subheader("Sensitivity Analysis: Distribution vs Sales")
st.caption("Price and advertising held constant.")

dist_range = range(40, 61)
sales_curve = [
    np.expm1(model.predict(
        np.array([[d, price_ratio, np.log1p(adstock)]])
    )[0])
    for d in dist_range
]

curve_df = pd.DataFrame({
    "Distribution": dist_range,
    "Sales": sales_curve
})

st.line_chart(curve_df.set_index("Distribution"))

st.divider()

# --------------------------------------------------
# Save & compare scenarios
# --------------------------------------------------
st.subheader("Save & Compare Scenarios")

if "scenarios" not in st.session_state:
    st.session_state.scenarios = []

if st.button("Save Current Scenario"):
    st.session_state.scenarios.append({
        "Distribution": distribution,
        "Price Ratio": price_ratio,
        "Adstock": adstock,
        "Predicted Sales": int(sales_pred)
    })

if st.session_state.scenarios:
    scenario_df = pd.DataFrame(st.session_state.scenarios)
    st.dataframe(scenario_df)

    st.download_button(
        "Download Scenario Report (CSV)",
        scenario_df.to_csv(index=False),
        "mmx_simulation_results.csv",
        "text/csv"
    )
else:
    st.info("No scenarios saved yet.")
