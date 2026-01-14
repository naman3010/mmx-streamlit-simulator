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
    min_value=40,
    max_value=60,
    value=50,
    help="Percentage of category sales coming from outlets where the brand is present"
)

price_ratio = st.sidebar.slider(
    "Price Ratio vs Competition",
    min_value=0.8,
    max_value=1.2,
    step=0.01,
    value=1.0,
    help="Brand price divided by average competitor price (above 1 = premium pricing)"
)

adstock = st.sidebar.slider(
    "Advertising Adstock",
    min_value=0,
    max_value=4000,
    value=1000,
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
# Main title
# --------------------------------------------------
st.title("Marketing Mix Decision Simulator")
st.caption("Interactive what-if analysis based on a trained regression model")

# --------------------------------------------------
# STEP 2A: Model Overview
# --------------------------------------------------
st.subheader("Model Overview")

st.markdown("""
This simulator is based on a **log-linear Marketing Mix Model (MMX)** 
trained on historical monthly data.

### Key Drivers Used
- **Weighted Distribution (%)**  
  Measures how widely the brand is available across high-value outlets.

- **Price Ratio vs Competition**  
  Ratio of brand price to competitor price. Values above 1 indicate premium pricing.

- **Advertising Adstock**  
  Captures the cumulative and lagged effect of advertising with diminishing returns.
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

c2.metric(
    "Base Scenario Sales",
    f"{int(base_sales):,}"
)

c3.metric(
    "Sales Change vs Base",
    f"{int(sales_pred - base_sales):,}"
)

st.divider()

# --------------------------------------------------
# Scenario comparison chart (clean & aesthetic)
# --------------------------------------------------
st.subheader("Scenario Comparison")

compare_df = pd.DataFrame({
    "Scenario": ["Base", "Current"],
    "Sales": [base_sales, sales_pred]
})

scenario_chart = alt.Chart(compare_df).mark_bar(size=45).encode(
    x=alt.X("Scenario:N", title=None),
    y=alt.Y("Sales:Q", title="Sales Volume"),
    tooltip=["Scenario", "Sales"]
).properties(
    height=300
)

st.altair_chart(scenario_chart, use_container_width=True)

st.divider()

# --------------------------------------------------
# Explainability panel (elasticities)
# --------------------------------------------------
st.subheader("Explainability (Elasticities)")

st.markdown("""
**Model Insights**
- **Distribution elasticity:** +2.1%  
- **Price elasticity:** −0.57  
- **Advertising elasticity:** +0.01 (diminishing returns)

**Interpretation**
- Distribution is the strongest long-term growth lever  
- Higher pricing vs competition reduces demand  
- Advertising supports sales but with diminishing marginal impact
""")

# --------------------------------------------------
# STEP 2B: How predictions are calculated
# --------------------------------------------------
st.subheader("How Predictions Are Calculated")

st.markdown("""
The model predicts **log(sales)** using a regression structure:

- Distribution contributes positively to base demand  
- Price has a negative impact when above competition  
- Advertising impact is modeled using a logarithmic (diminishing return) function  

Predicted values are then converted back to actual monthly sales volume.
""")

st.divider()

# --------------------------------------------------
# Driver contribution (horizontal & readable)
# --------------------------------------------------
st.subheader("Driver Contribution (Directional Impact)")

dist_effect = 0.021 * (distribution - 50)
price_effect = -0.569 * (price_ratio - 1.0)
ad_effect = 0.012 * (np.log1p(adstock) - np.log1p(1000))

contrib_df = pd.DataFrame({
    "Driver": ["Distribution", "Price", "Advertising"],
    "Impact": [dist_effect, price_effect, ad_effect]
})

driver_chart = alt.Chart(contrib_df).mark_bar(size=28).encode(
    y=alt.Y("Driver:N", sort="-x", title=None),
    x=alt.X("Impact:Q", title="Directional Impact (log scale)"),
    tooltip=["Driver", "Impact"]
).properties(
    height=250
)

st.altair_chart(driver_chart, use_container_width=True)

st.caption("Note: Contribution values are directional indicators, not exact attribution.")

st.divider()

# --------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------
st.subheader("Sensitivity Analysis: Distribution vs Sales")
st.caption(
    "Shows how sales respond to changes in distribution while price and advertising are held constant."
)

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
        label="Download Scenario Report (CSV)",
        data=scenario_df.to_csv(index=False),
        file_name="mmx_simulation_results.csv",
        mime="text/csv"
    )
else:
    st.info("No scenarios saved yet.")
