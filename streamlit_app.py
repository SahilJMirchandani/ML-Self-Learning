import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# STREAMLIT - RETAIL SALES FORECASTING
# ============================================================

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.title("📈 Retail Sales Forecasting")
st.write(
    "Machine Learning application for predicting daily retail revenue "
    "using a tuned Random Forest Regression model."
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

@st.cache_data
def load_data():
    train = pd.read_csv("data/ml_train.csv")
    validation = pd.read_csv("data/ml_validation.csv")
    test = pd.read_csv("data/ml_test.csv")

    return train, validation, test


train, validation, test = load_data()


# ------------------------------------------------------------
# Features and target
# ------------------------------------------------------------

features = [
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_7_Day_Mean"
]

target = "Daily_Revenue"


# ------------------------------------------------------------
# Train final tuned model
# ------------------------------------------------------------

@st.cache_resource
def train_model():

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=4,
        random_state=42
    )

    model.fit(
        train[features],
        train[target]
    )

    return model


model = train_model()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Model Information")

st.sidebar.write("**Algorithm:** Random Forest Regression")
st.sidebar.write("**Estimators:** 100")
st.sidebar.write("**Maximum Depth:** 10")
st.sidebar.write("**Minimum Samples Leaf:** 4")


# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------

st.header("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Test MAE",
    "18,918.01"
)

col2.metric(
    "Test RMSE",
    "29,559.63"
)

col3.metric(
    "Test R²",
    "0.1771"
)


# ------------------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------------------

st.header("Actual vs Predicted Revenue")

test_predictions = model.predict(test[features])

comparison = pd.DataFrame({
    "Date": test["Date"],
    "Actual Revenue": test[target],
    "Predicted Revenue": test_predictions
})

st.line_chart(
    comparison.set_index("Date")[
        ["Actual Revenue", "Predicted Revenue"]
    ]
)


# ------------------------------------------------------------
# Prediction interface
# ------------------------------------------------------------

st.header("Predict Daily Revenue")

st.write(
    "Enter the required time-series features to generate "
    "a revenue prediction."
)

col1, col2 = st.columns(2)

with col1:

    day_of_week = st.number_input(
        "Day of Week (0 = Monday)",
        min_value=0,
        max_value=6,
        value=0
    )

    day_of_month = st.number_input(
        "Day of Month",
        min_value=1,
        max_value=31,
        value=15
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

    week_of_year = st.number_input(
        "Week of Year",
        min_value=1,
        max_value=53,
        value=25
    )


with col2:

    lag_1 = st.number_input(
        "Previous Day Revenue (Lag 1)",
        min_value=0.0,
        value=50000.0
    )

    lag_7 = st.number_input(
        "Revenue 7 Days Ago (Lag 7)",
        min_value=0.0,
        value=50000.0
    )

    lag_14 = st.number_input(
        "Revenue 14 Days Ago (Lag 14)",
        min_value=0.0,
        value=50000.0
    )

    rolling_mean = st.number_input(
        "7-Day Rolling Mean",
        min_value=0.0,
        value=50000.0
    )


# ------------------------------------------------------------
# Prediction button
# ------------------------------------------------------------

if st.button("Predict Revenue"):

    input_data = pd.DataFrame([{
        "Day_of_Week": day_of_week,
        "Day_of_Month": day_of_month,
        "Month": month,
        "Week_of_Year": week_of_year,
        "Lag_1": lag_1,
        "Lag_7": lag_7,
        "Lag_14": lag_14,
        "Rolling_7_Day_Mean": rolling_mean
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Daily Revenue: ₹{prediction:,.2f}"
    )


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------

st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)


# ------------------------------------------------------------
# Dataset preview
# ------------------------------------------------------------

st.header("Test Dataset Preview")

st.dataframe(
    test.head(10),
    use_container_width=True
)


st.caption(
    "Developed as part of ML Self-Learning using the UCI Online Retail dataset."
)