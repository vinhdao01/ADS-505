\
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Vehicle Insurance Propensity", page_icon="🚗", layout="centered")
st.title("🚗 Vehicle Insurance Propensity")

MODEL_PATH = r"artifacts/vehicle_insurance_propensity_NoneType.joblib"
model = joblib.load(MODEL_PATH)

st.caption("Enter customer info to estimate probability of purchasing vehicle insurance.")

NUM_COLS = ["id", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
CAT_COLS = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
CAT_CHOICES = {"Gender": ["Male", "Female"], "Vehicle_Age": ["1-2 Year", "< 1 Year", "> 2 Years"], "Vehicle_Damage": ["Yes", "No"]}
NUM_DEFAULTS = {"id": 190886.0, "Age": 36.0, "Driving_License": 1.0, "Region_Code": 28.0, "Previously_Insured": 0.0, "Annual_Premium": 31697.0, "Policy_Sales_Channel": 134.0, "Vintage": 154.0}

cols = st.columns(2)
data = {}

with cols[0]:
    st.subheader("Numeric features")
    left = NUM_COLS[: max(1, len(NUM_COLS)//2)]
    for col in left:
        default = float(NUM_DEFAULTS.get(col, 0.0))
        data[col] = st.number_input(col, value=default)

with cols[1]:
    st.subheader("Numeric features (cont.)")
    right = NUM_COLS[max(1, len(NUM_COLS)//2):]
    for col in right:
        default = float(NUM_DEFAULTS.get(col, 0.0))
        data[col] = st.number_input(col, value=default)

st.subheader("Categorical features")
for col in CAT_COLS:
    choices = CAT_CHOICES.get(col, [])
    if choices:
        data[col] = st.selectbox(col, options=choices, index=0)
    else:
        data[col] = st.text_input(col, value="")

if st.button("Predict"):
    X = pd.DataFrame([data])
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        proba = 1.0/(1.0 + np.exp(-score))
    else:
        proba = float(model.predict(X)[0])
    st.metric("Estimated Purchase Probability", f"{proba:.3%}")

    if proba >= 0.5:
        st.success("Recommend: High-priority outreach (call/agent).")
    elif proba >= 0.2:
        st.info("Recommend: Programmatic email/SMS nurturing.")
    else:
        st.warning("Recommend: Suppress or lower-cost channels.")

st.caption("Model file: " + MODEL_PATH)
