import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Vehicle Insurance Propensity — Team LJV", page_icon="🚗", layout="centered")
st.title("🚗 Vehicle Insurance Propensity — Team LJV")
st.caption("Enter customer info to estimate probability of purchasing vehicle insurance.")

MODEL_PATH = Path("artifacts/final_pipeline.joblib")

@st.cache_resource(show_spinner=False)
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}"); st.stop()
    pipe = joblib.load(MODEL_PATH)
    if pipe is None:
        st.error("Model file exists but failed to load."); st.stop()
    return pipe

pipe = load_pipeline()
st.sidebar.success("✅ Model loaded")
st.sidebar.write(f"Using: {MODEL_PATH.resolve()}")

# schema from notebook
NUM_ALL = ["id", "Age", "Driving_License", "Region_Code", "Previously_Insured", "Annual_Premium", "Policy_Sales_Channel", "Vintage"]
CAT_ALL = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
OHE_CATS = {"Gender": ["Female", "Male"], "Vehicle_Age": ["1-2 Year", "< 1 Year", "> 2 Years"], "Vehicle_Damage": ["No", "Yes"]}
NUM_DEFAULTS = {"id": 190886.0, "Age": 36.0, "Driving_License": 1.0, "Region_Code": 28.0, "Previously_Insured": 0.0, "Annual_Premium": 31697.0, "Policy_Sales_Channel": 134.0, "Vintage": 154.0}
CAT_DEFAULTS = {"Gender": "Male", "Vehicle_Age": "1-2 Year", "Vehicle_Damage": "Yes"}
EXPOSE_NUM = ["Age", "Annual_Premium", "Vintage", "Driving_License", "Region_Code", "Policy_Sales_Channel", "Previously_Insured"]
EXPOSE_CAT = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
DESIRED_CATS = {"Gender": ["Male", "Female"], "Vehicle_Age": ["< 1 Year", "1-2 Year", "> 2 Years"], "Vehicle_Damage": ["Yes", "No"]}
LIMITS = {"Age": [20.0, 85.0], "Driving_License": [0.0, 1.0], "Region_Code": [0.0, 52.0], "Previously_Insured": [0.0, 1.0], "Annual_Premium": [2630.0, 540165.0], "Policy_Sales_Channel": [1.0, 163.0], "Vintage": [10.0, 299.0]}

LABELS = {
    "Annual_Premium":"Annual Premium",
    "Policy_Sales_Channel":"Policy Sales Channel",
    "Vehicle_Age":"Vehicle Age",
    "Vehicle_Damage":"Vehicle Damage",
    "Previously_Insured":"Previously Insured"
}
def label(c): return LABELS.get(c, c)

def ordered_choices(col, raw):
    want = DESIRED_CATS.get(col)
    if not want:
        return list(raw)
    extras = [x for x in raw if x not in want]
    return [x for x in want if x in raw] + extras

def predict_one_row(pipeline, row_dict):
    if pipeline is None:
        st.error("Pipeline not loaded."); st.stop()
    X = pd.DataFrame([row_dict])
    if hasattr(pipeline, "predict_proba"):
        p = float(pipeline.predict_proba(X)[:,1][0])
    elif hasattr(pipeline, "decision_function"):
        s = float(pipeline.decision_function(X)[0]); p = 1/(1+np.exp(-s))
    else:
        p = float(pipeline.predict(X)[0])
    return p, int(p >= 0.5)

def number_input_with_limits(col, default):
    low, high = LIMITS.get(col, (None, None))
    # integer-like fields formatting
    int_like = col in ["Driving_License","Previously_Insured","Region_Code","Policy_Sales_Channel","Vintage","Age"]
    step = 1.0 if int_like else 10.0
    fmt = "%.0f" if int_like else "%.2f"
    kwargs = {"label": label(col), "value": float(default), "step": step, "format": fmt}
    if low is not None: kwargs["min_value"] = float(low)
    if high is not None: kwargs["max_value"] = float(high)
    return st.number_input(**kwargs)

# --- UI ---
tab1, tab2, tab3 = st.tabs(["Single prediction","Batch CSV","About"])

with tab1:
    # start with defaults for full row
    row = {**NUM_DEFAULTS, **CAT_DEFAULTS}

    c1, c2 = st.columns(2)
    with c1:
        if EXPOSE_NUM: st.markdown("### Numeric features")
        left = EXPOSE_NUM[: max(1, len(EXPOSE_NUM)//2)]
        for col in left:
            row[col] = number_input_with_limits(col, NUM_DEFAULTS.get(col, LIMITS.get(col, (0,0))[0] or 0.0))

    with c2:
        if len(EXPOSE_NUM) > 1: st.markdown("### Numeric features (cont.)")
        right = EXPOSE_NUM[max(1, len(EXPOSE_NUM)//2):]
        for col in right:
            row[col] = number_input_with_limits(col, NUM_DEFAULTS.get(col, LIMITS.get(col, (0,0))[0] or 0.0))

    if EXPOSE_CAT: st.markdown("### Categorical features")
    for col in EXPOSE_CAT:
        raw = OHE_CATS.get(col, [])
        choices = ordered_choices(col, raw) if raw else []
        default = str(CAT_DEFAULTS.get(col, "Unknown"))
        if choices:
            idx = choices.index(default) if default in choices else 0
            row[col] = st.selectbox(label(col), choices, index=idx)
        else:
            row[col] = st.text_input(label(col), value=default)

    # fill any non-exposed columns (remain defaults)
    for c in NUM_ALL:
        row.setdefault(c, float(NUM_DEFAULTS.get(c, 0.0)))
    for c in CAT_ALL:
        row.setdefault(c, str(CAT_DEFAULTS.get(c, "Unknown")))

    if st.button("Predict", type="primary"):
        proba, pred = predict_one_row(pipe, row)
        st.metric("Estimated purchase probability", f"{proba:.3%}")
        st.write(f"Decision (0.50 threshold): **{pred}**")
        if proba >= 0.5:
            st.success("Recommend: include in today's outreach (high priority).")
        elif proba >= 0.2:
            st.info("Recommend: low-cost nurture (email/SMS).")
        else:
            st.warning("Recommend: suppress or deprioritize.")

with tab2:
    st.subheader("Upload CSV to score")
    up = st.file_uploader("CSV with the same columns used in training", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if hasattr(pipe, "predict_proba"):
            df["_score"] = pipe.predict_proba(df)[:,1]
        else:
            df["_score"] = pipe.predict(df)
        st.dataframe(df.head(20))
        out_path = "scored_output.csv"
        df.to_csv(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button("Download scored CSV", f, file_name="scored_output.csv", mime="text/csv")

with tab3:
    from pathlib import Path
    import json
    METRICS_PATH = Path("artifacts/metrics.json")
    FI_PATH = Path("artifacts/feature_importance.csv")
    LIFT_IMG = Path("artifacts/lift_chart.png")
    st.subheader("About the model")
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        base = m.get("base_rate", float("nan"))
        roc  = m.get("roc_auc",{}).get("value", float("nan"))
        pr   = m.get("pr_auc",{}).get("value", float("nan"))
        st.write(f"Base rate: {base:.2%}  |  ROC AUC: {roc:.4f}  |  PR AUC: {pr:.4f}")
        lift = m.get("lift",{})
        if lift:
            st.write("Lift Top 10%: {lift.get('top10',{}).get('value', float('nan')):.2f}×, "
                     "Top 20%: {lift.get('top20',{}).get('value', float('nan')):.2f}×, "
                     "Top 30%: {lift.get('top30',{}).get('value', float('nan')):.2f}×")
    if FI_PATH.exists():
        st.markdown("**Top features (permutation importance)**")
        st.dataframe(pd.read_csv(FI_PATH).head(10))
    if LIFT_IMG.exists():
        st.image(str(LIFT_IMG), caption="Lift by Contact Bucket")
    st.caption("This app uses the final scikit-learn Pipeline saved from the technical notebook.")
