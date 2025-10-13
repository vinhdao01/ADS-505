import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Vehicle Insurance Propensity", page_icon="🚗", layout="centered")
st.title("🚗 Vehicle Insurance Propensity — Team LJV")

MODEL_PATH = Path("artifacts/final_pipeline.joblib")
METRICS_PATH = Path("artifacts/metrics.json")
FI_PATH = Path("artifacts/feature_importance.csv")
LIFT_IMG = Path("artifacts/lift_chart.png")

@st.cache_resource(show_spinner=False)
def load_pipeline_safe():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run the notebook save step first.")
        st.stop()
    pipe = joblib.load(MODEL_PATH)
    if pipe is None:
        st.error("Model file exists but failed to load.")
        st.stop()
    return pipe

pipe = load_pipeline_safe()
st.sidebar.success("✅ Model loaded")
st.sidebar.write(f"Using: {MODEL_PATH.resolve()}")

def get_schema_from_pipeline(pipeline):
    """Return numeric columns, categorical columns, and OHE categories from the fitted pipeline."""
    pre = pipeline.named_steps["pre"]
    num_cols, cat_cols, cat_categories = [], [], {}
    # transformers_: [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols), ...]
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
            try:
                ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            except Exception:
                ohe = pre.named_transformers_["cat"]
            if hasattr(ohe, "categories_"):
                for c, cats in zip(cat_cols, ohe.categories_):
                    cat_categories[c] = list(map(str, cats))
    return num_cols, cat_cols, cat_categories

def predict_one_row(pipeline, row_dict):
    if pipeline is None:
        st.error("Pipeline not loaded.")
        st.stop()
    df = pd.DataFrame([row_dict])
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(df)[:, 1][0])
    elif hasattr(pipeline, "decision_function"):
        s = float(pipeline.decision_function(df)[0])
        proba = 1 / (1 + np.exp(-s))
    else:
        proba = float(pipeline.predict(df)[0])
    return proba, int(proba >= 0.5)

def score_csv(pipeline, file):
    df = pd.read_csv(file)
    df["_score"] = pipeline.predict_proba(df)[:, 1]
    return df

# Build UI from the trained pipeline schema
num_cols, cat_cols, cat_choices = get_schema_from_pipeline(pipe)

tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch CSV", "About"])

with tab1:
    st.subheader("Enter customer details")

    user_inputs = {}
    if num_cols:
        st.markdown("**Numeric features**")
    for col in num_cols:
        # numeric default 0; tweak if you want custom defaults
        user_inputs[col] = st.number_input(col, value=0.0, step=1.0, format="%.4f")

    if cat_cols:
        st.markdown("**Categorical features**")
    for col in cat_cols:
        choices = cat_choices.get(col, [])
        if choices:
            user_inputs[col] = st.selectbox(col, choices, index=0)
        else:
            user_inputs[col] = st.text_input(col, value="Unknown")

    if st.button("Predict", type="primary"):
        proba, pred = predict_one_row(pipe, user_inputs)
        st.metric("Estimated purchase probability", f"{proba:.3%}")
        st.write("Decision (0.50 threshold): **{}**".format(pred))
        if proba >= 0.5:
            st.success("Recommend: include in today's outreach (high priority).")
        elif proba >= 0.2:
            st.info("Recommend: low-cost nurture (email/SMS).")
        else:
            st.warning("Recommend: suppress or deprioritize.")

with tab2:
    st.subheader("Upload a CSV to score")
    up = st.file_uploader("CSV with the same columns used in training", type=["csv"])
    if up is not None:
        out_df = score_csv(pipe, up)
        st.dataframe(out_df.head(20))
        out_path = "scored_output.csv"
        out_df.to_csv(out_path, index=False)
        with open(out_path, "rb") as f:
            st.download_button("Download scored CSV", f, file_name="scored_output.csv", mime="text/csv")

with tab3:
    st.subheader("About the model")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        base = metrics.get("base_rate", float("nan"))
        roc  = metrics.get("roc_auc", {}).get("value", float("nan"))
        pr   = metrics.get("pr_auc", {}).get("value", float("nan"))
        st.write(f"Base rate: {base:.2%}  |  ROC AUC: {roc:.4f}  |  PR AUC: {pr:.4f}")
        lift = metrics.get("lift", {})
        if lift:
            st.write(
                "Lift Top 10%: {0:.2f}×, Top 20%: {1:.2f}×, Top 30%: {2:.2f}×".format(
                    lift.get("top10", {}).get("value", float("nan")),
                    lift.get("top20", {}).get("value", float("nan")),
                    lift.get("top30", {}).get("value", float("nan")),
                )
            )
    if FI_PATH.exists():
        st.markdown("**Top features (permutation importance)**")
        st.dataframe(pd.read_csv(FI_PATH).head(10))
    if LIFT_IMG.exists():
        st.image(str(LIFT_IMG), caption="Lift by Contact Bucket")

    st.caption("This app uses the final scikit-learn Pipeline saved from the technical notebook.")