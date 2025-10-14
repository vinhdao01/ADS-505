Vehicle Insurance Cross-Sell — Propensity Modeling (README)

Predict which existing health-insurance customers are most likely to purchase vehicle insurance next. The goal is to lift conversions, reduce CAC, and increase ARPU by focusing outreach on high-propensity segments.

Dataset: Health Insurance Cross Sell Prediction (Kaggle)
Source notebook used for reference EDA & baselines: https://www.kaggle.com/code/yashvi/vehicle-insurance-eda-and-boosting-models

----------------------------------------
What’s in this repo
----------------------------------------
- ADS505_Final_Notebook_CrossSell_Team_8.ipynb – main notebook (EDA → features → models → evaluation)
- artifacts/ – saved model(s) produced by the notebook (e.g., .joblib)
- app_vehicle_insurance.py – minimal Streamlit demo for scoring one customer at a time
- data/ (optional) – local copies / samples if you’re not pulling from Kaggle directly
- requirements.txt – Python dependencies
- Previous_Notebooks/ - Notebooks used for testing before moving to final
- README.txt – this file

----------------------------------------
Problem & Approach
----------------------------------------
Business problem: Among current health-insurance customers, who is most likely to buy vehicle insurance if targeted now?
Framing: Binary classification → output a propensity score (0–1).
Workflow: EDA → data cleaning (missing/outliers) → encode categoricals → model comparison with hyperparameter tuning → select and calibrate best model → business evaluation (deciles/lift, thresholds) → Streamlit demo.
Candidate models: Logistic Regression, Random Forest, Gradient Boosting (optional: XGBoost/LightGBM).

----------------------------------------
Setup
----------------------------------------
Python 3.10+ recommended

1) Create a virtual environment and install dependencies
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt

Minimal requirements.txt:
- pandas
- numpy
- scikit-learn
- matplotlib
- joblib
- streamlit
(Optional: xgboost, lightgbm, shap)

----------------------------------------
Quickstart
----------------------------------------
1) Get the data
   Download from Kaggle (“Health Insurance Cross Sell Prediction”). Place CSVs in data/ or update the notebook paths.

2) Run the notebook
   Open ADS505_Final_Notebook_CrossSell_Team_8.ipynb and run all cells. It will:
   - split data (train/test, stratified)
   - tune models via cross-validation
   - pick the winner and optionally calibrate it
   - report ROC AUC / PR AUC / Brier score
   - print decile lift and thresholded confusion matrices
   - save the final model to artifacts/vehicle_insurance_propensity_<MODEL>.joblib

3) Launch the web app
   streamlit run app_vehicle_insurance.py
   Enter feature values in the UI to get a predicted probability and a simple action recommendation (agent call vs programmatic vs suppress).

----------------------------------------
Results
----------------------------------------
- Selected model: Gradient Boosting
- Test ROC AUC: ~0.86
- Test PR AUC: ~0.42 (class-imbalance friendly)
- Top-decile lift: ~3.1× vs portfolio average conversion
Business takeaway: Targeting top deciles yields substantial marketing efficiency; prioritize agent outreach for the top decile, use SMS/email for mid-tier, and suppress bottom decile.

----------------------------------------
Reproducibility
----------------------------------------
- Fixed random_state=42 for splits and CV.
- Dependencies pinned via requirements.txt.
- Model artifacts include the full preprocessing pipeline (scaler + encoder) so predict_proba works on raw feature columns.

----------------------------------------
How we evaluate (for business)
----------------------------------------
- Ranking quality: ROC AUC, PR AUC
- Probability quality: Brier score, isotonic calibration (optional)
- Operating points:
  - F1-optimal threshold (balanced quality)
  - Top-decile threshold (marketing pilot focus)
- Lift analysis: Decile table to estimate conversion lift and ROI impact

----------------------------------------
Using scores in the business
----------------------------------------
- Top 10%: agent outreach / phone; tailored bundle offers
- 20–50%: programmatic email/SMS sequences; reminders around renewal windows
- Bottom deciles: suppress or cheap channels to reduce CAC
- A/B test: compare score-targeted vs BAU; track conversion, CAC, incremental profit

----------------------------------------
Model Summary
----------------------------------------
Intended use: Rank existing health-insurance customers by likelihood to buy vehicle insurance in the near term.
Data: Kaggle cross-sell dataset (customer profile, vehicle, policy).
Metrics: ROC AUC, PR AUC, Brier; business decile lift.
Fairness & ethics:
- Avoid using sensitive attributes directly for targeting decisions.
- Monitor disparate impact across regions/demographics.
- Provide opt-out and respectful contact frequency limits.
- Explainability: expose top drivers (global) and reason codes (local) where feasible.
Limitations: Historic bias, dataset coverage, temporal drift; not a guarantee of purchase, only a ranking signal.

----------------------------------------
Configure / Extend
----------------------------------------
- More models: Explore XGBoost / LightGBM in the notebook and add packages.

----------------------------------------
License & Attribution
----------------------------------------
- Data: Respect Kaggle’s dataset terms and any applicable privacy/compliance requirements.
- Cite: Kaggle dataset https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

----------------------------------------
Contributors
----------------------------------------
- Team: Linden Conrad, Vinh Dao, Jordan Torres

----------------------------------------
FAQ
----------------------------------------
- Can I train without GPUs? Yes; all current models are CPU-friendly.
- My features don’t match the app inputs. Update NUM_COLS and CAT_COLS in the notebook and re-export the artifact.
- Class imbalance? We use PR AUC, calibration, and decile lift to reflect imbalance; you can also try class weights or focal loss (with XGBoost/LightGBM).

