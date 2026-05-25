import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Premium Fraud Detection Portal",
    page_icon="🛡️",
    layout="wide"
)

# ── Custom Visual Theme (Premium Dark-Mode Glassmorphism) ─────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Body Overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Gradient Title */
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #38bdf8 0%, #6366f1 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(22, 28, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 24px;
    }

    .glass-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }

    /* Dynamic Metric Display Tiles */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        margin-bottom: 25px;
    }
    
    .metric-tile {
        flex: 1;
        background: rgba(13, 18, 30, 0.8);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 4px solid #64748b;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-tile:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    }
    
    .metric-tile.fraud {
        border-top-color: #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.15);
    }
    
    .metric-tile.legit {
        border-top-color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }
    
    .metric-tile.warn {
        border-top-color: #f59e0b;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.15);
    }

    .metric-tile-val {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
        letter-spacing: -0.5px;
    }
    
    .metric-tile-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Interactive Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        border: none !important;
        color: white !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s !important;
    }

    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model Pipeline ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load pipeline model file: {e}")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-text">🛡️ Fraud Shield Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">A high-precision real-time transaction auditing suite driven by calibrated XGBoost classifiers and SHAP local explainers.</div>', unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="glass-header">Transaction Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "DEPOSIT"],
        help="Only TRANSFER and CASH_OUT types contain historical instances of fraudulent activities."
    )
    amount = st.number_input("Transaction Amount (£)", min_value=0.01, value=150000.0, step=1000.0, format="%.2f")

with col2:
    st.caption("Account Balances")
    oldbalanceOrg = st.number_input("Origin Account (Old Balance)", min_value=0.0, value=150000.0, step=1000.0, format="%.2f")
    newbalanceOrig = st.number_input("Origin Account (New Balance)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    oldbalanceDest = st.number_input("Destination Account (Old Balance)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    newbalanceDest = st.number_input("Destination Account (New Balance)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")

st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction & Evaluation ──────────────────────────────────────────────────
if st.button("🛡️ Audit Transaction", use_container_width=True):

    # Deriving custom features (exactly as done in the pipeline)
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig
    balanceDiffDest = newbalanceDest - oldbalanceDest
    zeroBalanceOrig = int(oldbalanceOrg > 0 and newbalanceOrig == 0)

    # Creating prediction dataframe matching preprocessing names
    input_data = pd.DataFrame({
        "type":            [transaction_type],
        "amount":          [amount],
        "oldbalanceOrg":   [oldbalanceOrg],
        "newbalanceOrig":  [newbalanceOrig],
        "oldbalanceDest":  [oldbalanceDest],
        "newbalanceDest":  [newbalanceDest],
        "balanceDiffOrig": [balanceDiffOrig],
        "balanceDiffDest": [balanceDiffDest],
        "zeroBalanceOrig": [zeroBalanceOrig],
    })

    # Retrieve uncalibrated probability
    probability = model.predict_proba(input_data)[0][1]

    # Tuned optimal decision threshold for F1-score optimization
    tuned_threshold = 0.99085
    prediction = 1 if probability >= tuned_threshold else 0

    # ── Result Metric Tiles ────────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">Security Audit Summary</div>', unsafe_allow_html=True)

    # Determine risk classes
    if prediction == 1:
        decision_label = "🚨 FRAUD FLAGGED"
        decision_class = "fraud"
        risk_level = "High Risk"
        risk_class = "fraud"
        empirical_note = "87.5% Empirical Precision"
    elif probability >= 0.90:
        decision_label = "⚠️ AUDIT REQUIRED"
        decision_class = "warn"
        risk_level = "Medium Risk"
        risk_class = "warn"
        empirical_note = "0.5% - 50% Empirical Risk"
    else:
        decision_label = "✅ SECURE"
        decision_class = "legit"
        risk_level = "Low Risk"
        risk_class = "legit"
        empirical_note = "<0.01% Empirical Risk"

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-tile {decision_class}">
            <div class="metric-tile-label">Audit Decision</div>
            <div class="metric-tile-val">{decision_label}</div>
        </div>
        <div class="metric-tile {risk_class}">
            <div class="metric-tile-label">Model Suspicion Score</div>
            <div class="metric-tile-val">{probability * 100:.3f}%</div>
        </div>
        <div class="metric-tile {risk_class}">
            <div class="metric-tile-label">Empirical Risk Class</div>
            <div class="metric-tile-val">{risk_level}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Actionable Callouts
    if prediction == 1:
        st.error(f"🚨 **Critical Alert**: This transaction is flagged as high-risk. The model suspicion score ({probability*100:.2f}%) exceeds the optimal security threshold of {tuned_threshold*100:.3f}%. Financial networks are advised to hold this transfer.")
    elif probability >= 0.90:
        st.warning(f"⚠️ **Attention**: Transaction has elevated suspicion characteristics. While it falls below the direct classification threshold, empirical verification is recommended for secondary verification.")
    else:
        st.success(f"✅ **Clear**: Transaction exhibits low fraud-characteristic behavior. Allowed to execute.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Explainability (SHAP Values) ───────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-header">🔎 Local SHAP Attribution Explainer</div>', unsafe_allow_html=True)

    try:
        xgb_model = model.named_steps["clf"]
        preprocessor = model.named_steps["prep"]
        X_transformed = preprocessor.transform(input_data)

        # Retrieve feature name order directly from ColumnTransformer steps
        numerical_scaled = ["amount", "oldbalanceOrg", "newbalanceOrig",
                            "oldbalanceDest", "newbalanceDest",
                            "balanceDiffOrig", "balanceDiffDest"]
        cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(["type"]).tolist()
        numerical_passthrough = ["zeroBalanceOrig"]
        all_feature_names = numerical_scaled + cat_features + numerical_passthrough

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        # Reconstruct unscaled feature values for explanations (improving business readability)
        unscaled_values = {
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest,
            "balanceDiffOrig": balanceDiffOrig,
            "balanceDiffDest": balanceDiffDest,
            "zeroBalanceOrig": zeroBalanceOrig,
            "type_CASH_OUT": 1 if transaction_type == "CASH_OUT" else 0,
            "type_DEBIT": 1 if transaction_type == "DEBIT" else 0,
            "type_PAYMENT": 1 if transaction_type == "PAYMENT" else 0,
            "type_TRANSFER": 1 if transaction_type == "TRANSFER" else 0,
        }
        table_values = [unscaled_values.get(feat, X_transformed[0][i]) for i, feat in enumerate(all_feature_names)]

        # Render Waterfall Plot
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_transformed[0],
            feature_names=all_feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 4.5))
        # Style matplotlib output to match dark environment
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')
        for text in ax.texts:
            text.set_color('#f1f5f9')
            
        shap.waterfall_plot(explanation, show=False, max_display=8)
        
        # Color tweaks for matplotlib text objects in SHAP plot
        plt.gcf().patch.set_facecolor('#0f172a')
        for text in plt.gcf().findobj(match=plt.Text):
            text.set_fontfamily('sans-serif')
            text.set_color('#e2e8f0')

        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()

        # Feature contribution table
        shap_df = pd.DataFrame({
            "Feature Name": all_feature_names,
            "Observed Value": table_values,
            "SHAP Influence Score": shap_values[0]
        }).sort_values("SHAP Influence Score", key=abs, ascending=False).head(8)

        # Style and render
        st.markdown("<p style='font-size:0.95rem; color:#94a3b8;'>Top feature attributions driving this decision (Positive score = pushes toward fraud, Negative = pulls toward legitimate):</p>", unsafe_allow_html=True)
        
        # Format the display DataFrame
        formatted_df = shap_df.copy()
        formatted_df["Observed Value"] = formatted_df.apply(
            lambda r: f"£{r['Observed Value']:,.2f}" if r["Feature Name"] in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "balanceDiffOrig", "balanceDiffDest"]
            else ("Yes" if r["Observed Value"] == 1 else "No" if r["Observed Value"] == 0 else str(r["Observed Value"])),
            axis=1
        )
        
        st.dataframe(
            formatted_df.style.format({"SHAP Influence Score": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.info(f"SHAP explanation rendering was bypassed for this transaction. (Error: {e})")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Computed Features ──────────────────────────────────────────────────────
    with st.expander("📊 Audit Details & Feature Computations"):
        feat_df = pd.DataFrame({
            "Feature Engineered": ["balanceDiffOrig", "balanceDiffDest", "zeroBalanceOrig"],
            "Computed Value": [f"£{balanceDiffOrig:,.2f}", f"£{balanceDiffDest:,.2f}", "Yes" if zeroBalanceOrig == 1 else "No"],
            "Operational Significance": [
                "Total sender balance reduction (Should match amount in honest transactions).",
                "Total receiver balance increase (Should match amount in honest transactions).",
                "Flag checks if the sender drained their account entirely to 0.00 (Highly associated with fraud)."
            ]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
    Model Engine: XGBoost Classifier (scale_pos_weight = 773.7) | Auditing Threshold: 0.99085 (Optimized F1) | Explainer: SHAP TreeExplainer
</div>
""", unsafe_allow_html=True)
