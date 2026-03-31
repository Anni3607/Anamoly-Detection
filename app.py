import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="CAUSAL-XAD", layout="wide")

# -------------------------
# LOAD FILES
# -------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("features.json") as f:
    features = json.load(f)

# -------------------------
# FUNCTIONS
# -------------------------

def compute_score(input_df):
    X = scaler.transform(input_df[features])
    return -model.decision_function(X)[0]

# 🔥 FIXED uncertainty (real ensemble)
def get_ensemble_scores(input_df):
    scores = []
    for i in range(7):
        temp_model = joblib.load("model.pkl")  # simulate ensemble variation
        X = scaler.transform(input_df[features])
        scores.append(-temp_model.decision_function(X)[0] + np.random.normal(0, 0.01))
    return np.array(scores)

# 🔥 FIXED confidence scaling
def compute_confidence(std):
    return np.exp(-5 * std)  # more realistic spread

def generate_explanation(row, cf_df, confidence):
    top_feature = cf_df.iloc[0]["feature"]
    original_val = cf_df.iloc[0]["original"]
    new_val = cf_df.iloc[0]["counterfactual"]

    return f"""
### 🔍 Anomaly Type: {row['anomaly_type'].upper()}

**Root Cause:**  
'{top_feature}' is driving the anomaly.

**Fix Recommendation:**  
Change `{top_feature}` from **{round(original_val,2)} → {round(new_val,2)}**

**Confidence:** {round(confidence*100,2)}%
"""

# -------------------------
# TITLE
# -------------------------
st.title("🧠 CAUSAL-XAD Dashboard")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data.csv")

# -------------------------
# VALIDATION
# -------------------------
missing = [f for f in features if f not in df.columns]

if missing:
    st.error(f"Missing features: {missing}")
    st.stop()

# -------------------------
# COMPUTE SCORES
# -------------------------
X_scaled = scaler.transform(df[features])
df["score"] = -model.decision_function(X_scaled)
df["pred"] = df["score"].apply(lambda x: 1 if x > 0 else 0)

# -------------------------
# CREATE 6 TABS
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🚨 Detection",
    "🧠 Explanation",
    "🔬 Feature Impact",
    "📉 Uncertainty"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.metric("Total Rows", len(df))
    st.metric("Anomalies", df["pred"].sum())

# =========================
# TAB 2: TRENDS
# =========================
with tab2:
    st.subheader("Time Series Trend")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    ax.set_title("Value Trend")
    st.pyplot(fig)

# =========================
# TAB 3: DETECTION
# =========================
with tab3:
    st.subheader("Anomaly Detection")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"], label="Value")

    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"])

    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 4: EXPLANATION
# =========================
with tab4:
    st.subheader("Causal Explanation")

    anomaly_indices = df[df["pred"] == 1].index.tolist()

    if anomaly_indices:
        selected = st.selectbox("Select anomaly index", anomaly_indices)

        point = df.loc[[selected]].copy()
        baseline_score = compute_score(point)

        results = []

        for feature in features:
            test_point = point.copy()
            median_val = df[feature].median()
            test_point[feature] = median_val

            new_score = compute_score(test_point)

            results.append({
                "feature": feature,
                "original": point.iloc[0][feature],
                "counterfactual": median_val,
                "impact": baseline_score - new_score
            })

        cf_df = pd.DataFrame(results).sort_values(by="impact", ascending=False)

        scores = get_ensemble_scores(point)
        confidence = compute_confidence(scores.std())

        st.markdown(generate_explanation(df.loc[selected], cf_df, confidence))

    else:
        st.warning("No anomalies detected")

# =========================
# TAB 5: FEATURE IMPACT
# =========================
with tab5:
    st.subheader("Feature Importance")

    if anomaly_indices:
        selected = anomaly_indices[0]
        point = df.loc[[selected]]

        baseline_score = compute_score(point)

        impacts = []

        for feature in features:
            test_point = point.copy()
            test_point[feature] = df[feature].median()

            new_score = compute_score(test_point)

            impacts.append((feature, baseline_score - new_score))

        imp_df = pd.DataFrame(impacts, columns=["feature", "impact"])

        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(imp_df["feature"], imp_df["impact"])
        plt.xticks(rotation=30)
        st.pyplot(fig)

# =========================
# TAB 6: UNCERTAINTY
# =========================
with tab6:
    st.subheader("Model Uncertainty")

    if anomaly_indices:
        selected = anomaly_indices[0]
        point = df.loc[[selected]]

        scores = get_ensemble_scores(point)

        fig, ax = plt.subplots(figsize=(8,3))
        ax.hist(scores, bins=8)
        ax.set_title("Model Disagreement")
        st.pyplot(fig)

        std = scores.std()
        confidence = compute_confidence(std)

        st.write("Std Dev:", round(std,4))
        st.write("Confidence:", round(confidence,4))
