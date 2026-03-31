import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

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

def get_ensemble_scores(input_df, n_models=5):
    scores = []
    for i in range(n_models):
        m = model
        X = scaler.transform(input_df[features])
        scores.append(-m.decision_function(X)[0])
    return np.array(scores)

def generate_explanation(row, cf_df, confidence):
    top_feature = cf_df.iloc[0]["feature"]
    original_val = cf_df.iloc[0]["original"]
    new_val = cf_df.iloc[0]["counterfactual"]

    return f"""
### 🔍 Anomaly Type: {row['anomaly_type'].upper()}

**Root Cause:**  
Feature `{top_feature}` is driving the anomaly.

**Actionable Fix:**  
Adjust `{top_feature}` from **{round(original_val,2)} → {round(new_val,2)}**

**Confidence:** {round(confidence*100,2)}%
"""

# -------------------------
# UI CONFIG
# -------------------------

st.set_page_config(page_title="CAUSAL-XAD", layout="wide")

st.title("🧠 CAUSAL-XAD: Explainable Anomaly Detection System")

# -------------------------
# FILE UPLOAD
# -------------------------

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data.csv")

# -------------------------
# FEATURE CHECK
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
# TABS
# -------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚨 Anomaly Detection",
    "🧠 Explanation",
    "📈 Uncertainty"
])

# =========================
# TAB 1: OVERVIEW
# =========================

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    fig = plt.figure()
    plt.plot(df["value"])
    plt.title("Value Trend")
    st.pyplot(fig)

# =========================
# TAB 2: ANOMALY DETECTION
# =========================

with tab2:
    st.subheader("Detected Anomalies")

    fig = plt.figure(figsize=(10,4))
    plt.plot(df["value"], label="Value")

    anomalies = df[df["pred"] == 1]
    plt.scatter(anomalies.index, anomalies["value"])

    plt.legend()
    st.pyplot(fig)

    st.write("Total anomalies:", df["pred"].sum())

# =========================
# TAB 3: EXPLANATION
# =========================

with tab3:
    st.subheader("Causal Explanation")

    anomaly_indices = df[df["pred"] == 1].index.tolist()

    if len(anomaly_indices) == 0:
        st.warning("No anomalies found")
    else:
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

        # plot impact
        fig = plt.figure()
        plt.bar(cf_df["feature"], cf_df["impact"])
        plt.xticks(rotation=45)
        plt.title("Feature Impact")
        st.pyplot(fig)

        # uncertainty
        scores = get_ensemble_scores(point)
        confidence = 1 / (1 + scores.std())

        # explanation
        explanation = generate_explanation(df.loc[selected], cf_df, confidence)
        st.markdown(explanation)

# =========================
# TAB 4: UNCERTAINTY
# =========================

with tab4:
    st.subheader("Model Uncertainty")

    if len(df[df["pred"] == 1]) > 0:
        idx = df[df["pred"] == 1].index[0]
        point = df.loc[[idx]]

        scores = get_ensemble_scores(point)

        fig = plt.figure()
        plt.hist(scores, bins=10)
        plt.title("Model Disagreement")
        st.pyplot(fig)

        st.write("Std Dev:", scores.std())
        st.write("Confidence:", 1 / (1 + scores.std()))
