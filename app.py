import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

st.set_page_config(layout="wide")

# -------------------------
# LOAD MODEL FILES
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = json.load(open("features.json"))

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def compute_score(df_row):
    X = scaler.transform(df_row[features])
    return model.decision_function(X)[0]

def get_ensemble_scores(df_row, n=10):
    scores = []
    for _ in range(n):
        noise = np.random.normal(0, 0.2, size=len(features))
        temp = df_row.copy()
        temp[features] += noise
        score = compute_score(temp)
        scores.append(score)
    return np.array(scores)

def compute_confidence(std):
    return 1 / (1 + std)

def generate_features(df):
    if "lag_1" not in df.columns:
        df["lag_1"] = df["value"].shift(1)
        df["lag_2"] = df["value"].shift(2)
        df["rolling_mean"] = df["value"].rolling(5).mean()
        df["rolling_std"] = df["value"].rolling(5).std()
        df = df.fillna(method="bfill")
    return df

# -------------------------
# UI HEADER
# -------------------------
st.title("CAUSAL-XAD Dashboard")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data.csv")

df = generate_features(df)

# -------------------------
# MODEL PREDICTION
# -------------------------
X = scaler.transform(df[features])
df["score"] = model.decision_function(X)
df["pred"] = (df["score"] > 0).astype(int)

# -------------------------
# TABS
# -------------------------
tabs = st.tabs([
    "Overview",
    "Trends",
    "Detection",
    "Explanation",
    "Feature Impact",
    "Confidence",
    "Simulator"
])

# =========================
# TAB 1: OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    col1.metric("Total Rows", len(df))
    col2.metric("Anomalies", df["pred"].sum())

    st.dataframe(df.head(20))

# =========================
# TAB 2: TRENDS
# =========================
with tabs[1]:
    st.subheader("Time Series Trend")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    ax.set_title("Value over Time")
    st.pyplot(fig)

    st.info("Shows overall behavior of the signal. Look for sudden jumps or gradual shifts.")

# =========================
# TAB 3: DETECTION
# =========================
with tabs[2]:
    st.subheader("Detected Anomalies")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"], label="Value")

    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"])

    ax.legend()
    st.pyplot(fig)

    st.info("Points marked are detected anomalies.")

# =========================
# TAB 4: EXPLANATION
# =========================
with tabs[3]:
    st.subheader("Causal Explanation")

    anomaly_indices = df[df["pred"] == 1].index.tolist()

    if anomaly_indices:
        idx = st.selectbox("Select anomaly index", anomaly_indices)

        point = df.loc[[idx]].copy()
        baseline = compute_score(point)

        impacts = []

        for f in features:
            temp = point.copy()
            temp[f] = df[f].median()
            new_score = compute_score(temp)

            impacts.append({
                "feature": f,
                "impact": baseline - new_score
            })

        cf_df = pd.DataFrame(impacts).sort_values("impact", ascending=False)

        top = cf_df.iloc[0]

        st.write("Anomaly Type: Detected")
        st.write(f"Root Cause: {top['feature']}")

        st.write(f"Suggested Fix: Adjust {top['feature']} to normal range")

# =========================
# TAB 5: FEATURE IMPACT
# =========================
with tabs[4]:
    st.subheader("Feature Impact")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]]

        impacts = []

        for f in features:
            temp = point.copy()
            temp[f] = df[f].median()
            new_score = compute_score(temp)

            impacts.append({
                "feature": f,
                "impact": compute_score(point) - new_score
            })

        cf_df = pd.DataFrame(impacts)

        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(cf_df["feature"], cf_df["impact"])
        st.pyplot(fig)

        st.info("Higher impact means stronger contribution to anomaly.")

# =========================
# TAB 6: CONFIDENCE
# =========================
with tabs[5]:
    st.subheader("Model Confidence")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]]

        scores = get_ensemble_scores(point)

        std = scores.std()
        confidence = compute_confidence(std)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(scores, bins=8)
            st.pyplot(fig)

        with col2:
            st.metric("Uncertainty (std)", round(std,4))
            st.metric("Confidence", f"{round(confidence*100,2)}%")

            if confidence > 0.85:
                st.success("High confidence")
            elif confidence > 0.6:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence")

# =========================
# TAB 7: SIMULATOR (NEW)
# =========================
with tabs[6]:
    st.subheader("What-If Simulator")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]].copy()

        st.write("Adjust feature values:")

        for f in features:
            val = float(point[f])
            point[f] = st.slider(f, val-20, val+20, val)

        new_score = compute_score(point)

        st.metric("New Score", round(new_score,4))

        if new_score < 0:
            st.success("Anomaly reduced")
        else:
            st.error("Still anomalous")
