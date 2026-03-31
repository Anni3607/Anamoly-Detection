import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os

st.set_page_config(layout="wide")

# =========================
# SAFE MODEL LOADING
# =========================

def load_pickle(path, name):
    if not os.path.exists(path):
        st.error(f"{name} not found")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {name}: {e}")
        st.stop()

model = load_pickle("model.pkl", "model.pkl")
scaler = load_pickle("scaler.pkl", "scaler.pkl")

if not os.path.exists("features.json"):
    st.error("features.json not found")
    st.stop()

with open("features.json") as f:
    feature_cols = json.load(f)

# =========================
# TITLE
# =========================

st.title("CAUSAL-XAD Dashboard")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(uploaded_file)

# =========================
# FEATURE ENGINEERING
# =========================

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["rolling_mean"] = df["value"].rolling(5).mean()
df["rolling_std"] = df["value"].rolling(5).std()
df = df.fillna(method="bfill")

# =========================
# MODEL PREDICTION
# =========================

X = df[feature_cols]
X_scaled = scaler.transform(X)

df["score"] = model.decision_function(X_scaled)
df["pred"] = model.predict(X_scaled)
df["pred"] = df["pred"].map({1: 0, -1: 1})

# =========================
# SIMPLE ANOMALY TYPE
# =========================

df["diff"] = df["value"].diff().fillna(0)

threshold = df["diff"].std() * 2

df["anomaly_type"] = "normal"

for i in range(len(df)):
    if df.loc[i, "pred"] == 1:
        if abs(df.loc[i, "diff"]) > threshold:
            df.loc[i, "anomaly_type"] = "spike"
        else:
            df.loc[i, "anomaly_type"] = "drift"

# =========================
# TABS
# =========================

tabs = st.tabs([
    "Overview",
    "Trend",
    "Detection",
    "Anomaly Types",
    "Explanation",
    "Feature Impact",
    "Confidence",
    "What-if Simulator"
])

# =========================
# TAB 1: OVERVIEW
# =========================

with tabs[0]:
    st.subheader("Dataset Overview")
    st.caption("Basic dataset preview and anomaly count")

    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Detected Anomalies", int(df["pred"].sum()))

# =========================
# TAB 2: TREND
# =========================

with tabs[1]:
    st.subheader("Time Series Trend")
    st.caption("Overall behavior of the signal")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# TAB 3: DETECTION
# =========================

with tabs[2]:
    st.subheader("Detected Anomalies")
    st.caption("Points flagged as anomalies")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"])
    st.pyplot(fig)

# =========================
# TAB 4: ANOMALY TYPES
# =========================

with tabs[3]:
    st.subheader("Anomaly Types")
    st.caption("Simple classification into spike or drift")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])

    for t in ["spike", "drift"]:
        subset = df[df["anomaly_type"] == t]
        ax.scatter(subset.index, subset["value"], label=t)

    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 5: EXPLANATION
# =========================

with tabs[4]:
    st.subheader("Explanation")
    st.caption("Top contributing feature for anomaly")

    idx = st.number_input("Select index", 0, len(df)-1, 0)

    row = df.loc[idx]

    contributions = {}
    for col in feature_cols:
        contributions[col] = abs(row[col] - df[col].median())

    top_feature = max(contributions, key=contributions.get)

    st.write("Anomaly Type:", row["anomaly_type"])
    st.write("Top Feature:", top_feature)

# =========================
# TAB 6: FEATURE IMPACT
# =========================

with tabs[5]:
    st.subheader("Feature Impact")
    st.caption("Relative deviation of features")

    idx = st.number_input("Index", 0, len(df)-1, 0, key="impact")

    row = df.loc[idx]

    impacts = [abs(row[c] - df[c].median()) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(feature_cols, impacts)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# =========================
# TAB 7: CONFIDENCE
# =========================

with tabs[6]:
    st.subheader("Confidence")
    st.caption("Model certainty using score distribution")

    scores = df["score"]
    confidence = 1 / (1 + scores.std())

    st.metric("Confidence", f"{confidence*100:.2f}%")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(scores, bins=20)
    st.pyplot(fig)

# =========================
# TAB 8: WHAT-IF SIMULATOR
# =========================

with tabs[7]:
    st.subheader("What-if Simulator")
    st.caption("Modify inputs and see score change")

    idx = st.number_input("Index", 0, len(df)-1, 0, key="sim")

    test_point = df.loc[[idx]].copy()

    for col in feature_cols:
        test_point[col] = st.slider(col,
                                   float(df[col].min()),
                                   float(df[col].max()),
                                   float(test_point[col]))

    scaled = scaler.transform(test_point[feature_cols])
    new_score = model.decision_function(scaled)[0]

    st.metric("New Score", round(new_score, 4))

    if new_score > 0:
        st.success("Normal")
    else:
        st.error("Anomalous")
