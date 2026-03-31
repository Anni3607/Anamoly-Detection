import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

# =========================
# SAFE LOADING OR FALLBACK
# =========================

def load_or_train():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        st.success("Loaded trained model")

    except Exception as e:
        st.warning("Model load failed → using fallback model")

        scaler = StandardScaler()
        model = IsolationForest()

        return model, scaler, False

    return model, scaler, True

model, scaler, is_loaded = load_or_train()

# =========================
# FEATURES
# =========================

if os.path.exists("features.json"):
    with open("features.json") as f:
        feature_cols = json.load(f)
else:
    feature_cols = ["value", "lag_1", "lag_2", "rolling_mean", "rolling_std"]

# =========================
# TITLE
# =========================

st.title("CAUSAL-XAD Dashboard")

# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

# =========================
# FEATURE ENGINEERING
# =========================

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["rolling_mean"] = df["value"].rolling(5).mean()
df["rolling_std"] = df["value"].rolling(5).std()
df = df.bfill()

# =========================
# TRAIN IF FALLBACK
# =========================

X = df[feature_cols]

if not is_loaded:
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled)
else:
    X_scaled = scaler.transform(X)

# =========================
# PREDICTION
# =========================

df["score"] = model.decision_function(X_scaled)
df["pred"] = model.predict(X_scaled)
df["pred"] = df["pred"].map({1: 0, -1: 1})

# =========================
# ANOMALY TYPES
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
# CONFIDENCE (FIXED)
# =========================

score_std = np.std(df["score"])
confidence = 1 / (1 + score_std)

# =========================
# TABS
# =========================

tabs = st.tabs([
    "Overview",
    "Trend",
    "Detection",
    "Types",
    "Explanation",
    "Impact",
    "Confidence",
    "Simulator"
])

# =========================
# OVERVIEW
# =========================

with tabs[0]:
    st.caption("Dataset summary and anomaly count")

    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Anomalies", int(df["pred"].sum()))

# =========================
# TREND
# =========================

with tabs[1]:
    st.caption("Overall signal pattern")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# DETECTION
# =========================

with tabs[2]:
    st.caption("Detected anomaly points")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])

    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"], s=20)

    st.pyplot(fig)

# =========================
# TYPES
# =========================

with tabs[3]:
    st.caption("Spike vs Drift classification")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])

    for t in ["spike", "drift"]:
        subset = df[df["anomaly_type"] == t]
        ax.scatter(subset.index, subset["value"], label=t, s=20)

    ax.legend()
    st.pyplot(fig)

# =========================
# EXPLANATION
# =========================

with tabs[4]:
    st.caption("Top contributing feature")

    idx = st.number_input("Index", 0, len(df)-1, 0)

    row = df.loc[idx]

    contributions = {
        col: abs(row[col] - df[col].median())
        for col in feature_cols
    }

    top_feature = max(contributions, key=contributions.get)

    st.write("Type:", row["anomaly_type"])
    st.write("Driver:", top_feature)

# =========================
# IMPACT
# =========================

with tabs[5]:
    st.caption("Feature deviation magnitude")

    idx = st.number_input("Index ", 0, len(df)-1, 0)

    row = df.loc[idx]

    impacts = [abs(row[c] - df[c].median()) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(feature_cols, impacts)
    plt.xticks(rotation=30)

    st.pyplot(fig)

# =========================
# CONFIDENCE
# =========================

with tabs[6]:
    st.caption("Model uncertainty estimate")

    st.metric("Confidence", f"{confidence*100:.2f}%")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.hist(df["score"], bins=20)
    st.pyplot(fig)

# =========================
# SIMULATOR
# =========================

with tabs[7]:
    st.caption("Modify values and test outcome")

    idx = st.number_input("Index  ", 0, len(df)-1, 0)

    test_point = df.loc[[idx]].copy()

    for col in feature_cols:
        test_point[col] = st.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(test_point[col])
        )

    scaled = scaler.transform(test_point[feature_cols])
    new_score = model.decision_function(scaled)[0]

    st.metric("Score", round(new_score, 4))

    if new_score > 0:
        st.success("Normal")
    else:
        st.error("Anomaly")
