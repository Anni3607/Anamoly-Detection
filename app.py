import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

st.set_page_config(layout="wide")

# -------------------------
# LOAD MODEL
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = json.load(open("features.json"))

# -------------------------
# FUNCTIONS
# -------------------------
def compute_score(df_row):
    X = scaler.transform(df_row[features])
    return model.decision_function(X)[0]

def compute_confidence(scores):
    std = np.std(scores)
    return 1 / (1 + std)

def get_ensemble_scores(df_row, n=10):
    scores = []
    for _ in range(n):
        noise = np.random.normal(0, 0.2, len(features))
        temp = df_row.copy()
        temp[features] += noise
        scores.append(compute_score(temp))
    return np.array(scores)

def generate_features(df):
    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["rolling_mean"] = df["value"].rolling(5).mean()
    df["rolling_std"] = df["value"].rolling(5).std()
    df = df.fillna(method="bfill")
    return df

# -------------------------
# UI
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

threshold = np.percentile(df["score"], 10)
df["pred"] = (df["score"] < threshold).astype(int)

anomalies = df[df["pred"] == 1]
anomaly_indices = anomalies.index.tolist()

# -------------------------
# TABS (8)
# -------------------------
tabs = st.tabs([
    "Overview",
    "Data",
    "Trends",
    "Detection",
    "Anomaly Types",
    "Explanation",
    "Feature Impact",
    "Confidence"
])

# =========================
# 1. OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Overview")
    st.write("Basic dataset statistics and anomaly count.")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Anomalies", len(anomalies))

# =========================
# 2. DATA
# =========================
with tabs[1]:
    st.subheader("Dataset Preview")
    st.write("Shows first rows of processed dataset.")

    st.dataframe(df.head(20))

# =========================
# 3. TRENDS
# =========================
with tabs[2]:
    st.subheader("Trend")
    st.write("Displays overall pattern of the time series.")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# 4. DETECTION
# =========================
with tabs[3]:
    st.subheader("Anomaly Detection")
    st.write("Detected anomalies highlighted in red.")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["value"])

    ax.scatter(anomalies.index, anomalies["value"], color="red", s=25)

    st.pyplot(fig)

# =========================
# 5. ANOMALY TYPES
# =========================
with tabs[4]:
    st.subheader("Anomaly Types")
    st.write("Classifies anomalies into spike or drift.")

    df["diff"] = df["value"].diff().fillna(0)
    df["anomaly_type"] = "normal"

    for i in anomaly_indices:
        if abs(df.loc[i, "diff"]) > df["diff"].std()*2:
            df.loc[i, "anomaly_type"] = "spike"
        else:
            df.loc[i, "anomaly_type"] = "drift"

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["value"])

    for t, c in [("spike","red"),("drift","orange")]:
        subset = df[df["anomaly_type"] == t]
        ax.scatter(subset.index, subset["value"], color=c, s=25, label=t)

    ax.legend()
    st.pyplot(fig)

# =========================
# 6. EXPLANATION
# =========================
with tabs[5]:
    st.subheader("Explanation")
    st.write("Shows which feature contributes most to anomaly.")

    if anomaly_indices:
        idx = st.selectbox("Select anomaly", anomaly_indices)

        point = df.loc[[idx]]
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

        st.write(f"Root Cause: {top['feature']}")
        st.write("Adjusting this feature reduces anomaly.")

# =========================
# 7. FEATURE IMPACT
# =========================
with tabs[6]:
    st.subheader("Feature Impact")
    st.write("Relative importance of features in anomaly.")

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

        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(cf_df["feature"], cf_df["impact"])
        st.pyplot(fig)

# =========================
# 8. CONFIDENCE
# =========================
with tabs[7]:
    st.subheader("Confidence")
    st.write("Model reliability based on prediction stability.")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]]

        scores = get_ensemble_scores(point)
        confidence = compute_confidence(scores)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(scores, bins=8)
            st.pyplot(fig)

        with col2:
            st.metric("Confidence", f"{round(confidence*100,2)}%")
