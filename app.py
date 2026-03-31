import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

# -------------------------
# FEATURE ENGINEERING
# -------------------------
def generate_features(df):
    df = df.copy()

    if "value" not in df.columns:
        st.error("CSV must contain 'value' column")
        st.stop()

    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["rolling_mean"] = df["value"].rolling(5).mean()
    df["rolling_std"] = df["value"].rolling(5).std()

    df = df.fillna(method="bfill")

    return df

# -------------------------
# TRAIN MODEL
# -------------------------
def train_model(df, features):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )

    model.fit(X)

    return model, scaler

# -------------------------
# SCORING
# -------------------------
def compute_score(model, scaler, df_row, features):
    X = scaler.transform(df_row[features])
    return model.decision_function(X)[0]

def get_ensemble_scores(model, scaler, df_row, features, n=10):
    scores = []

    for _ in range(n):
        noise = np.random.normal(0, 0.2, size=len(features))
        temp = df_row.copy()
        temp[features] += noise

        score = compute_score(model, scaler, temp, features)
        scores.append(score)

    return np.array(scores)

def compute_confidence(std):
    return 1 / (1 + std)

# -------------------------
# UI
# -------------------------
st.title("CAUSAL-XAD Dashboard")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using sample synthetic data")

    n = 500
    t = np.arange(n)

    val = 50 + 10*np.sin(t/10) + np.random.normal(0,2,n)
    val[100] += 40
    val[200] -= 30
    val[300:350] += np.linspace(0,30,50)

    df = pd.DataFrame({"value": val})

# -------------------------
# PROCESS DATA
# -------------------------
df = generate_features(df)

features = ["value", "lag_1", "lag_2", "rolling_mean", "rolling_std"]

model, scaler = train_model(df, features)

X = scaler.transform(df[features])
df["score"] = model.decision_function(X)
df["pred"] = (df["score"] > 0).astype(int)

anomaly_indices = df[df["pred"] == 1].index.tolist()

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
# OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Anomalies", df["pred"].sum())

    st.dataframe(df.head(20))

# =========================
# TRENDS
# =========================
with tabs[1]:
    st.subheader("Trend")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# DETECTION
# =========================
with tabs[2]:
    st.subheader("Anomaly Detection")

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])

    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"])

    st.pyplot(fig)

# =========================
# EXPLANATION
# =========================
with tabs[3]:
    st.subheader("Causal Explanation")

    if anomaly_indices:
        idx = st.selectbox("Select anomaly", anomaly_indices)

        point = df.loc[[idx]]
        baseline = compute_score(model, scaler, point, features)

        impacts = []

        for f in features:
            temp = point.copy()
            temp[f] = df[f].median()

            new_score = compute_score(model, scaler, temp, features)

            impacts.append({
                "feature": f,
                "impact": baseline - new_score
            })

        cf_df = pd.DataFrame(impacts).sort_values("impact", ascending=False)

        top = cf_df.iloc[0]

        st.write("Root Cause:", top["feature"])
        st.write("Fix: Adjust feature to normal range")

# =========================
# FEATURE IMPACT
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

            new_score = compute_score(model, scaler, temp, features)

            impacts.append({
                "feature": f,
                "impact": compute_score(model, scaler, point, features) - new_score
            })

        cf_df = pd.DataFrame(impacts)

        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(cf_df["feature"], cf_df["impact"])
        st.pyplot(fig)

# =========================
# CONFIDENCE
# =========================
with tabs[5]:
    st.subheader("Model Confidence")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]]

        scores = get_ensemble_scores(model, scaler, point, features)

        std = scores.std()
        confidence = compute_confidence(std)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(scores, bins=8)
            st.pyplot(fig)

        with col2:
            st.metric("Confidence", f"{round(confidence*100,2)}%")

# =========================
# SIMULATOR
# =========================
with tabs[6]:
    st.subheader("What-if Simulator")

    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]].copy()

        for f in features:
            val = float(point[f])
            point[f] = st.slider(f, val-20, val+20, val)

        new_score = compute_score(model, scaler, point, features)

        st.metric("New Score", round(new_score,4))

        if new_score < 0:
            st.success("Anomaly reduced")
        else:
            st.error("Still anomalous")
