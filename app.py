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

    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["rolling_mean"] = df["value"].rolling(5).mean()
    df["rolling_std"] = df["value"].rolling(5).std()

    df = df.fillna(method="bfill")

    return df

# -------------------------
# MODEL TRAINING
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
# SCORE
# -------------------------
def compute_score(model, scaler, df_row, features):
    X = scaler.transform(df_row[features])
    return model.decision_function(X)[0]

# -------------------------
# CONDITIONAL COUNTERFACTUAL (NEW)
# -------------------------
def apply_structural_update(df, idx, new_value):
    temp_df = df.copy()

    temp_df.loc[idx, "value"] = new_value

    # update dependent features
    temp_df = generate_features(temp_df)

    return temp_df.loc[[idx]]

# -------------------------
# UI
# -------------------------
st.title("CAUSAL-XAD Dashboard")

uploaded = st.file_uploader("Upload CSV")

if uploaded:
    df = pd.read_csv(uploaded)
else:
    n = 500
    t = np.arange(n)

    val = 50 + 10*np.sin(t/10) + np.random.normal(0,2,n)
    val[100] += 40
    val[200] -= 30
    val[300:350] += np.linspace(0,30,50)

    df = pd.DataFrame({"value": val})

# -------------------------
# PROCESS
# -------------------------
df = generate_features(df)

features = ["value", "lag_1", "lag_2", "rolling_mean", "rolling_std"]

model, scaler = train_model(df, features)

X = scaler.transform(df[features])
df["score"] = model.decision_function(X)

# ✅ CORRECT THRESHOLD
threshold = np.percentile(df["score"], 10)
df["pred"] = (df["score"] < threshold).astype(int)

anomaly_indices = df[df["pred"] == 1].index.tolist()

# -------------------------
# TABS
# -------------------------
tabs = st.tabs([
    "Overview",
    "Trends",
    "Detection",
    "Causal Explanation",
    "Feature Impact",
    "Confidence",
    "Simulator"
])

# =========================
# OVERVIEW
# =========================
with tabs[0]:
    st.subheader("Overview")

    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Anomalies", df["pred"].sum())

    st.dataframe(df.head(20))

# =========================
# TRENDS
# =========================
with tabs[1]:
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# DETECTION
# =========================
with tabs[2]:
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df["value"])

    anomalies = df[df["pred"] == 1]

    ax.scatter(anomalies.index, anomalies["value"], color="red", s=30)

    st.pyplot(fig)

# =========================
# CAUSAL EXPLANATION
# =========================
with tabs[3]:
    st.subheader("Causal Explanation")

    if anomaly_indices:
        idx = st.selectbox("Select anomaly", anomaly_indices)

        original = df.loc[[idx]]
        baseline = compute_score(model, scaler, original, features)

        results = []

        for f in ["value"]:  # focus on core driver

            median_val = df[f].median()

            cf_point = apply_structural_update(df, idx, median_val)

            new_score = compute_score(model, scaler, cf_point, features)

            results.append({
                "feature": f,
                "impact": baseline - new_score,
                "new_score": new_score
            })

        cf_df = pd.DataFrame(results).sort_values("impact", ascending=False)

        top = cf_df.iloc[0]

        st.write(f"Root Cause: {top['feature']}")
        st.write(f"If {top['feature']} moves toward normal, anomaly reduces")

# =========================
# FEATURE IMPACT
# =========================
with tabs[4]:
    if anomaly_indices:
        idx = anomaly_indices[0]

        impacts = []

        for f in features:
            temp = df.loc[[idx]].copy()
            temp[f] = df[f].median()

            new_score = compute_score(model, scaler, temp, features)

            impacts.append({
                "feature": f,
                "impact": compute_score(model, scaler, df.loc[[idx]], features) - new_score
            })

        cf_df = pd.DataFrame(impacts)

        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(cf_df["feature"], cf_df["impact"])
        st.pyplot(fig)

# =========================
# CONFIDENCE
# =========================
with tabs[5]:
    if anomaly_indices:
        idx = anomaly_indices[0]
        point = df.loc[[idx]]

        scores = []

        for _ in range(10):
            noise = np.random.normal(0, 0.2, len(features))
            temp = point.copy()
            temp[features] += noise

            scores.append(compute_score(model, scaler, temp, features))

        scores = np.array(scores)

        std = scores.std()
        confidence = 1 / (1 + std)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5,3))
            ax.hist(scores, bins=8)
            st.pyplot(fig)

        with col2:
            st.metric("Confidence", f"{round(confidence*100,2)}%")

# =========================
# SIMULATOR (FIXED)
# =========================
with tabs[6]:
    st.subheader("What-if Simulator")

    if anomaly_indices:
        idx = anomaly_indices[0]

        val = float(df.loc[idx, "value"])

        new_val = st.slider(
            "value",
            float(df["value"].min()),
            float(df["value"].max()),
            val
        )

        cf_point = apply_structural_update(df, idx, new_val)

        new_score = compute_score(model, scaler, cf_point, features)

        st.metric("New Score", round(new_score,4))

        if new_score < threshold:
            st.error("Still anomalous")
        else:
            st.success("Anomaly reduced")
