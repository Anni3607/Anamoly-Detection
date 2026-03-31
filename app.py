import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

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
# BASIC VALIDATION
# =========================

if "value" not in df.columns:
    st.error("CSV must contain 'value' column")
    st.stop()

# =========================
# FEATURE ENGINEERING
# =========================

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["rolling_mean"] = df["value"].rolling(5).mean()
df["rolling_std"] = df["value"].rolling(5).std()

df = df.bfill()

feature_cols = ["value", "lag_1", "lag_2", "rolling_mean", "rolling_std"]

# =========================
# MODEL (DYNAMIC)
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X_scaled)

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
# TAB 1: OVERVIEW
# =========================

with tabs[0]:
    st.caption("Dataset summary and anomaly count")

    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Anomalies", int(df["pred"].sum()))

# =========================
# TAB 2: TREND
# =========================

with tabs[1]:
    st.caption("Overall time-series pattern")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])
    st.pyplot(fig)

# =========================
# TAB 3: DETECTION
# =========================

with tabs[2]:
    st.caption("Detected anomaly points")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])

    anomalies = df[df["pred"] == 1]
    ax.scatter(anomalies.index, anomalies["value"], s=20)

    st.pyplot(fig)

# =========================
# TAB 4: TYPES
# =========================

with tabs[3]:
    st.caption("Spike vs drift classification")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["value"])

    for t in ["spike", "drift"]:
        subset = df[df["anomaly_type"] == t]
        ax.scatter(subset.index, subset["value"], label=t, s=20)

    ax.legend()
    st.pyplot(fig)

# =========================
# TAB 5: EXPLANATION
# =========================

with tabs[4]:
    st.caption("Most deviating feature (proxy explanation)")

    idx = st.number_input("Select index", 0, len(df)-1, 0)

    row = df.loc[idx]

    contributions = {
        col: abs(row[col] - df[col].median())
        for col in feature_cols
    }

    top_feature = max(contributions, key=contributions.get)

    st.write("Anomaly Type:", row["anomaly_type"])
    st.write("Main Driver:", top_feature)

# =========================
# TAB 6: IMPACT
# =========================

with tabs[5]:
    st.caption("Feature deviation magnitude")

    idx = st.number_input("Index", 0, len(df)-1, 0, key="impact")

    row = df.loc[idx]

    impacts = [abs(row[c] - df[c].median()) for c in feature_cols]

    fig, ax = plt.subplots(figsize=(7,3))
    ax.bar(feature_cols, impacts)
    plt.xticks(rotation=30)

    st.pyplot(fig)

# =========================
# TAB 7: CONFIDENCE
# =========================

with tabs[6]:
    st.caption("Model uncertainty (lower variance = higher confidence)")

    st.metric("Confidence", f"{confidence*100:.2f}%")

    fig, ax = plt.subplots(figsize=(7,3))
    ax.hist(df["score"], bins=20)

    st.pyplot(fig)

# =========================
# TAB 8: SIMULATOR
# =========================

with tabs[7]:
    st.caption("Modify inputs and observe score change")

    idx = st.number_input("Index ", 0, len(df)-1, 0)

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

    st.metric("New Score", round(new_score, 4))

    if new_score > 0:
        st.success("Normal")
    else:
        st.error("Anomalous")
