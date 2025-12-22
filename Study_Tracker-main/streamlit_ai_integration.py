# ai_model.py (SMART AI SYSTEM)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import IsolationForest
import joblib
import datetime
from pathlib import Path

BASE_DIR = Path.cwd()

# -------- SMALL HELPER FUNCTIONS -------- #

def compute_slope(series):
    """Return slope of last N values."""
    if len(series) < 3:
        return 0
    x = np.arange(len(series))
    return np.polyfit(x, series, 1)[0]

def consistency_score(series):
    """Lower std → more consistency."""
    return round(1 / (1 + np.std(series)), 3)

def fatigue_detector(hours_list):
    """Detect decreasing trend over last 5 sessions."""
    if len(hours_list) < 5:
        return False
    last5 = hours_list[-5:]
    return compute_slope(last5) < -0.15

def generate_ai_advice(sub, avg, slope, cons, anomalies):
    """AI-based text recommendations."""
    advice = []

    if slope > 0.1:
        advice.append("📈 You are improving steadily. Keep the momentum!")
    elif slope < -0.1:
        advice.append("📉 Your recent hours are decreasing. Revisit your schedule.")

    if cons < 0.4:
        advice.append("⚠️ Your study pattern is inconsistent. Fix a routine time.")
    else:
        advice.append("👌 Your consistency is good.")

    if anomalies > 0:
        advice.append("❗Some study sessions were significantly below your usual level.")

    if avg < 1.5:
        advice.append("⏫ Increase at least 30–45 minutes more per session.")
    elif avg > 3:
        advice.append("💪 Great discipline. Maintain the current effort.")

    return " ".join(advice)


# -------- MAIN FUNCTION -------- #

def run_ai_insights(df_user, username):

    st.header("🤖 Smart AI Insights — Deep Study Analysis")

    if df_user.empty:
        st.warning("No study history yet.")
        return

    st.write("Recent entries")
    st.dataframe(df_user.tail(30))

    # -------- CLEAN FEATURE ENGINEERING -------- #

    df = df_user.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["subject","date"])
    df["dow"] = df["date"].dt.dayofweek

    df["cumulative_subject_hours"] = df.groupby("subject")["hours"].cumsum()
    df["prev_hours"] = df.groupby("subject")["hours"].shift(1).fillna(df["hours"].median())
    df["avg_hours_subject"] = df.groupby("subject")["hours"].transform("mean")

    le = LabelEncoder()
    df["subject_enc"] = le.fit_transform(df["subject"])

    # -------- Isolation Forest for Outliers -------- #
    iso = IsolationForest(contamination=0.08, random_state=42)
    df["anomaly"] = iso.fit_predict(df[["hours"]])
    df["is_low_outlier"] = (df["anomaly"] == -1).astype(int)

    # -------- REGRESSION MODELS -------- #

    st.subheader("📊 ML Predictions")

    feature_cols = ["subject_enc","dow","prev_hours","cumulative_subject_hours"]
    Xr = df[feature_cols].fillna(0)
    yr = df["hours"]

    tree_reg = None
    knn_reg = None

    try:
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.25, random_state=42)

        tree_reg = DecisionTreeRegressor(max_depth=6, random_state=42)
        knn_reg = KNeighborsRegressor(n_neighbors=3)

        tree_reg.fit(Xtr, ytr)
        knn_reg.fit(Xtr, ytr)

        preds_tree = tree_reg.predict(Xte)
        preds_knn = knn_reg.predict(Xte)

        st.write("DecisionTree → R²:", round(r2_score(yte, preds_tree),2),
                 "| MSE:", round(mean_squared_error(yte, preds_tree),2))

        st.write("KNN → R²:", round(r2_score(yte, preds_knn),2),
                 "| MSE:", round(mean_squared_error(yte, preds_knn),2))

    except Exception as e:
        st.info("Regression model skipped: " + str(e))

    # -------- SMART PER-SUBJECT INSIGHTS -------- #

    st.subheader("🧠 Smart Subject Insights")

    smart_rows = []

    for sub in df["subject"].unique():

        sub_df = df[df["subject"] == sub]
        hours = list(sub_df["hours"])

        avg = np.mean(hours)
        slope = compute_slope(hours[-7:])
        cons = consistency_score(hours)
        anomalies = sub_df["is_low_outlier"].sum()

        last = sub_df.iloc[-1:]
        feat = last[["subject_enc","dow","prev_hours","cumulative_subject_hours"]].values

        pred = None
        if tree_reg:
            try:
                pred = float(tree_reg.predict(feat)[0])
            except:
                pred = avg

        advice = generate_ai_advice(sub, avg, slope, cons, anomalies)

        smart_rows.append({
            "Subject": sub,
            "Avg Hours": round(avg,2),
            "Trend (slope)": round(slope,3),
            "Consistency Score": cons,
            "Predicted Next Session": round(pred,2) if pred else "—",
            "Outliers": anomalies,
            "AI Recommendation": advice
        })

    st.dataframe(pd.DataFrame(smart_rows))

    # -------- AI 7-DAY STUDY PLAN -------- #

    st.subheader("📅 Your AI-Generated 7-Day Study Plan")

    plan = []
    now = datetime.date.today()

    for i in range(7):
        date = now + datetime.timedelta(days=i)
        best_sub = df.groupby("subject")["hours"].mean().sort_values(ascending=False).index[i % len(df["subject"].unique())]
        plan.append({"Date": date, "Suggested Subject": best_sub})

    st.table(pd.DataFrame(plan))

    # -------- MODEL SAVING -------- #

    st.markdown("---")
    st.subheader("💾 Save Models")

    if st.button("Save trained models"):
        try:
            if tree_reg:
                joblib.dump(tree_reg, BASE_DIR / f"{username}_tree_reg.joblib")
            if knn_reg:
                joblib.dump(knn_reg, BASE_DIR / f"{username}_knn_reg.joblib")
            st.success("Models saved.")
        except Exception as e:
            st.error("Could not save models: " + str(e))
