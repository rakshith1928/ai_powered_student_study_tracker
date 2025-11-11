# ai_model.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import datetime
from pathlib import Path

BASE_DIR = Path.cwd()

def run_ai_insights(df_user, username):
    st.header("🤖 AI Insights — Automated Analysis + Trainable View")

    if df_user.empty:
        st.warning("No study history yet.")
        return

    st.write("Recent entries")
    st.dataframe(df_user.tail(30))

    # Feature engineering
    df = df_user.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["subject","date"])
    df["dow"] = df["date"].dt.dayofweek
    df["cumulative_subject_hours"] = df.groupby("subject")["hours"].cumsum()
    df["prev_hours"] = df.groupby("subject")["hours"].shift(1).fillna(df["hours"].median())
    df["avg_hours_subject"] = df.groupby("subject")["hours"].transform("mean")
    le = LabelEncoder()
    df["subject_enc"] = le.fit_transform(df["subject"])

    # AUTOMATIC regression predictions
    st.subheader("Auto predictions (no action required)")
    feature_cols = ["subject_enc","dow","prev_hours","cumulative_subject_hours"]
    Xr = df[feature_cols].fillna(0)
    yr = df["hours"]
    try:
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.25, random_state=42)
        tree_reg = DecisionTreeRegressor(max_depth=6, random_state=42)
        knn_reg = KNeighborsRegressor(n_neighbors=3)
        tree_reg.fit(Xtr, ytr)
        knn_reg.fit(Xtr, ytr)
        preds_tree = tree_reg.predict(Xte)
        preds_knn = knn_reg.predict(Xte)
        st.write("Regression metrics:")
        st.write("Tree R²:", round(r2_score(yte, preds_tree),2), " MSE:", round(mean_squared_error(yte, preds_tree),2))
        st.write("KNN R²:", round(r2_score(yte, preds_knn),2), " MSE:", round(mean_squared_error(yte, preds_knn),2))
    except Exception as e:
        st.info("Automatic regression could not run: " + str(e))
        tree_reg = None
        knn_reg = None

    # Per-subject summary & recommendations
    st.subheader("Per-subject summary & quick recommendations")
    subj_list = df["subject"].unique()
    summary = []

    for s in subj_list:
        sub_df = df[df["subject"] == s]
        last = sub_df.iloc[-1:]
        feat = last[["subject_enc", "dow", "prev_hours", "cumulative_subject_hours"]].fillna(0).values

        avg_hours = sub_df["hours"].mean()
        pred_tree_val = avg_hours
        pred_knn_val = avg_hours

        if tree_reg is not None:
            try:
                pred_tree_val = float(tree_reg.predict(feat)[0])
            except Exception:
                pred_tree_val = avg_hours
        if knn_reg is not None:
            try:
                pred_knn_val = float(knn_reg.predict(feat)[0])
            except Exception:
                pred_knn_val = avg_hours

        rec = "Maintain"
        if avg_hours < sub_df["hours"].median():
            rec = "Increase time slightly"
        elif avg_hours > sub_df["hours"].median():
            rec = "Doing well"

        summary.append({
            "subject": s,
            "avg_hours": round(float(avg_hours), 2),
            "pred_tree": round(float(pred_tree_val), 2),
            "pred_knn": round(float(pred_knn_val), 2),
            "recommendation": rec
        })

    st.dataframe(pd.DataFrame(summary))

    # Optional Train view
    st.markdown("---")
    train_expand = st.expander("🔧 Train AI models and view details (click to expand)")
    with train_expand:
        st.write("This view lets you train small ML models explicitly and inspect metrics.")
        pass_threshold = st.slider("Pass threshold (avg hours)", 1.0, 10.0, float(df["avg_hours_subject"].median()), step=0.5)
        subj_agg = df.groupby("subject").agg({
            "avg_hours_subject":"mean",
            "cumulative_subject_hours":"max",
            "prev_hours":"mean",
            "subject_enc":"first"
        }).reset_index()
        subj_agg["label_pass"] = (subj_agg["avg_hours_subject"] >= pass_threshold).astype(int)
        if len(subj_agg) >= 3:
            Xc = subj_agg[["subject_enc","avg_hours_subject","cumulative_subject_hours","prev_hours"]]
            yc = subj_agg["label_pass"]
            clf_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf_knn = KNeighborsClassifier(n_neighbors=3)
            clf_tree.fit(Xc, yc)
            clf_knn.fit(Xc, yc)
            preds_tree = clf_tree.predict(Xc)
            st.write("Pass/Fail training (on subject-aggregate rows):")
            st.write("DecisionTree accuracy (train):", round(accuracy_score(yc, preds_tree),2))
            st.table(subj_agg[["subject","avg_hours_subject","label_pass"]])
        else:
            st.info("Not enough distinct subjects to train pass/fail classifier.")

        miss_threshold = st.slider("Miss threshold (hours)", 0.25, 3.0, 1.0, step=0.25)
        sess = df.copy()
        sess["label_miss"] = (sess["hours"] < miss_threshold).astype(int)
        feat_cols = ["subject_enc","dow","prev_hours","avg_hours_subject"]
        Xs = sess[feat_cols].fillna(0)
        ys = sess["label_miss"]
        if len(sess) >= 10:
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.25, random_state=42, stratify=ys if len(ys.unique())>1 else None)
            miss_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
            miss_tree.fit(Xtr, ytr)
            pred_mt = miss_tree.predict(Xte)
            st.write("Miss predictor accuracy:", round(accuracy_score(yte, pred_mt),2))
            sess["pred_miss"] = miss_tree.predict(Xs)
            risky = sess[sess["pred_miss"]==1][["date","subject","hours","notes"]].head(10)
            if not risky.empty:
                st.table(risky)
        else:
            st.info("Not enough rows to train miss predictor reliably (need ~10+).")

        if st.button("Save current trained models"):
            try:
                if 'clf_tree' in locals():
                    joblib.dump(clf_tree, BASE_DIR / f"pass_tree_{username}.joblib")
                if 'miss_tree' in locals():
                    joblib.dump(miss_tree, BASE_DIR / f"miss_tree_{username}.joblib")
                st.success("Saved models.")
            except Exception as e:
                st.error("Error saving models: " + str(e))
