# study_tracker_final.py
# Final integrated Study Tracker with persistent auth, AI Insights (auto + train), Planner, and repo features.

# ---------------------------
# IMPORTS
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sqlite3
import json
import os
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ML
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

# optional auth package
try:
    import streamlit_authenticator as stauth
    HAS_STAUTH = True
except Exception:
    HAS_STAUTH = False

# ---------------------------
# CONFIG / PATHS
# ---------------------------
st.set_page_config(page_title="📚 Study Tracker", layout="wide")
BASE_DIR = Path.cwd()
USERS_DB = BASE_DIR / "users.db"
STUDY_DB = BASE_DIR / "study_data.db"
USER_STATE_DIR = BASE_DIR / "user_states"
SUBJECT_CSV = BASE_DIR / "Subject_Study_Time_Table.csv"

USER_STATE_DIR.mkdir(exist_ok=True)

# ---------------------------
# DB UTIL: Users & Study
# ---------------------------
def init_users_db():
    conn = sqlite3.connect(str(USERS_DB))
    c = conn.cursor()
    # If table already exists with older schema this will not change it; deleting users.db is easiest if mismatch
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    name TEXT,
                    created_at TEXT
                )''')
    conn.commit()
    conn.close()

def register_user_db(username, password, name=""):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(str(USERS_DB))
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                  (username, password_hash, name, datetime.datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return True, "Registered successfully."
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username already exists."

def authenticate_user_db(username, password):
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(str(USERS_DB))
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == password_hash:
        return True
    return False

def list_users_db():
    conn = sqlite3.connect(str(USERS_DB))
    c = conn.cursor()
    c.execute("SELECT username, name FROM users")
    rows = c.fetchall()
    conn.close()
    return rows

def init_study_db():
    conn = sqlite3.connect(str(STUDY_DB))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS study_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    subject TEXT,
                    hours REAL,
                    notes TEXT,
                    date TEXT
                )''')
    conn.commit()
    conn.close()

def save_study_data_db(username, subject, hours, notes, date):
    conn = sqlite3.connect(str(STUDY_DB))
    c = conn.cursor()
    c.execute("INSERT INTO study_log (username, subject, hours, notes, date) VALUES (?, ?, ?, ?, ?)",
              (username, subject, hours, notes, date))
    conn.commit()
    conn.close()

def get_study_data_db(username=None):
    conn = sqlite3.connect(str(STUDY_DB))
    if username:
        df = pd.read_sql_query("SELECT * FROM study_log WHERE username = ?", conn, params=(username,))
    else:
        df = pd.read_sql_query("SELECT * FROM study_log", conn)
    conn.close()
    return df

# ---------------------------
# User state JSON
# ---------------------------
def load_user_state(username):
    path = USER_STATE_DIR / f"state_{username}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"checkboxes": {}, "notes": {}, "last_reset": None}

def save_user_state(username, state):
    path = USER_STATE_DIR / f"state_{username}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ---------------------------
# INIT DB
# ---------------------------
init_users_db()
init_study_db()

# ---------------------------
# Session state defaults (persistence)
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "display_name" not in st.session_state:
    st.session_state["display_name"] = ""

# ---------------------------
# PAGE TITLE
# ---------------------------
st.title("📖 Study Tracker — Full (with AI Insights, Planner & Auth)")

# ---------------------------
# AUTHENTICATION UI (stauth optional)
# ---------------------------
if HAS_STAUTH:
    # Try to create a minimal credentials structure from DB (stauth expects hashed passwords in a special format;
    # full stauth integration requires building a config; we only use stauth login UI if available and credentials built)
    users = list_users_db()
    if users:
        # fallback: present DB-based login if stauth full integration is complex
        st.sidebar.info("STREAMLIT AUTHENTICATOR YET TO IMPLEMENT TILL THEN PROJECT USES DBbasedLogin.")
    # fallback to DB form below
# DB-backed auth form (used if stauth not fully integrated)
if not st.session_state["logged_in"]:
    st.sidebar.header("Account")
    auth_mode = st.sidebar.selectbox("Action", ["Login", "Register"])
    input_username = st.sidebar.text_input("Username")
    input_password = st.sidebar.text_input("Password", type="password")
    input_display = st.sidebar.text_input("Display name (optional)")

    if auth_mode == "Register":
        if st.sidebar.button("Create account"):
            if input_username.strip() == "" or input_password.strip() == "":
                st.sidebar.error("Username & password required.")
            else:
                ok, msg = register_user_db(input_username.strip(), input_password.strip(), input_display.strip())
                if ok:
                    st.sidebar.success(msg + " Now login.")
                else:
                    st.sidebar.error(msg)
            st.rerun()
    else:  # Login
        if st.sidebar.button("Login"):
            if authenticate_user_db(input_username.strip(), input_password.strip()):
                st.session_state["logged_in"] = True
                st.session_state["username"] = input_username.strip()
                st.session_state["display_name"] = input_display.strip() or input_username.strip()
                st.sidebar.success(f"Welcome, {st.session_state['display_name']}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials.")

# Logout button (always available in sidebar once logged in)
if st.session_state["logged_in"]:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

# Stop the app until logged in
if not st.session_state["logged_in"]:
    st.info("Please login or register in the sidebar to continue.")
    st.stop()

# Use shorthand variables
username = st.session_state["username"]
display_name = st.session_state["display_name"]

# ---------------------------
# MAIN NAVIGATION
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Progress", "Backlog", "AI Insights", "Planner", "Account"])

# ---------------------------
# Dashboard: Log hours & checkbox grid
# ---------------------------
if page == "Dashboard":
    st.header("🗓️ Log Your Study Hours")

    # subjects from CSV or typed
    if SUBJECT_CSV.exists():
        subj_df = pd.read_csv(SUBJECT_CSV)
        if "subject" in subj_df.columns:
            subjects = subj_df["subject"].astype(str).unique().tolist()
        elif "Subject" in subj_df.columns:
            subjects = subj_df["Subject"].astype(str).unique().tolist()
        else:
            subjects = subj_df.iloc[:,0].astype(str).unique().tolist()
    else:
        subjects = []

    col1, col2 = st.columns(2)
    with col1:
        if subjects:
            subject = st.selectbox("Subject", options=subjects)
        else:
            subject = st.text_input("Subject (type)")
        hours = st.number_input("Hours Studied", min_value=0.0, step=0.25, value=1.0)
    with col2:
        date = st.date_input("Date", datetime.date.today())
        notes = st.text_area("Notes (optional)", placeholder="What did you study? Quick note...")

    if st.button("Save Entry"):
        save_study_data_db(username, subject, float(hours), notes, str(date))
        st.success("✅ Entry saved.")
        # save to user state
        state = load_user_state(username)
        dkey = str(date)
        state["checkboxes"].setdefault(dkey, {})
        state["checkboxes"][dkey][subject] = True
        if notes:
            state["notes"][dkey] = notes
        save_user_state(username, state)
        st.rerun()

    st.divider()
    st.header("✅ Daily Checkbox Grid (next 14 days)")
    days = [datetime.date.today() + datetime.timedelta(days=i) for i in range(0,14)]
    state = load_user_state(username)
    grid_subjects = subjects if subjects else ["General"]

    for d in days:
        dkey = str(d)
        st.markdown(f"**{dkey}**")
        state["checkboxes"].setdefault(dkey, {})
        rcols = st.columns(len(grid_subjects))
        for i, subj in enumerate(grid_subjects):
            checked = state["checkboxes"][dkey].get(subj, False)
            key = f"chk_{username}_{dkey}_{subj}"
            val = rcols[i].checkbox(subj, value=checked, key=key)
            state["checkboxes"][dkey][subj] = val
    save_user_state(username, state)

    st.divider()
    st.header("📊 Recent Study Log")
    df_user = get_study_data_db(username)
    if not df_user.empty:
        st.dataframe(df_user.sort_values("date", ascending=False).reset_index(drop=True))
        st.line_chart(df_user.groupby("date")["hours"].sum())
    else:
        st.info("No entries yet.")

    if st.button("Reset my progress (checkbox grid + JSON state)"):
        p = USER_STATE_DIR / f"state_{username}.json"
        if p.exists():
            p.unlink()
        st.success("State reset.")
        st.rerun()

# ---------------------------
# Progress: charts & donut
# ---------------------------
elif page == "Progress":
    st.header("📈 Subject-Wise Progress & Daily Completion")
    df_user = get_study_data_db(username)
    if df_user.empty:
        st.warning("No data to show.")
    else:
        subj_totals = df_user.groupby("subject")["hours"].sum().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=subj_totals, x="subject", y="hours", ax=ax1, ci=None)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.pie(subj_totals["hours"], labels=subj_totals["subject"], wedgeprops=dict(width=0.4))
        st.pyplot(fig2)

        st.subheader("Daily Completion Table (last 7 days)")
        state = load_user_state(username)
        last7 = [str(datetime.date.today() - datetime.timedelta(days=i)) for i in range(0,7)]
        table = []
        subjects_in_log = sorted(df_user["subject"].unique()) if not df_user.empty else []
        for d in reversed(last7):
            row = {"date": d}
            boxes = state.get("checkboxes", {}).get(d, {})
            for subj in subjects_in_log:
                row[subj] = "✓" if boxes.get(subj) else ""
            table.append(row)
        st.table(pd.DataFrame(table))

# ---------------------------
# Backlog
# ---------------------------
elif page == "Backlog":
    st.header("⏳ Backlog Tracker")
    df_user = get_study_data_db(username)
    state = load_user_state(username)
    missed = []
    for dkey, subj_map in state.get("checkboxes", {}).items():
        try:
            d_date = datetime.datetime.strptime(dkey, "%Y-%m-%d").date()
        except Exception:
            continue
        if d_date < datetime.date.today():
            for subj, done in subj_map.items():
                if not done:
                    missed.append({"date": dkey, "subject": subj})
    if missed:
        st.table(pd.DataFrame(missed))
    else:
        st.success("No backlog.")

    if missed:
        s = pd.DataFrame(missed).groupby("subject").size().reset_index(name="missed_count")
        for _, r in s.iterrows():
            st.warning(f"Missed {int(r['missed_count'])} sessions of {r['subject']}")

# ---------------------------
# Planner
# ---------------------------
elif page == "Planner":
    st.header("🗺️ Study Planner")
    df_user = get_study_data_db(username)
    subjects = []
    if SUBJECT_CSV.exists():
        subj_df = pd.read_csv(SUBJECT_CSV)
        if "subject" in subj_df.columns:
            subjects = subj_df["subject"].astype(str).unique().tolist()
        elif "Subject" in subj_df.columns:
            subjects = subj_df["Subject"].astype(str).unique().tolist()
    if df_user.empty and not subjects:
        st.warning("No subjects or data. Add subjects or log hours first.")
    else:
        if not subjects:
            subjects = sorted(df_user["subject"].unique())
        target_date = st.date_input("Target completion date", datetime.date.today() + datetime.timedelta(days=14))
        planned_days = (target_date - datetime.date.today()).days
        if planned_days <= 0:
            st.error("Pick a future date.")
        else:
            avg_hours = df_user.groupby("subject")["hours"].sum().to_dict() if not df_user.empty else {}
            target_hours = {}
            for s in subjects:
                default = float(max(2.0, avg_hours.get(s, 0.0) + 5.0))
                target_hours[s] = st.number_input(f"Target hours for {s}", min_value=0.0, value=default, key=f"target_{s}")
            if st.button("Generate plan"):
                plan_rows = []
                for s in subjects:
                    logged = float(avg_hours.get(s, 0.0))
                    target = float(target_hours[s])
                    remaining = max(0.0, target - logged)
                    per_day = remaining / planned_days if planned_days>0 else 0.0
                    for i in range(planned_days):
                        day = datetime.date.today() + datetime.timedelta(days=i)
                        plan_rows.append({"date": str(day), "subject": s, "hours_planned": round(per_day,2)})
                plan_df = pd.DataFrame(plan_rows)
                st.dataframe(plan_df.head(50))
                csv_buf = plan_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download plan CSV", data=csv_buf, file_name="study_plan.csv")

# ---------------------------
# AI Insights (auto predictions + optional Train AI view)
# ---------------------------
# ---------------------------
# AI Insights
# ---------------------------
elif page == "AI Insights":
    from streamlit_ai_integration import run_ai_insights
    df_user = get_study_data_db(username)
    run_ai_insights(df_user, username)


# ---------------------------
# Account: manage & delete
# ---------------------------
elif page == "Account":
    st.header("🔐 Account")
    st.write(f"Logged in as **{username}**")
    if st.button("Refresh my data"):
        st.rerun()
    if st.checkbox("Delete my account & data (irreversible)"):
        st.warning("This will delete your account, study logs, and saved JSON state.")
        if st.button("Confirm delete my account"):
            conn = sqlite3.connect(str(USERS_DB))
            c = conn.cursor()
            c.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
            conn2 = sqlite3.connect(str(STUDY_DB))
            c2 = conn2.cursor()
            c2.execute("DELETE FROM study_log WHERE username = ?", (username,))
            conn2.commit()
            conn2.close()
            path = USER_STATE_DIR / f"state_{username}.json"
            if path.exists():
                path.unlink()
            st.success("Account and data deleted. Refresh to see changes.")
            st.session_state.clear()
            st.rerun()

# ---------------------------
# Fallback
# ---------------------------
else:
    st.info("Choose a page from the sidebar.")
