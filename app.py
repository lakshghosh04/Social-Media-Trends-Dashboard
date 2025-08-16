import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error

st.set_page_config(page_title="Viral Trends BI + Predictions", layout="wide")
st.title("Viral Social Media Trends — BI & Predictions")

REQUIRED = [
    "Platform","Hashtag","Content_Type","Region",
    "Views","Likes","Shares","Comments",
    "Engagement_Level","Post_Date"
]

# -------------------------
# Data loading
# -------------------------
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload CSV (must include required columns)", type=["csv"])

def load_df():
    if up:
        return pd.read_csv(up, encoding="utf-8", encoding_errors="replace")
    path = "data/Cleaned_Viral_Social_Media_Trends.csv"
    if os.path.exists(path):
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    st.info("Upload the dataset or place it at data/Cleaned_Viral_Social_Media_Trends.csv")
    st.stop()

df = load_df()
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

# -------------------------
# Basic prep
# -------------------------
df["Post_Date"] = pd.to_datetime(df["Post_Date"], errors="coerce")
for c in ["Views","Likes","Shares","Comments"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["Engagement_Rate"] = (df["Likes"] + df["Shares"] + df["Comments"]) / df["Views"].replace(0, np.nan)

# Filters
st.sidebar.header("Filters")
dmin, dmax = df["Post_Date"].min(), df["Post_Date"].max()
date_range = st.sidebar.date_input("Date range", value=(dmin, dmax))
f = df.copy()
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["Post_Date"] >= start) & (f["Post_Date"] <= end)]

def opts(col):
    return ["All"] + sorted(f[col].dropna().astype(str).unique().tolist())

sel_platform = st.sidebar.selectbox("Platform", opts("Platform"))
sel_region   = st.sidebar.selectbox("Region",   opts("Region"))
sel_ctype    = st.sidebar.selectbox("Content Type", opts("Content_Type"))

if sel_platform != "All": f = f[f["Platform"] == sel_platform]
if sel_region   != "All": f = f[f["Region"] == sel_region]
if sel_ctype    != "All": f = f[f["Content_Type"] == sel_ctype]

# Tabs
tab_overview, tab_insights, tab_predict = st.tabs(["Overview", "Insights", "Predict"])

# -------------------------
# Overview
# -------------------------
with tab_overview:
    st.subheader("KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Views", f"{int(f['Views'].sum()):,}")
    c2.metric("Likes", f"{int(f['Likes'].sum()):,}")
    c3.metric("Shares", f"{int(f['Shares'].sum()):,}")
    c4.metric("Comments", f"{int(f['Comments'].sum()):,}")
    avg_er = f["Engagement_Rate"].mean() * 100 if f["Engagement_Rate"].notna().any() else 0
    c5.metric("Avg Engagement Rate", f"{avg_er:.2f}%")

    st.subheader("Trend")
    metric_choice = st.radio("Metric", ["Views", "Engagement_Rate"], horizontal=True)
    daily = f.groupby("Post_Date").agg(
        Views=("Views","sum"),
        Likes=("Likes","sum"),
        Shares=("Shares","sum"),
        Comments=("Comments","sum"),
        Engagement_Rate=("Engagement_Rate","mean")
    ).reset_index()
    if len(daily):
        fig = px.line(daily, x="Post_Date", y=metric_choice, markers=True,
                      title=f"{metric_choice.replace('_',' ')} over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rows after filters.")

# -------------------------
# Insights
# -------------------------
with tab_insights:
    st.subheader("Platform & Content")
    if len(f):
        by_plat = f.groupby("Platform").agg(
            Views=("Views","sum"),
            Engagement_Rate=("Engagement_Rate","mean")
        ).reset_index()
        by_ct = f.groupby("Content_Type").agg(
            Views=("Views","sum"),
            Engagement_Rate=("Engagement_Rate","mean")
        ).reset_index()

        colA, colB = st.columns(2)
        colA.plotly_chart(px.bar(by_plat, x="Platform", y="Views", title="Views by platform"), use_container_width=True)
        colB.plotly_chart(px.bar(by_plat, x="Platform", y="Engagement_Rate", title="Engagement Rate by platform"), use_container_width=True)

        colC, colD = st.columns(2)
        colC.plotly_chart(px.bar(by_ct, x="Content_Type", y="Views", title="Views by content type"), use_container_width=True)
        colD.plotly_chart(px.bar(by_ct, x="Content_Type", y="Engagement_Rate", title="ER by content type"), use_container_width=True)
    else:
        st.info("No data after filters.")

# -------------------------
# Predict
# -------------------------
with tab_predict:
    st.subheader("What-if inputs")
    col1, col2, col3 = st.columns(3)
    p_platform = col1.selectbox("Platform", sorted(df["Platform"].dropna().unique()))
    p_region   = col2.selectbox("Region", sorted(df["Region"].dropna().unique()))
    # user can pick a content type, but we will also recommend one:
    p_ct_user  = col3.selectbox("Content Type (you can pick any)", sorted(df["Content_Type"].dropna().unique()))

    col4, col5, col6, col7 = st.columns(4)
    p_views    = col4.number_input("Views",    min_value=0, value=int(df["Views"].median()))
    p_likes    = col5.number_input("Likes",    min_value=0, value=int(df["Likes"].median()))
    p_shares   = col6.number_input("Shares",   min_value=0, value=int(df["Shares"].median()))
    p_comments = col7.number_input("Comments", min_value=0, value=int(df["Comments"].median()))

    # Prepare base training frame (common features)
    base = df.dropna(subset=["Engagement_Level"]).copy()
    base["Post_Date"] = pd.to_datetime(base["Post_Date"], errors="coerce")
    base["DayOfWeek"] = base["Post_Date"].dt.dayofweek
    base["Month"]     = base["Post_Date"].dt.month
    base["Engagement_Rate"] = (base["Likes"] + base["Shares"] + base["Comments"]) / base["Views"].replace(0, np.nan)

    cat_cols = ["Platform","Content_Type","Region"]
    num_cols = ["Views","Likes","Shares","Comments","Engagement_Rate","DayOfWeek","Month"]

    # Clean numerics
    for c in num_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    # ---------- A) Predict Engagement Level (classification) ----------
    st.subheader("Prediction 1 — Engagement Level (High/Medium/Low)")
    if len(base) > 200:
        X_cls = base[cat_cols + num_cols].copy()
        y_cls = base["Engagement_Level"].astype(str)

        Xtr, Xte, ytr, yte = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

        pre_cls = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols)
            ]
        )
        clf = Pipeline(steps=[
            ("prep", pre_cls),
            ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1))
        ])
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        st.code(classification_report(yte, yhat, zero_division=0), language="text")

        # What-if sample for classification
        er = (p_likes + p_shares + p_comments) / p_views if p_views > 0 else 0.0
        sample_cls = pd.DataFrame([{
            "Platform": p_platform,
            "Content_Type": p_ct_user,
            "Region": p_region,
            "Views": p_views, "Likes": p_likes, "Shares": p_shares, "Comments": p_comments,
            "Engagement_Rate": er, "DayOfWeek": 3, "Month": 6
        }])
        pred_level = clf.predict(sample_cls)[0]
        st.success(f"Predicted Engagement Level: **{pred_level}**")
    else:
        st.info("Not enough labeled rows to train Engagement Level model.")

    # ---------- B) Predict Engagement Rate % (regression) ----------
    st.subheader("Prediction 2 — Engagement Rate (%)")
    # train on rows with ER available
    reg_df = base.dropna(subset=["Engagement_Rate"]).copy()
    if len(reg_df) > 200:
        X_reg = reg_df[cat_cols + num_cols].copy()
        y_reg = reg_df["Engagement_Rate"].astype(float)

        Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        pre_reg = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols)
            ]
        )
        reg = Pipeline(steps=[
            ("prep", pre_reg),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
        ])
        reg.fit(Xtr, ytr)
        yhat_reg = reg.predict(Xte)
        mae = mean_absolute_error(yte, yhat_reg)
        st.write(f"Validation MAE: **{mae:.4f}** (Engagement Rate points)")

        sample_reg = sample_cls.copy()
        pred_er = float(reg.predict(sample_reg)[0]) * 100
        st.success(f"Predicted Engagement Rate: **{pred_er:.2f}%**")
    else:
        st.info("Not enough rows to train Engagement Rate regressor.")

    # ---------- C) Recommend Best Content Type (classification to High) ----------
    st.subheader("Prediction 3 — Recommended Content Type")
    # Binary label: High vs Not-High
    bin_df = base.copy()
    bin_df = bin_df.dropna(subset=["Engagement_Level"])
    bin_df["HighFlag"] = (bin_df["Engagement_Level"].astype(str) == "High").astype(int)

    if len(bin_df) > 200:
        X_bin = bin_df[cat_cols + num_cols].copy()
        y_bin = bin_df["HighFlag"].astype(int)

        Xtr, Xte, ytr, yte = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

        pre_bin = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols)
            ]
        )
        bin_clf = Pipeline(steps=[
            ("prep", pre_bin),
            ("rf", RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1))
        ])
        bin_clf.fit(Xtr, ytr)

        # For recommendation, evaluate all available content types with the same inputs
        candidate_cts = sorted(df["Content_Type"].dropna().unique().tolist())
        rows = []
        for ct in candidate_cts:
            rows.append({
                "Platform": p_platform,
                "Content_Type": ct,
                "Region": p_region,
                "Views": p_views, "Likes": p_likes, "Shares": p_shares, "Comments": p_comments,
                "Engagement_Rate": er, "DayOfWeek": 3, "Month": 6
            })
        rec_frame = pd.DataFrame(rows)
        probs = bin_clf.predict_proba(rec_frame)[:, 1]  # probability of High
        rec_df = pd.DataFrame({"Content_Type": candidate_cts, "P(High)": probs})
        rec_df = rec_df.sort_values("P(High)", ascending=False)
        best_ct = rec_df.iloc[0]["Content_Type"]
        st.success(f"Recommended Content Type: **{best_ct}**")
        st.dataframe(rec_df.head(10), use_container_width=True)
    else:
        st.info("Not enough rows to train Content Type recommender.")

    # ---------- D) Hashtag Suggestions (historical top ER)
    st.subheader("Hashtag Suggestions")
    # Filter historical data to the user's context (platform/region and recommended CT if available)
    ctx = df.copy()
    ctx = ctx[(ctx["Platform"] == p_platform) & (ctx["Region"] == p_region)]
    if "best_ct" in locals():
        ctx = ctx[ctx["Content_Type"] == best_ct]
    elif p_ct_user:
        ctx = ctx[ctx["Content_Type"] == p_ct_user]

    if len(ctx):
        tag_perf = (ctx.groupby("Hashtag")
                       .agg(ER=("Engagement_Rate","mean"), Views=("Views","sum"))
                       .reset_index())
        # prioritize hashtags with non-trivial volume
        tag_perf = tag_perf[tag_perf["Views"] > 0]
        tag_perf = tag_perf.sort_values(["ER","Views"], ascending=[False, False]).head(15)
        st.write("Top hashtags (by Engagement Rate, with volume):")
        st.dataframe(tag_perf[["Hashtag","ER","Views"]].round({"ER":4}), use_container_width=True)
    else:
        st.write("No historical matches for this context. Try different filters.")
