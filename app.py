import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="Viral Trends BI + Prediction", layout="wide")
st.title("📈 Viral Social Media Trends — BI & Prediction (Clean)")

REQUIRED = [
    "Platform","Hashtag","Content_Type","Region",
    "Views","Likes","Shares","Comments",
    "Engagement_Level","Post_Date"
]

# ---- Sidebar: Upload + Filters ----
st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload CSV (required columns present)", type=["csv"])

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

df["Post_Date"] = pd.to_datetime(df["Post_Date"], errors="coerce")
for c in ["Views","Likes","Shares","Comments"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Engagement_Rate"] = (df["Likes"] + df["Shares"] + df["Comments"]) / df["Views"].replace(0, np.nan)

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

# ---- Tabs to reduce clutter ----
tab_overview, tab_insights, tab_predict = st.tabs(["Overview", "Insights", "Predict"])

with tab_overview:
    st.subheader("KPI Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Views", f"{int(f['Views'].sum()):,}")
    c2.metric("Likes", f"{int(f['Likes'].sum()):,}")
    c3.metric("Shares", f"{int(f['Shares'].sum()):,}")
    c4.metric("Comments", f"{int(f['Comments'].sum()):,}")
    avg_er = f["Engagement_Rate"].mean() * 100 if f["Engagement_Rate"].notna().any() else 0
    c5.metric("Avg Engagement Rate", f"{avg_er:.2f}%")

    st.markdown("**Trend**")
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

with tab_insights:
    st.subheader("Platform vs Content")
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
        colA.plotly_chart(px.bar(by_plat, x="Platform", y="Views", title="Views by platform"),
                          use_container_width=True)
        colB.plotly_chart(px.bar(by_plat, x="Platform", y="Engagement_Rate",
                                 title="Engagement Rate by platform"),
                          use_container_width=True)

        colC, colD = st.columns(2)
        colC.plotly_chart(px.bar(by_ct, x="Content_Type", y="Views", title="Views by content type"),
                          use_container_width=True)
        colD.plotly_chart(px.bar(by_ct, x="Content_Type", y="Engagement_Rate",
                                 title="Engagement Rate by content type"),
                          use_container_width=True)
    else:
        st.info("No data after filters.")

with tab_predict:
    st.subheader("AI: Predict Engagement Level")

    clf_df = df.dropna(subset=["Engagement_Level"]).copy()
    clf_df["Engagement_Rate"] = (clf_df["Likes"] + clf_df["Shares"] + clf_df["Comments"]) / clf_df["Views"].replace(0, np.nan)
    clf_df["Post_Date"] = pd.to_datetime(clf_df["Post_Date"], errors="coerce")
    clf_df["DayOfWeek"] = clf_df["Post_Date"].dt.dayofweek
    clf_df["Month"]     = clf_df["Post_Date"].dt.month

    cat_cols = [c for c in ["Platform","Content_Type","Region"] if c in clf_df.columns]
    num_cols = [c for c in [
        "Views","Likes","Shares","Comments","Engagement_Rate","DayOfWeek","Month"
    ] if c in clf_df.columns]

    X = clf_df[cat_cols + num_cols].copy()
    y = clf_df["Engagement_Level"].astype(str)

    if len(clf_df) <= 100 or X.isna().all(axis=None):
        st.info("Not enough labeled rows for training.")
    else:
        for c in num_cols:
            X[c] = X[c].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols)
            ]
        )

        model = Pipeline(steps=[
            ("prep", pre),
            ("rf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.code(classification_report(y_test, y_pred), language="text")

        st.markdown("**What-if**")
        col1, col2, col3 = st.columns(3)
        p_platform = col1.selectbox("Platform", sorted(df["Platform"].dropna().unique()))
        p_ct       = col2.selectbox("Content Type", sorted(df["Content_Type"].dropna().unique()))
        p_region   = col3.selectbox("Region", sorted(df["Region"].dropna().unique()))
        col4, col5, col6, col7 = st.columns(4)
        p_views    = col4.number_input("Views",    min_value=0, value=int(df["Views"].median()))
        p_likes    = col5.number_input("Likes",    min_value=0, value=int(df["Likes"].median()))
        p_shares   = col6.number_input("Shares",   min_value=0, value=int(df["Shares"].median()))
        p_comments = col7.number_input("Comments", min_value=0, value=int(df["Comments"].median()))

        sample = pd.DataFrame([{
            **({"Platform": p_platform} if "Platform" in X.columns else {}),
            **({"Content_Type": p_ct} if "Content_Type" in X.columns else {}),
            **({"Region": p_region} if "Region" in X.columns else {}),
            **({"Views": p_views} if "Views" in X.columns else {}),
            **({"Likes": p_likes} if "Likes" in X.columns else {}),
            **({"Shares": p_shares} if "Shares" in X.columns else {}),
            **({"Comments": p_comments} if "Comments" in X.columns else {}),
            **({"Engagement_Rate": (p_likes+p_shares+p_comments)/p_views if p_views>0 else 0.0}
               if "Engagement_Rate" in X.columns else {}),
            **({"DayOfWeek": 3} if "DayOfWeek" in X.columns else {}),
            **({"Month": 6} if "Month" in X.columns else {}),
        }])

        if st.button("Predict"):
            pred = model.predict(sample)[0]
            st.success(f"Predicted Engagement Level: **{pred}**")
