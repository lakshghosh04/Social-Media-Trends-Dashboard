# app.py — Viral Social Media Trends: BI + Prediction
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Viral Trends BI + Prediction", layout="wide")
st.title("📈 Viral Social Media Trends — BI & Prediction")

REQUIRED = [
    "Platform","Hashtag","Content_Type","Region",
    "Views","Likes","Shares","Comments",
    "Engagement_Level","Post_Date"
]

st.sidebar.header("Upload CSV")
up = st.sidebar.file_uploader("Choose CSV (with required columns)", type=["csv"])

def load_df():
    if up:
        return pd.read_csv(up, encoding="utf-8", encoding_errors="replace")
    local = "data/Cleaned_Viral_Social_Media_Trends.csv"
    if os.path.exists(local):
        return pd.read_csv(local, encoding="utf-8", encoding_errors="replace")
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

st.subheader("KPI Overview")
total_views   = int(f["Views"].sum())
total_likes   = int(f["Likes"].sum())
total_shares  = int(f["Shares"].sum())
total_comments= int(f["Comments"].sum())
avg_er = f["Engagement_Rate"].mean()*100 if f["Engagement_Rate"].notna().any() else 0

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Views", f"{total_views:,}")
k2.metric("Likes", f"{total_likes:,}")
k3.metric("Shares", f"{total_shares:,}")
k4.metric("Comments", f"{total_comments:,}")
k5.metric("Avg Engagement Rate", f"{avg_er:.2f}%")

st.subheader("Trends Over Time")
daily = f.groupby("Post_Date").agg(
    Views=("Views","sum"),
    Likes=("Likes","sum"),
    Shares=("Shares","sum"),
    Comments=("Comments","sum")
).reset_index()
if len(daily):
    daily["Engagement_Rate"] = (daily["Likes"]+daily["Shares"]+daily["Comments"]) / daily["Views"].replace(0, np.nan)
    st.plotly_chart(px.line(daily, x="Post_Date", y="Views", markers=True, title="Views over time"), use_container_width=True)
    st.plotly_chart(px.line(daily, x="Post_Date", y="Engagement_Rate", markers=True, title="Engagement Rate over time"), use_container_width=True)
else:
    st.info("No rows after filters to plot.")

st.subheader("Platform & Content Insights")
if len(f):
    by_plat = f.groupby("Platform").agg(
        Views=("Views","sum"),
        Likes=("Likes","sum"),
        Shares=("Shares","sum"),
        Comments=("Comments","sum")
    ).reset_index()
    by_plat["Engagement_Rate"] = (by_plat["Likes"]+by_plat["Shares"]+by_plat["Comments"]) / by_plat["Views"].replace(0, np.nan)
    cA, cB = st.columns(2)
    cA.plotly_chart(px.bar(by_plat, x="Platform", y="Views", title="Views by platform"), use_container_width=True)
    cB.plotly_chart(px.bar(by_plat, x="Platform", y="Engagement_Rate", title="Engagement Rate by platform"), use_container_width=True)

    by_ct = f.groupby("Content_Type").agg(
        Views=("Views","sum"),
        Likes=("Likes","sum"),
        Shares=("Shares","sum"),
        Comments=("Comments","sum")
    ).reset_index()
    by_ct["Engagement_Rate"] = (by_ct["Likes"]+by_ct["Shares"]+by_ct["Comments"]) / by_ct["Views"].replace(0, np.nan)
    cC, cD = st.columns(2)
    cC.plotly_chart(px.bar(by_ct, x="Content_Type", y="Views", title="Views by content type"), use_container_width=True)
    cD.plotly_chart(px.bar(by_ct, x="Content_Type", y="Engagement_Rate", title="Engagement Rate by content type"), use_container_width=True)

st.subheader("Top Hashtags")
top_ht = (f.groupby("Hashtag")
            .agg(Views=("Views","sum"),
                 Likes=("Likes","sum"),
                 Shares=("Shares","sum"),
                 Comments=("Comments","sum"))
            .reset_index())
if len(top_ht):
    top_ht["Engagement_Rate"] = (top_ht["Likes"]+top_ht["Shares"]+top_ht["Comments"]) / top_ht["Views"].replace(0, np.nan)
    top_ht = top_ht.sort_values("Views", ascending=False).head(20)
    st.plotly_chart(px.bar(top_ht, x="Hashtag", y="Views", title="Top 20 hashtags by views"), use_container_width=True)

st.subheader("AI: Predict Engagement Level")
clf_df = df.dropna(subset=["Engagement_Level"]).copy()
clf_df["Engagement_Rate"] = (clf_df["Likes"] + clf_df["Shares"] + clf_df["Comments"]) / clf_df["Views"].replace(0, np.nan)
clf_df["Like_Rate"]    = clf_df["Likes"]    / clf_df["Views"].replace(0, np.nan)
clf_df["Share_Rate"]   = clf_df["Shares"]   / clf_df["Views"].replace(0, np.nan)
clf_df["Comment_Rate"] = clf_df["Comments"] / clf_df["Views"].replace(0, np.nan)
clf_df["Post_Date"] = pd.to_datetime(clf_df["Post_Date"], errors="coerce")
clf_df["DayOfWeek"] = clf_df["Post_Date"].dt.dayofweek
clf_df["Month"]     = clf_df["Post_Date"].dt.month

cat_cols = [c for c in ["Platform","Content_Type","Region"] if c in clf_df.columns]
num_cols = [c for c in [
    "Views","Likes","Shares","Comments",
    "Engagement_Rate","Like_Rate","Share_Rate","Comment_Rate",
    "DayOfWeek","Month"
] if c in clf_df.columns]

X = clf_df[cat_cols + num_cols].copy()
y = clf_df["Engagement_Level"].astype(str)

if len(clf_df) <= 100 or X.isna().all(axis=None):
    st.info("Not enough clean/labeled rows for training after feature engineering.")
else:
    for c in num_cols: X[c] = X[c].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.code(classification_report(y_test, y_pred), language="text")

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(cm,
        index=[f"Actual_{l}" for l in labels_sorted],
        columns=[f"Pred_{l}" for l in labels_sorted]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**What-if prediction**")
    col1, col2, col3 = st.columns(3)
    p_platform = col1.selectbox("Platform", sorted(df["Platform"].dropna().unique()))
    p_ct       = col2.selectbox("Content Type", sorted(df["Content_Type"].dropna().unique()))
    p_region   = col3.selectbox("Region", sorted(df["Region"].dropna().unique()))
    col4, col5, col6, col7 = st.columns(4)
    p_views    = col4.number_input("Views",    min_value=0, value=int(df["Views"].median()))
    p_likes    = col5.number_input("Likes",    min_value=0, value=int(df["Likes"].median()))
    p_shares   = col6.number_input("Shares",   min_value=0, value=int(df["Shares"].median()))
    p_comments = col7.number_input("Comments", min_value=0, value=int(df["Comments"].median()))

    p_eng_rate = (p_likes + p_shares + p_comments) / p_views if p_views > 0 else 0.0
    p_like_r   = p_likes    / p_views if p_views > 0 else 0.0
    p_share_r  = p_shares   / p_views if p_views > 0 else 0.0
    p_comm_r   = p_comments / p_views if p_views > 0 else 0.0

    sample = pd.DataFrame([{
        **({"Platform": p_platform} if "Platform" in X.columns else {}),
        **({"Content_Type": p_ct} if "Content_Type" in X.columns else {}),
        **({"Region": p_region} if "Region" in X.columns else {}),
        **({"Views": p_views} if "Views" in X.columns else {}),
        **({"Likes": p_likes} if "Likes" in X.columns else {}),
        **({"Shares": p_shares} if "Shares" in X.columns else {}),
        **({"Comments": p_comments} if "Comments" in X.columns else {}),
        **({"Engagement_Rate": p_eng_rate} if "Engagement_Rate" in X.columns else {}),
        **({"Like_Rate": p_like_r} if "Like_Rate" in X.columns else {}),
        **({"Share_Rate": p_share_r} if "Share_Rate" in X.columns else {}),
        **({"Comment_Rate": p_comm_r} if "Comment_Rate" in X.columns else {}),
        **({"DayOfWeek": 3} if "DayOfWeek" in X.columns else {}),
        **({"Month": 6} if "Month" in X.columns else {}),
    }])

    if st.button("Predict Engagement Level"):
        pred = model.predict(sample)[0]
        if hasattr(model.named_steps["rf"], "predict_proba"):
            proba = model.predict_proba(sample)[0]
            proba_df = pd.DataFrame({
                "Class": model.named_steps["rf"].classes_,
                "Probability": np.round(proba, 3)
            }).sort_values("Probability", ascending=False)
            st.success(f"Predicted Engagement Level: **{pred}**")
            st.dataframe(proba_df, use_container_width=True)
        else:
            st.success(f"Predicted Engagement Level: **{pred}**")
