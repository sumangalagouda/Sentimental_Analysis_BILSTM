"""Dashboard page — stats, charts, and full prediction history."""
import altair as alt
import pandas as pd
import streamlit as st
from src.ui import load_css, footer
from src.database import init_db, get_history, get_stats

st.set_page_config(page_title="Dashboard · Sentiment Analyzer", page_icon="📊", layout="centered")
load_css()
init_db()

st.markdown(
    """
    <div class="hero">
        <h1>📊 Dashboard</h1>
        <p>An overview of every prediction made so far — trends, breakdowns,
        and the full history.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = get_stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total Predictions", stats["total"])
c2.metric("Positive", stats["positive"])
c3.metric("Negative", stats["negative"])

history = get_history(limit=500)
if not history:
    st.info("No predictions yet — head to the **Analyze** page to get started.")
    st.page_link("pages/1_Analyze.py", label="Go to Analyzer", icon="🔍")
    footer()
    st.stop()

df = pd.DataFrame(history, columns=["Text", "Prediction", "Confidence", "Timestamp"])
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

st.markdown('<div class="section-title">📈 Breakdown</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 1.4])
with col_a:
    counts = df["Prediction"].value_counts().reset_index()
    counts.columns = ["Prediction", "Count"]
    pie = (
        alt.Chart(counts).mark_arc(innerRadius=55).encode(
            theta="Count",
            color=alt.Color(
                "Prediction",
                scale=alt.Scale(domain=["Positive", "Negative"], range=["#2ed573", "#ff4757"]),
            ),
            tooltip=["Prediction", "Count"],
        ).properties(height=240)
    )
    st.altair_chart(pie, use_container_width=True)
with col_b:
    daily = (
        df.assign(Day=df["Timestamp"].dt.date)
        .groupby(["Day", "Prediction"])
        .size()
        .reset_index(name="Count")
    )
    line = (
        alt.Chart(daily).mark_line(point=True).encode(
            x="Day:T",
            y="Count:Q",
            color=alt.Color(
                "Prediction",
                scale=alt.Scale(domain=["Positive", "Negative"], range=["#2ed573", "#ff4757"]),
            ),
            tooltip=["Day", "Prediction", "Count"],
        ).properties(height=240)
    )
    st.altair_chart(line, use_container_width=True)

st.markdown('<div class="section-title">🗂️ Full history</div>', unsafe_allow_html=True)
filter_col, search_col = st.columns([1, 2])
sentiment_filter = filter_col.selectbox("Filter", ["All", "Positive", "Negative"])
search_term = search_col.text_input("Search text", placeholder="Search reviews...")

filtered = df.copy()
if sentiment_filter != "All":
    filtered = filtered[filtered["Prediction"] == sentiment_filter]
if search_term:
    filtered = filtered[filtered["Text"].str.contains(search_term, case=False, na=False)]

display_df = filtered.copy()
display_df["Confidence"] = display_df["Confidence"].apply(lambda value: f"{value:.1%}")
display_df["Text"] = display_df["Text"].apply(lambda value: value[:80] + "..." if len(value) > 80 else value)
st.dataframe(display_df, use_container_width=True, hide_index=True)
st.caption(f"Showing {len(filtered)} of {len(df)} predictions")
st.download_button(
    "⬇️ Download full history as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_history.csv",
    mime="text/csv",
    use_container_width=True,
)

footer()
