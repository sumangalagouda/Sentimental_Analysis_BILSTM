"""
Streamlit web app for the Sentiment Analysis project.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from src.predict import load_model, predict_sentiment
from src.database import init_db, insert_prediction, get_history, get_stats

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

# ---------- Setup (runs once, cached) ----------
init_db()

@st.cache_resource
def get_model():
    return load_model("models/sentiment_model.pt")

model, vocab, max_len = get_model()

# ---------- Header ----------
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a review and the model (trained BiLSTM, PyTorch) will predict whether it's positive or negative.")

# ---------- Input form ----------
with st.form("prediction_form"):
    user_input = st.text_area("Write your review here:", height=150,
                               placeholder="e.g. This movie completely blew me away...")
    submitted = st.form_submit_button("Analyze Sentiment")

if submitted:
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        label, confidence = predict_sentiment(user_input, model, vocab, max_len)
        insert_prediction(user_input, label, confidence)

        if label == "Positive":
            st.success(f"**Prediction: {label}** 🎉")
        else:
            st.error(f"**Prediction: {label}** 😞")
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

# ---------- Stats ----------
st.divider()
stats = get_stats()
col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", stats["total"])
col2.metric("Positive", stats["positive"])
col3.metric("Negative", stats["negative"])

# ---------- History ----------
st.subheader("Recent Predictions")
history = get_history(limit=20)

if history:
    df = pd.DataFrame(history, columns=["Text", "Prediction", "Confidence", "Timestamp"])
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.1%}")
    df["Text"] = df["Text"].apply(lambda x: x[:80] + "..." if len(x) > 80 else x)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No predictions yet — analyze a review above to see it appear here.")
