"""Analyze page — the core sentiment prediction tool."""
import streamlit as st
from src.ui import load_css, footer
from src.predict import load_model, predict_sentiment
from src.database import init_db, insert_prediction

st.set_page_config(page_title="Analyze · Sentiment Analyzer", page_icon="🔍", layout="centered")
load_css()
init_db()


@st.cache_resource
def get_model():
    return load_model("models/best_sentiment_model.pt")


st.markdown(
    """
    <div class="hero">
        <h1>🔍 Analyze a Review</h1>
        <p>Type or paste a movie review below and let the BiLSTM model predict
        its sentiment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vocab, max_len = get_model()
    model_ready = True
except Exception as error:
    model_ready = False
    st.error(
        "Couldn't load the model checkpoint. Make sure "
        "`models/sentiment_model.pt` exists.\n\n"
        f"Details: {error}"
    )

st.markdown('<div class="section-title">💡 Try a sample</div>', unsafe_allow_html=True)
samples = {
    "Glowing review": "This movie completely blew me away — the acting, the score, the cinematography, everything was flawless.",
    "Harsh review": "A complete waste of time. The plot made no sense and the acting was wooden throughout.",
    "Mixed review": "Some great visuals but the pacing dragged and the ending felt rushed and unearned.",
}
sample_cols = st.columns(len(samples))
if "review_text" not in st.session_state:
    st.session_state.review_text = ""
for column, (label, text) in zip(sample_cols, samples.items()):
    if column.button(label, use_container_width=True):
        st.session_state.review_text = text

with st.form("prediction_form"):
    user_input = st.text_area(
        "Write your review here:",
        height=160,
        placeholder="e.g. This movie completely blew me away...",
        key="review_text",
    )
    submitted = st.form_submit_button(
        "Analyze Sentiment", use_container_width=True, disabled=not model_ready
    )

if submitted:
    if not user_input.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence = predict_sentiment(user_input, model, vocab, max_len)
            insert_prediction(user_input, label, confidence)

        if label == "Positive":
            st.markdown(
                '<span class="badge badge-green">● Positive</span> '
                '<span style="opacity:0.7">Prediction result</span>',
                unsafe_allow_html=True,
            )
            st.success(f"**{label}** 🎉")
        else:
            st.markdown(
                '<span class="badge badge-red">● Negative</span> '
                '<span style="opacity:0.7">Prediction result</span>',
                unsafe_allow_html=True,
            )
            st.error(f"**{label}** 😞")

        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(confidence)

st.page_link("pages/2_Dashboard.py", label="See full prediction history →", icon="📊")
footer()
