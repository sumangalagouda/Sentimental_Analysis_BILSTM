import html
import streamlit as st

from src.ui import load_css, footer
from src.database import init_db, insert_prediction, get_stats
from src.predict import load_model, predict_sentiment


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()
init_db()


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():
    return load_model("models/best_sentiment_model.pt")


try:
    model, vocab, max_len = get_model()
    model_ready = True

except Exception as error:
    model_ready = False
    model = None
    vocab = None
    max_len = 200

    st.error(
        f"Model could not be loaded.\n\n"
        f"Make sure `models/best_sentiment_model.pt` exists.\n\n"
        f"Error: {error}"
    )


# ============================================================
# NAVBAR
# ============================================================

st.markdown(
    '<div class="custom-navbar">'
    '<div class="navbar-brand">'
    '<span class="navbar-logo">🎬</span>'
    '<span>Movie Sentiment Analyzer</span>'
    '</div>'
    '<div class="navbar-links">'
    '<span class="nav-active">Home</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="cinematic-hero">'
    '<div class="hero-content">'
    '<div class="hero-eyebrow">🎥 DEEP LEARNING • IMDb • BiLSTM</div>'
    '<h1>Movie Review<br>'
    '<span>Sentiment Analysis</span></h1>'
    '<p>Enter a movie review and discover whether it expresses '
    'a positive or negative sentiment using a '
    'Bidirectional LSTM deep learning model trained on the IMDb dataset.</p>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ANALYZER TITLE
# ============================================================

st.markdown(
    '<div class="prediction-heading">'
    '<h2>🎬 Analyze Your Review</h2>'
    '<p>What did you think about the movie?</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# REVIEW INPUT
# ============================================================

if "home_review_text" not in st.session_state:
    st.session_state.home_review_text = ""


user_input = st.text_area(
    "Movie review",
    value=st.session_state.home_review_text,
    height=170,
    max_chars=500,
    placeholder=(
        "Example: I absolutely loved this movie. "
        "The acting was fantastic and the story was amazing!"
    ),
    label_visibility="collapsed",
    key="home_review_text",
)


st.markdown(
    f'<div class="character-counter">'
    f'{len(user_input)} / 500 characters'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_clicked = st.button(
    "🔍  Predict Sentiment",
    type="primary",
    use_container_width=True,
    disabled=not model_ready,
)


# ============================================================
# EXAMPLES
# ============================================================

st.markdown(
    '<div class="examples-title">Try some examples</div>',
    unsafe_allow_html=True,
)


samples = {
    "❤️ Loved it":
        "I absolutely loved this movie. "
        "The acting was fantastic and the story was amazing!",

    "😴 Boring":
        "It was boring and nothing interesting "
        "happened during the entire movie.",

    "👍 Not bad":
        "Not bad at all. The acting was decent "
        "and I enjoyed the story.",

    "👎 Waste of time":
        "A complete waste of time. "
        "The acting was terrible and the story made no sense.",
}


sample_columns = st.columns(4)

for index, (label, text) in enumerate(samples.items()):

    with sample_columns[index]:

        if st.button(
            label,
            use_container_width=True,
            key=f"sample_button_{index}",
        ):
            st.session_state.home_review_text = text
            st.rerun()


# ============================================================
# PREDICTION RESULT
# ============================================================

if predict_clicked:

    if not user_input.strip():

        st.warning(
            "Please enter a movie review before predicting."
        )

    elif not model_ready:

        st.error(
            "The sentiment model could not be loaded."
        )

    else:

        with st.spinner("Analyzing your review..."):

            label, confidence = predict_sentiment(
                user_input,
                model,
                vocab,
                max_len,
            )

            insert_prediction(
                user_input,
                label,
                confidence,
            )


        # Safely display review
        quote = html.escape(
            user_input.strip()
        )

        if len(quote) > 150:
            quote = quote[:150] + "..."


        if label == "Positive":

            st.markdown(
                '<div class="sentiment-result positive-result">'
                '<div class="sentiment-icon">😊</div>'
                '<div class="sentiment-content">'
                '<div class="sentiment-label">Positive</div>'
                f'<div class="sentiment-confidence">'
                f'Confidence: {confidence:.2%}'
                f'</div>'
                f'<div class="sentiment-review">'
                f'"{quote}"'
                f'</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="sentiment-result negative-result">'
                '<div class="sentiment-icon">😞</div>'
                '<div class="sentiment-content">'
                '<div class="sentiment-label">Negative</div>'
                f'<div class="sentiment-confidence">'
                f'Confidence: {confidence:.2%}'
                f'</div>'
                f'<div class="sentiment-review">'
                f'"{quote}"'
                f'</div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


        st.markdown(
            '<div class="confidence-title">'
            'Model confidence'
            '</div>',
            unsafe_allow_html=True,
        )

        st.progress(confidence)


# ============================================================
# FEATURES
# ============================================================

st.markdown(
    '<div class="features-section-title">'
    'Why this analyzer?'
    '</div>',
    unsafe_allow_html=True,
)


features = [
    (
        "🎯",
        "Trained on IMDb",
        "Trained using 50,000 movie reviews from the IMDb dataset.",
    ),
    (
        "🧠",
        "Deep Learning Model",
        "Powered by a Bidirectional LSTM neural network built with PyTorch.",
    ),
    (
        "⚡",
        "Instant Results",
        "Get sentiment predictions in real time.",
    ),
    (
        "📈",
        "~83% Accuracy",
        "Achieves approximately 83% accuracy on the IMDb test set.",
    ),
]


feature_columns = st.columns(4)


for index, (icon, title, description) in enumerate(features):

    with feature_columns[index]:

        st.markdown(
            '<div class="feature-card">'
            f'<div class="feature-icon">{icon}</div>'
            f'<h3>{title}</h3>'
            f'<p>{description}</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# ACTIVITY
# ============================================================

st.markdown(
    '<div class="features-section-title">'
    '📊 Activity'
    '</div>',
    unsafe_allow_html=True,
)


stats = get_stats()

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Total Predictions",
        stats["total"],
    )

with c2:
    st.metric(
        "Positive",
        stats["positive"],
    )

with c3:
    st.metric(
        "Negative",
        stats["negative"],
    )


# ============================================================
# FOOTER
# ============================================================

footer()