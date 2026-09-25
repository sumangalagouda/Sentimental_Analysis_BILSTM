"""About page — model architecture, preprocessing, and tech stack info."""
import streamlit as st
from src.ui import load_css, footer

st.set_page_config(page_title="About · Sentiment Analyzer", page_icon="ℹ️", layout="centered")
load_css()

st.markdown(
    """
    <div class="hero">
        <h1>ℹ️ About This Project</h1>
        <p>How the model works, under the hood.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">🧠 Model architecture</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
        <p>The model is a <b>Bidirectional LSTM</b> built in PyTorch:</p>
        <ul>
            <li>Embedding layer (padding-aware)</li>
            <li>2-layer bidirectional LSTM</li>
            <li>Dropout regularization</li>
            <li>Fully-connected output layer to a single logit</li>
        </ul>
        <p>Sequences are packed with <code>pack_padded_sequence</code> so the
        LSTM ignores padding tokens, matching model training.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">🧹 Text preprocessing</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
        <p>Each review follows the training-time cleaning pipeline:</p>
        <ol>
            <li>Lowercase the text</li>
            <li>Strip HTML line breaks and URLs</li>
            <li>Replace punctuation with spaces</li>
            <li>Collapse whitespace</li>
            <li>Tokenize, map to vocabulary IDs, and pad or truncate</li>
        </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">⚙️ Tech stack</div>', unsafe_allow_html=True)
badges = ["Python", "PyTorch", "Streamlit", "SQLite", "Pandas", "Altair"]
st.markdown(
    " ".join(f'<span class="badge badge-purple">{badge}</span>' for badge in badges),
    unsafe_allow_html=True,
)

st.page_link("pages/1_Analyze.py", label="Try the Analyzer", icon="🔍")
footer()
