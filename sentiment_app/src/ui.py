"""Shared UI helpers for the multipage Streamlit app."""
from pathlib import Path
import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_css():
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def footer():
    st.markdown(
        '<div class="footer-note">Built with Streamlit + PyTorch (BiLSTM) · '
        'Sentiment Analyzer Project</div>',
        unsafe_allow_html=True,
    )
