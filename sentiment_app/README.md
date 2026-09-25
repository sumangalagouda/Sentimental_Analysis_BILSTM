# Movie Review Sentiment Analyzer

A multi-page Streamlit app around a PyTorch BiLSTM sentiment model.

## Pages

- `Home.py` - landing page with quick links and usage stats
- `pages/1_Analyze.py` - analyze a review and get a sentiment prediction
- `pages/2_Dashboard.py` - charts, filters, history, and CSV export
- `pages/3_About.py` - model architecture and preprocessing details

## Setup

```bash
pip install -r requirements.txt
```

The included checkpoint is `models/sentiment_model.pt`. It must contain
`model_state` and `vocab`, with optional model configuration metadata.

## Run

```bash
streamlit run Home.py
```

Streamlit automatically discovers the pages in `pages/` and adds them to the
sidebar navigation.
