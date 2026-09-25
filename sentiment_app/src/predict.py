"""
Loads the trained model + vocabulary from the .pt checkpoint and exposes
a simple predict_sentiment() function used by the Streamlit app.

Matches the packed-sequence training checkpoint, which stores:
  model_state, vocab, max_len, embed_dim, hidden_dim, n_layers, dropout
"""
import torch
from src.model import SentimentLSTM
from src.preprocessing import clean_text, encode_with_length

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str = "models/best_sentiment_model.pt"):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    vocab = checkpoint["vocab"]

    model = SentimentLSTM(
        vocab_size=len(vocab),
        embed_dim=checkpoint.get("embed_dim", 128),
        hidden_dim=checkpoint.get("hidden_dim", 128),
        n_layers=checkpoint.get("n_layers", 2),
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    max_len = checkpoint.get("max_len", 200)
    return model, vocab, max_len


def predict_sentiment(text: str, model, vocab: dict, max_len: int = 200):
    cleaned = clean_text(text)
    ids, length = encode_with_length(cleaned, vocab, max_len)

    x = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([length], dtype=torch.long)

    with torch.no_grad():
        prob = torch.sigmoid(model(x, lengths)).item()

    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    return label, confidence


if __name__ == "__main__":
    # Quick manual test: python -m src.predict
    model, vocab, max_len = load_model()
    sample = "This movie was absolutely fantastic, I loved every minute!"
    label, conf = predict_sentiment(sample, model, vocab, max_len)
    print(f"'{sample}' -> {label} ({conf:.2%} confidence)")
