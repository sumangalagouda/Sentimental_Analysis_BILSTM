"""
Text cleaning + encoding — must exactly match what was used during training
in the Colab notebook (packed-sequence version), otherwise predictions will
be inaccurate.
"""
import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)          # remove HTML line breaks
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)             # punctuation -> SPACE (not deleted)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_with_length(text: str, vocab: dict, max_len: int = 200):
    """
    Matches the training-time encode_with_length():
    returns (padded_ids, true_length_before_padding)
    """
    tokens = text.split()[:max_len]
    ids = [vocab.get(tok, vocab.get("<UNK>", 1)) for tok in tokens]

    length = len(ids)
    if length == 0:
        ids = [vocab.get("<UNK>", 1)]
        length = 1

    ids += [vocab.get("<PAD>", 0)] * (max_len - len(ids))
    return ids, length
