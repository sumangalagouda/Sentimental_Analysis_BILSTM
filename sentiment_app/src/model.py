"""
SentimentLSTM model definition — packed-sequence version.
IMPORTANT: This must exactly match the architecture you trained in Colab,
otherwise the saved weights (.pt file) won't load correctly.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, output_dim=1, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        # Pack so the LSTM ignores padding tokens (matches training exactly)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output.squeeze(1)  # raw logits
