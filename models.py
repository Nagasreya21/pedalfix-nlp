"""Text preparation and the two model architectures."""

import re

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch import nn

LABELS = ("brake", "drivetrain", "lock", "tire")
MAX_LENGTH = 160
PAD_ID = 0
UNKNOWN_ID = 1


def normalize(text):
    """Ignore casing and punctuation while preserving typos and word order."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_baseline(analyzer):
    ngrams = (1, 2) if analyzer == "word" else (3, 5)
    return make_pipeline(
        TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, preprocessor=normalize),
        LogisticRegression(max_iter=1000),
    )


def build_vocabulary(texts):
    characters = sorted(set("".join(normalize(text) for text in texts)))
    return {character: index for index, character in enumerate(characters, start=2)}


def encode(texts, vocabulary):
    inputs = torch.full((len(texts), MAX_LENGTH), PAD_ID, dtype=torch.long)
    for index, text in enumerate(texts):
        token_ids = [
            vocabulary.get(character, UNKNOWN_ID)
            for character in normalize(text)[:MAX_LENGTH]
        ]
        inputs[index, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return inputs


class CharacterCNN(nn.Module):
    """Learn short character patterns directly from repair notes."""

    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 24, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(24, 48, kernel_size=size, padding=size // 2)
            for size in (3, 5, 7)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(48 * len(self.convolutions), 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, len(LABELS)),
        )

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).transpose(1, 2)
        valid_positions = token_ids.ne(PAD_ID).unsqueeze(1)
        pooled = []
        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))
            features = features.masked_fill(~valid_positions, 0)
            pooled.append(features.amax(dim=2))
        return self.classifier(torch.cat(pooled, dim=1))


@torch.inference_mode()
def predict_cnn(model, texts, vocabulary):
    model.eval()
    logits = model(encode(texts, vocabulary))
    return [LABELS[index] for index in logits.argmax(dim=1).tolist()]
