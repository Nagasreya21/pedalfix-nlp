"""Train three classifiers, then evaluate them on the same paired test notes."""

import argparse
import copy
import csv
import json
from pathlib import Path

import joblib
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import DATA
from models import LABELS, CharacterCNN, build_baseline, build_vocabulary, encode, predict_cnn

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
SEED = 42


def load_splits(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return tuple(
        [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    )


def train_cnn(texts, labels, validation, epochs):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    vocabulary = build_vocabulary(texts)
    targets = torch.tensor([LABELS.index(label) for label in labels])
    loader = DataLoader(TensorDataset(encode(texts, vocabulary), targets), batch_size=32, shuffle=True)

    model = CharacterCNN(len(vocabulary) + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    validation_texts = [row["text"] for row in validation]
    validation_labels = [row["label"] for row in validation]
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(inputs)

        predictions = predict_cnn(model, validation_texts, vocabulary)
        score = f1_score(validation_labels, predictions, labels=LABELS, average="macro", zero_division=0)
        print(f"Epoch {epoch:02d} | training loss {running_loss / len(texts):.3f} | validation macro-F1 {score:.3f}")
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    print(f"Selected CNN checkpoint from epoch {best_epoch}.")
    return model, vocabulary, best_epoch


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output, test, predictors):
    truth = [row["label"] for row in test]
    metrics = []
    details = {}
    paired_predictions = [
        {"id": row["id"], "label": row["label"], "clean": row["text"], "noisy": row["noisy_text"]}
        for row in test
    ]

    for name, predict in predictors.items():
        details[name] = {}
        for condition in ("clean", "noisy"):
            predictions = predict([row[condition] for row in paired_predictions])
            accuracy = accuracy_score(truth, predictions)
            macro_f1 = f1_score(truth, predictions, labels=LABELS, average="macro", zero_division=0)
            metrics.append({
                "model": name, "condition": condition, "notes": len(test),
                "accuracy": f"{accuracy:.4f}", "macro_f1": f"{macro_f1:.4f}",
            })
            details[name][condition] = {
                "classification_report": classification_report(
                    truth, predictions, labels=LABELS, output_dict=True, zero_division=0
                ),
                "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist(),
            }
            for row, prediction in zip(paired_predictions, predictions, strict=True):
                row[f"{name}_{condition}"] = prediction
            print(f"{name:<12} {condition:<5} | accuracy {accuracy:.3f} | macro-F1 {macro_f1:.3f}")

    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "predictions.csv", paired_predictions)
    report = {"label_order": LABELS, "models": details}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    train, validation, test = load_splits(DATA)
    print(f"Train: {len(train)} | validation: {len(validation)} | paired test: {len(test)}")
    texts = [row["text"] for row in train]
    labels = [row["label"] for row in train]
    baselines = {
        "word_tfidf": build_baseline("word").fit(texts, labels),
        "char_tfidf": build_baseline("char").fit(texts, labels),
    }
    cnn, vocabulary, best_epoch = train_cnn(texts, labels, validation, args.epochs)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(baselines, output / "baselines.joblib")
    torch.save(
        {"weights": cnn.state_dict(), "vocabulary": vocabulary, "epoch": best_epoch},
        output / "character_cnn.pt",
    )
    predictors = {name: model.predict for name, model in baselines.items()}
    predictors["char_cnn"] = lambda notes: predict_cnn(cnn, notes, vocabulary)
    evaluate(output, test, predictors)
    print(f"Saved models and reports to {output.resolve()}")


if __name__ == "__main__":
    main()
