"""Compare the three trained models on one new maintenance note."""

import argparse
from pathlib import Path

import joblib
import torch

from models import CharacterCNN, normalize, predict_cnn
from train import DEFAULT_OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("note", help="A single English maintenance note")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not normalize(args.note):
        parser.error("provide a note containing English letters or digits")

    baselines_path = args.models_dir / "baselines.joblib"
    cnn_path = args.models_dir / "character_cnn.pt"
    if not baselines_path.is_file() or not cnn_path.is_file():
        parser.error("model files are missing. Run python train.py first.")

    # Load only the model files you created by running train.py.
    baselines = joblib.load(baselines_path)
    saved = torch.load(cnn_path, map_location="cpu", weights_only=True)
    vocabulary = saved["vocabulary"]
    cnn = CharacterCNN(len(vocabulary) + 2)
    cnn.load_state_dict(saved["weights"])

    for name, model in baselines.items():
        print(f"{name:<12} {model.predict([args.note])[0]}")
    print(f"{'char_cnn':<12} {predict_cnn(cnn, [args.note], vocabulary)[0]}")


if __name__ == "__main__":
    main()
