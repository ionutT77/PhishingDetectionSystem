import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages
import warnings
warnings.filterwarnings('ignore')

import argparse
import json
from pathlib import Path
import numpy as np
from tensorflow import keras

def load_model_and_config(model_dir: Path = Path("models")):
    model_path = model_dir / "url_phishing_model.keras"
    config_path = model_dir / "model_config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Model or config not found in {model_dir}")
    model = keras.models.load_model(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config

def encode_url(url: str, char_to_idx: dict, max_len: int):
    # Ensure url is a string
    url = "" if url is None else str(url)
    seq = [char_to_idx.get(ch, char_to_idx.get("<UNK>", 1)) for ch in url[:max_len]]
    if len(seq) < max_len:
        seq += [char_to_idx.get("<PAD>", 0)] * (max_len - len(seq))
    return np.array(seq, dtype=np.int32)

def predict_urls(model, config, urls):
    char_to_idx = config["char_to_idx"]
    max_len = int(config.get("max_url_length", 200))
    X = np.stack([encode_url(u, char_to_idx, max_len) for u in urls])
    probs = model.predict(X, verbose=0).flatten()
    results = []
    for url, p in zip(urls, probs):
        pred = "PHISHING" if p >= 0.5 else "LEGITIMATE"
        confidence = float(p) if p >= 0.5 else float(1 - p)
        results.append({"url": url, "prediction": pred, "probability": float(p), "confidence": confidence})
    return results

def main():
    parser = argparse.ArgumentParser(description="Predict URL phishing using saved model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", "-u", help="Single URL to classify")
    group.add_argument("--file", "-f", help="File with one URL per line")
    parser.add_argument("--model-dir", "-m", default="models", help="Directory with saved model (default: models)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    try:
        model, config = load_model_and_config(model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.url:
        urls = [args.url.strip()]
    else:
        p = Path(args.file)
        if not p.exists():
            print(f"Input file not found: {p}")
            return
        with open(p, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

    results = predict_urls(model, config, urls)
    for r in results:
        url_short = (r["url"][:120] + "...") if len(r["url"]) > 120 else r["url"]
        print(f"\nURL: {url_short}")
        print(f" -> Prediction: {r['prediction']}")
        print(f" -> Probability (model output): {r['probability']:.4f}")
        print(f" -> Confidence: {r['confidence']:.2%}")

if __name__ == "__main__":
    main()