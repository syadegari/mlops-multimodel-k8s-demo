#!/usr/bin/env python3
"""
Small debug script to query the classifier service with a real MNIST sample.

Example (run from repo root, with the service running on localhost:8000):

    python clients/query_classifier.py --sample-index 0

This will:
  - load MNIST from OpenML (first run might be slow),
  - pick one sample (by index),
  - POST it to http://127.0.0.1:8000/predict,
  - print the true label and predicted label + top probabilities.

NOTE: This is not meant for the HPA scaling.
"""

import argparse
from typing import Tuple

import numpy as np
import requests
from sklearn.datasets import fetch_openml


def load_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST and return (X, y) with X scaled to [0, 1]."""
    print("[query_classifier] Fetching MNIST from OpenML (cached after first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(int)
    print(f"[query_classifier] Loaded MNIST: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def parse_args():
    p = argparse.ArgumentParser(description="Query classifier service with a MNIST sample.")
    p.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the MNIST sample to use (0-based, default: 0).",
    )
    p.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/predict",
        help="URL of the classifier /predict endpoint.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    X, y = load_mnist()

    if not (0 <= args.sample_index < len(X)):
        raise SystemExit(f"sample-index must be in [0, {len(X) - 1}]")

    x = X[args.sample_index]
    true_label = int(y[args.sample_index])

    print(f"[query_classifier] Using sample index {args.sample_index}, true label = {true_label}")

    payload = {
        "features": x.tolist()
    }

    print(f"[query_classifier] Sending request to {args.url} ...")
    resp = requests.post(args.url, json=payload, timeout=10.0)
    print(f"[query_classifier] Status: {resp.status_code}")

    if resp.status_code != 200:
        print("[query_classifier] Error response:", resp.text)
        return

    data = resp.json()
    pred_label = data.get("label")
    probs = data.get("probabilities", [])

    print("\n=== Result ===")
    print(f"True label:      {true_label}")
    print(f"Predicted label: {pred_label}")
    if probs:
        # Show top-3 probabilities
        probs_np = np.array(probs)
        top3 = probs_np.argsort()[::-1][:3]
        print("Top-3 classes by probability (class: prob):")
        for cls in top3:
            print(f"  {cls}: {probs_np[cls]:.3f}")


if __name__ == "__main__":
    main()
