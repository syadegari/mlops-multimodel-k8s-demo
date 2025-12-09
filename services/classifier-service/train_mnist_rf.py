#!/usr/bin/env python3
"""
Train a RandomForest classifier on MNIST and save a versioned model.

Example (to run it from repo root):

    python services/classifier-service/train_mnist_rf.py \
        --n-estimators 100 \
        --max-depth 10 \
        --version-id mnist_rf_v1
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"].astype(np.float32) / 255.0
    y = mnist["target"].astype(int)
    return X, y


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest on MNIST and save model artifact.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the RandomForest (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum depth of each tree (default: 10).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for test split (default: 0.2).",
    )
    parser.add_argument(
        "--version-id",
        type=str,
        required=True,
        help="Version identifier for the saved model (e.g. mnist_rf_v1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"[train_mnist_rf] Project root: {root_dir}")
    print(f"[train_mnist_rf] Models directory: {models_dir}")

    # Load data
    X, y = load_mnist()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=101,
        stratify=y,
    )
    print(f"[train_mnist_rf] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=101,
        n_jobs=-1,
    )

    print(
        f"[train_mnist_rf] Training RandomForest("
        f"n_estimators={args.n_estimators}, max_depth={args.max_depth})..."
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train_mnist_rf] Test accuracy: {acc:.4f}")
    print("[train_mnist_rf] Classification report (truncated):")
    print(classification_report(y_test, y_pred, digits=3))

    model_path = models_dir / f"{args.version_id}.joblib"
    joblib.dump(clf, model_path)
    print(f"[train_mnist_rf] Saved model to: {model_path}")


if __name__ == "__main__":
    main()

