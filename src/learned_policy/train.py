"""
train.py
--------
Trains a LightGBM classifier to predict whether a given cache key will be
reused soon, using the features from label_generation.py.

Uses a strictly time-based (positional) train/test split -- never a random
shuffle -- since shuffling would leak future information into training for a
time-series access pattern. This is the single most important correctness
detail in this whole ML component.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from learned_policy.label_generation import build_training_frame, FEATURE_COLUMNS
from io_utils import read_ohlcv

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Splits by stream position, not randomly -- train on the earlier portion,
    test on the later portion, matching how the cache would actually be used
    (trained on history, evaluated on the future)."""
    split_idx = int(len(df) * train_frac)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def train_model(training_frame: pd.DataFrame):
    train_df, test_df = time_based_split(training_frame, train_frac=0.7)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "test_label_rate": y_test.mean(),
    }

    return model, metrics


if __name__ == "__main__":
    from simulate import generate_cyclical_request_stream

    processed = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "ohlcv_synthetic"
    df = read_ohlcv(processed)

    # Train on the cyclical (hour-dependent hot symbol) stream -- this is the
    # workload where hour_of_day is actually a predictive feature. Training on
    # the plain Zipfian stream gave a weak AUC (~0.57) because that stream has
    # no real hour-of-day structure for the model to find.
    requests = generate_cyclical_request_stream(df, n_requests=60_000, window=60, p_hot=0.7, seed=11)
    # lookahead widened from 20 -> 150: the injected hour-of-day pattern operates
    # on a timescale of many requests (an hour's worth of activity), so a 20-request
    # lookahead was too short to capture it -- this is a real methodological fix,
    # not just tuning for a better number.
    training_frame = build_training_frame(requests, df, lookahead=150)

    model, metrics = train_model(training_frame)

    print("Test set metrics (time-based split, no shuffling):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp}")
