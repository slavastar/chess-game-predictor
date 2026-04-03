"""
Step 3: Train and evaluate models for chess outcome prediction.

Models:
  - Logistic Regression (with feature normalization)
  - Decision Tree / Random Forest (no normalization needed)

Metrics:
  - Log loss (cross-entropy): how well-calibrated are predicted probabilities
  - Accuracy: overall correctness
  - Macro F1: average F1 across all 3 classes (handles class imbalance)
  - Per-class precision / recall / F1
  - Confusion matrix

Usage:
  python scripts/train_model.py                    # default: logistic regression
  python scripts/train_model.py --model tree       # decision tree
  python scripts/train_model.py --model forest     # random forest
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATASET_DIR = Path(__file__).parent.parent / "data" / "dataset"

FEATURE_COLS = [
    "rating_diff",
    "round",
    "win_rate_diff",
    "draw_rate_diff",
    "total_games_diff",
    "best_rating_diff",
]

TARGET_COL = "outcome"


def load_data():
    """Load train and test CSVs."""
    train_df = pd.read_csv(DATASET_DIR / "train.csv")
    test_df = pd.read_csv(DATASET_DIR / "test.csv")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test


def build_model(model_type: str) -> Pipeline:
    """Build a sklearn Pipeline for the chosen model type."""
    if model_type == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ])
    elif model_type == "tree":
        return Pipeline([
            ("model", DecisionTreeClassifier(
                random_state=42,
            )),
        ])
    elif model_type == "forest":
        return Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            )),
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate(model, X_test, y_test):
    """Compute and print all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba, labels=classes)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:   {acc:.4f}")
    print(f"Log loss:   {logloss:.4f}")
    print(f"Macro F1:   {macro_f1:.4f}")

    # Per-class report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"Confusion Matrix (rows=actual, cols=predicted):")
    print(f"Classes: {list(classes)}")
    print(cm)

    return {"accuracy": acc, "log_loss": logloss, "macro_f1": macro_f1}


def print_feature_importance(model, model_type: str):
    """Print feature importance for interpretability."""
    print(f"\nFeature Importance:")

    if model_type == "logistic":
        coefs = model.named_steps["model"].coef_
        classes = model.named_steps["model"].classes_
        print(f"{'Feature':<22} " + "  ".join(f"{c:>8}" for c in classes))
        print("-" * 50)
        for i, feat in enumerate(FEATURE_COLS):
            vals = "  ".join(f"{coefs[j][i]:>8.4f}" for j in range(len(classes)))
            print(f"{feat:<22} {vals}")

    elif model_type in ("tree", "forest"):
        importances = model.named_steps["model"].feature_importances_
        pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
        for feat, imp in pairs:
            bar = "#" * int(imp * 40)
            print(f"  {feat:<22} {imp:.4f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Train chess outcome prediction model")
    parser.add_argument(
        "--model",
        choices=["logistic", "tree", "forest"],
        default="logistic",
        help="Model type (default: logistic)",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)} games")
    print(f"  Test:  {len(X_test)} games")
    print(f"  Features: {FEATURE_COLS}")

    # Train
    print(f"\nTraining: {args.model}...")
    model = build_model(args.model)
    model.fit(X_train, y_train)
    print("  Done.")

    # Evaluate
    evaluate(model, X_test, y_test)

    # Feature importance
    print_feature_importance(model, args.model)


if __name__ == "__main__":
    main()
