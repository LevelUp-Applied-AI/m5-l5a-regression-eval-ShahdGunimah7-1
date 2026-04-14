import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


def load_data(filepath="data/telecom_churn.csv"):
    candidate_paths = [
        Path(filepath),
        Path("data/telecom_churn.csv"),
        Path("starter/data/telecom_churn.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path)

    print(f"Error: Could not find file at '{filepath}'")
    return None


def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_arg = y if target_col == "churned" else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )


def build_logistic_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


# -----------------------------
# Tier 1: Threshold Tuning
# -----------------------------
def tune_thresholds(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    print("\nThreshold tuning results:")
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"Threshold={threshold:.1f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )

    best_result = max(results, key=lambda x: x["f1"])
    print(f"\nBest threshold by F1: {best_result['threshold']}")
    print(
        f"Best Precision={best_result['precision']:.3f}, "
        f"Best Recall={best_result['recall']:.3f}, "
        f"Best F1={best_result['f1']:.3f}"
    )

    x_vals = [r["threshold"] for r in results]
    precision_vals = [r["precision"] for r in results]
    recall_vals = [r["recall"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, precision_vals, marker="o", label="Precision")
    plt.plot(x_vals, recall_vals, marker="o", label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results, best_result


# -----------------------------
# Tier 2: Config-Driven Sweep
# -----------------------------
def build_model_from_config(model_config):
    model_type = model_config["model"]
    params = model_config.get("params", {})

    if model_type == "logistic":
        model = LogisticRegression(**params)
    elif model_type == "ridge":
        model = Ridge(**params)
    elif model_type == "lasso":
        model = Lasso(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


def run_model_sweep(config_path, X, y, classification=True):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    results = []

    for item in config["models"]:
        name = item["name"]
        pipeline = build_model_from_config(item)

        if classification:
            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring="accuracy"
            )
        else:
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=5,
                scoring="r2"
            )

        results.append({
            "name": name,
            "mean_score": scores.mean(),
            "std_score": scores.std()
        })

    print("\nModel sweep results:")
    for result in results:
        print(
            f"{result['name']}: "
            f"{result['mean_score']:.3f} +/- {result['std_score']:.3f}"
        )

    return results


# -----------------------------
# Tier 3: Logistic from Scratch
# -----------------------------
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iters=5000, lambda_reg=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (
                (1 / n_samples) * np.dot(X.T, (y_pred - y))
                + (self.lambda_reg / n_samples) * self.weights
            )
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def compare_scratch_vs_sklearn(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scratch_model = LogisticRegressionScratch(
        learning_rate=0.01,
        n_iters=5000,
        lambda_reg=0.1
    )
    scratch_model.fit(X_train_scaled, y_train)
    scratch_preds = scratch_model.predict(X_test_scaled)

    sklearn_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_preds = sklearn_model.predict(X_test_scaled)

    print("\nScratch Logistic Regression Report:")
    print(classification_report(y_test, scratch_preds, zero_division=0))

    print("\nScikit-learn Logistic Regression Report:")
    print(classification_report(y_test, sklearn_preds, zero_division=0))

    print("\nScratch coefficients:")
    print(scratch_model.weights)

    print("\nScikit-learn coefficients:")
    print(sklearn_model.coef_[0])


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        # Classification dataset
        cls_features = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
            "has_partner",
            "has_dependents"
        ]

        df_cls = df[cls_features + ["churned"]].dropna()
        X_train, X_test, y_train, y_test = split_data(df_cls, "churned")

        pipe = build_logistic_pipeline()

        # Tier 1
        tune_thresholds(pipe, X_train, X_test, y_train, y_test)

        # Tier 2
        run_model_sweep("model_config.json", X_train, y_train, classification=True)

        # Tier 3
        compare_scratch_vs_sklearn(X_train, X_test, y_train, y_test)