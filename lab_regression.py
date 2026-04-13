"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
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
    """Split data into train and test sets with stratification when appropriate.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        stratify_arg = y if target_col == "churned" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg
        )

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data for target '{target_col}': {e}")
        return None


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipe


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipe


def build_lasso_pipeline():
    """Build a Pipeline with StandardScaler and Lasso regression.

    Returns:
        sklearn Pipeline object.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
    return pipe


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        return metrics
    except Exception as e:
        print(f"Error evaluating classifier: {e}")
        return None


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        return metrics
    except Exception as e:
        print(f"Error evaluating regressor: {e}")
        return None


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    try:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="accuracy"
        )

        print("\nCross-validation fold scores:")
        print(scores)
        print(f"CV Mean: {scores.mean():.3f} +/- {scores.std():.3f}")

        return scores
    except Exception as e:
        print(f"Error running cross-validation: {e}")
        return None


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Task 1: Basic EDA
        print("\nMissing values per column:")
        print(df.isnull().sum())

        print("\nChurn distribution:")
        print(df["churned"].value_counts(normalize=True))

        # Select numeric features for classification
        numeric_features = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
            "has_partner",
            "has_dependents"
        ]

        # Classification: predict churn
        df_cls = df[numeric_features + ["churned"]].dropna()
        split = split_data(df_cls, "churned")

        if split is not None:
            X_train, X_test, y_train, y_test = split

            print("\nClassification split shapes:")
            print("X_train:", X_train.shape)
            print("X_test:", X_test.shape)
            print("y_train:", y_train.shape)
            print("y_test:", y_test.shape)

            print("\nTrain churn rate:")
            print(y_train.value_counts(normalize=True))

            print("\nTest churn rate:")
            print(y_test.value_counts(normalize=True))

            pipe = build_logistic_pipeline()
            metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)

            if metrics is not None:
                print(f"\nLogistic Regression Metrics: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV Summary: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Regression: predict monthly_charges
        reg_features = [
            "tenure",
            "total_charges",
            "num_support_calls",
            "senior_citizen",
            "has_partner",
            "has_dependents"
        ]

        df_reg = df[reg_features + ["monthly_charges"]].dropna()
        split_reg = split_data(df_reg, "monthly_charges")

        if split_reg is not None:
            X_tr, X_te, y_tr, y_te = split_reg

            ridge_pipe = build_ridge_pipeline()
            reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)

            if reg_metrics is not None:
                print(f"\nRidge Regression Metrics: {reg_metrics}")

            # Task 5: Lasso comparison
            lasso_pipe = build_lasso_pipeline()

            try:
                ridge_pipe.fit(X_tr, y_tr)
                lasso_pipe.fit(X_tr, y_tr)

                ridge_coefs = ridge_pipe.named_steps["model"].coef_
                lasso_coefs = lasso_pipe.named_steps["model"].coef_

                print("\nRidge coefficients:")
                for feature, coef in zip(reg_features, ridge_coefs):
                    print(f"{feature}: {coef:.4f}")

                print("\nLasso coefficients:")
                for feature, coef in zip(reg_features, lasso_coefs):
                    print(f"{feature}: {coef:.4f}")

                zeroed_features = [
                    feature for feature, coef in zip(reg_features, lasso_coefs) if coef == 0
                ]

                print("\nFeatures driven to zero by Lasso:")
                if zeroed_features:
                    for feature in zeroed_features:
                        print(feature)
                else:
                    print("None")
            except Exception as e:
                print(f"Error comparing Ridge and Lasso coefficients: {e}")

    """
    Summary of findings:

    From the results, it looks like some features such as monthly_charges and
    num_support_calls have a stronger impact on predicting churn compared to others.

    The logistic regression model performs reasonably well, but since the dataset
    is imbalanced, accuracy alone is not very reliable. Recall is more important
    in this case because it helps us identify more customers who are likely to churn.

    To improve performance, we could try tuning the model parameters, adding new
    features, or using more advanced models like Random Forest or Gradient Boosting.
    """