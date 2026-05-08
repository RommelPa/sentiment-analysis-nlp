from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from train_models import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_reviews.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_reviews.csv"

CROSS_VALIDATION_METRICS_PATH = REPORTS_DIR / "cross_validation_metrics.csv"
CROSS_VALIDATION_SUMMARY_PATH = REPORTS_DIR / "cross_validation_summary.csv"

TEXT_COLUMN = "clean_review"
TARGET_LABEL_COLUMN = "sentiment_label"

N_SPLITS = 3
RANDOM_STATE = 42


def load_train_validation_data() -> pd.DataFrame:
    """
    Load train and validation data for cross-validation.

    The held-out test split is not used here.
    """
    required_files = [
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
    ]

    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Missing processed files. Run 'python src/preprocess_text.py' first.\n"
            f"Missing files:\n{missing}"
        )

    train_data = pd.read_csv(TRAIN_DATA_PATH)
    validation_data = pd.read_csv(VALIDATION_DATA_PATH)

    combined_data = pd.concat(
        [train_data, validation_data],
        axis=0,
        ignore_index=True,
    )

    return combined_data


def get_model_scores(model, X_valid: pd.Series) -> np.ndarray:
    """
    Get positive-class score for metrics like ROC-AUC and Average Precision.
    """
    classifier = model.named_steps["model"]

    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X_valid)[:, 1]

    if hasattr(classifier, "decision_function"):
        return model.decision_function(X_valid)

    raise AttributeError(
        "Model does not expose predict_proba or decision_function."
    )


def evaluate_fold_predictions(
    model_name: str,
    fold: int,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Evaluate one cross-validation fold.
    """
    return {
        "model": model_name,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def run_cross_validation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run stratified K-fold cross-validation for all sentiment models.
    """
    X = data[TEXT_COLUMN].astype(str)
    y = data[TARGET_LABEL_COLUMN]

    models = build_models()

    stratified_kfold = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, model_pipeline in models.items():
        print(f"Cross-validating {model_name}...")

        for fold_index, (train_index, valid_index) in enumerate(
            stratified_kfold.split(X, y),
            start=1,
        ):
            X_train_fold = X.iloc[train_index]
            X_valid_fold = X.iloc[valid_index]
            y_train_fold = y.iloc[train_index]
            y_valid_fold = y.iloc[valid_index]

            fold_model = clone(model_pipeline)
            fold_model.fit(X_train_fold, y_train_fold)

            y_pred = fold_model.predict(X_valid_fold)
            y_score = get_model_scores(fold_model, X_valid_fold)

            fold_metrics = evaluate_fold_predictions(
                model_name=model_name,
                fold=fold_index,
                y_true=y_valid_fold,
                y_pred=y_pred,
                y_score=y_score,
            )

            results.append(fold_metrics)

    return pd.DataFrame(results)


def build_cross_validation_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation metrics by model.
    """
    summary = (
        cv_results
        .groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
        )
        .sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
        .reset_index(drop=True)
    )

    return summary


def save_outputs(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
) -> None:
    """
    Save cross-validation outputs.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cv_results.to_csv(CROSS_VALIDATION_METRICS_PATH, index=False)
    cv_summary.to_csv(CROSS_VALIDATION_SUMMARY_PATH, index=False)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Fold metrics: {CROSS_VALIDATION_METRICS_PATH}")
    print(f"Summary metrics: {CROSS_VALIDATION_SUMMARY_PATH}")


def print_cross_validation_summary(cv_summary: pd.DataFrame) -> None:
    """
    Print cross-validation summary.
    """
    print("\n" + "=" * 80)
    print("IMDB SENTIMENT CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    formatted = cv_summary.copy()

    metric_columns = [
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "average_precision_mean",
        "average_precision_std",
    ]

    for column in metric_columns:
        formatted[column] = formatted[column].round(4)

    print(formatted)


if __name__ == "__main__":
    train_validation_reviews = load_train_validation_data()

    cross_validation_results = run_cross_validation(train_validation_reviews)
    cross_validation_summary = build_cross_validation_summary(cross_validation_results)

    print_cross_validation_summary(cross_validation_summary)

    save_outputs(
        cv_results=cross_validation_results,
        cv_summary=cross_validation_summary,
    )