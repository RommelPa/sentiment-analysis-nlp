from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_reviews.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_reviews.csv"

MODEL_METRICS_PATH = REPORTS_DIR / "model_metrics.csv"
VALIDATION_PREDICTIONS_PATH = REPORTS_DIR / "validation_predictions.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TEXT_COLUMN = "clean_review"
TARGET_LABEL_COLUMN = "sentiment_label"

RANDOM_STATE = 42


def load_modeling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train and validation datasets.
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

    return train_data, validation_data


def split_features_target(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split text features and binary target.
    """
    X_train = train_data[TEXT_COLUMN].astype(str)
    y_train = train_data[TARGET_LABEL_COLUMN]

    X_valid = validation_data[TEXT_COLUMN].astype(str)
    y_valid = validation_data[TARGET_LABEL_COLUMN]

    return X_train, y_train, X_valid, y_valid


def build_tfidf_vectorizer(
    max_features: int = 50_000,
) -> TfidfVectorizer:
    """
    Build TF-IDF vectorizer for cleaned review text.
    """
    return TfidfVectorizer(
        lowercase=False,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        max_features=max_features,
        sublinear_tf=True,
    )


def build_models() -> dict[str, Pipeline]:
    """
    Build baseline and TF-IDF classification pipelines.
    """
    models = {
        "baseline_most_frequent": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer(max_features=10_000)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression_tfidf": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer(max_features=50_000)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm_tfidf": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer(max_features=50_000)),
                (
                    "model",
                    LinearSVC(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "naive_bayes_tfidf": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer(max_features=50_000)),
                ("model", MultinomialNB()),
            ]
        ),
        "random_forest_tfidf": Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer(max_features=10_000)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=40,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


def get_model_scores(model: Pipeline, X_valid: pd.Series) -> np.ndarray:
    """
    Get score for positive class.

    Some models expose predict_proba. LinearSVC exposes decision_function.
    Both are valid ranking scores for ROC-AUC and Average Precision.
    """
    classifier = model.named_steps["model"]

    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X_valid)[:, 1]

    if hasattr(classifier, "decision_function"):
        return model.decision_function(X_valid)

    raise AttributeError(
        "Model does not expose predict_proba or decision_function."
    )


def evaluate_model(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """
    Evaluate classifier using classification and ranking metrics.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }

    return metrics


def train_and_evaluate_models(
    models: dict[str, Pipeline],
    X_train: pd.Series,
    y_train: pd.Series,
    X_valid: pd.Series,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    """
    Train all models and evaluate them on validation data.
    """
    results = []
    trained_models = {}
    prediction_frames = []

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_valid)
        y_score = get_model_scores(pipeline, X_valid)

        metrics = evaluate_model(
            model_name=model_name,
            y_true=y_valid,
            y_pred=y_pred,
            y_score=y_score,
        )

        results.append(metrics)
        trained_models[model_name] = pipeline

        prediction_frame = pd.DataFrame(
            {
                "model": model_name,
                "actual": y_valid.values,
                "predicted_label": y_pred,
                "score": y_score,
            }
        )

        prediction_frames.append(prediction_frame)

    results_df = (
        pd.DataFrame(results)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    validation_predictions = pd.concat(
        prediction_frames,
        axis=0,
        ignore_index=True,
    )

    return results_df, trained_models, validation_predictions


def save_outputs(
    results_df: pd.DataFrame,
    trained_models: dict[str, Pipeline],
    validation_predictions: pd.DataFrame,
) -> None:
    """
    Save model metrics, validation predictions, and best model artifact.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(MODEL_METRICS_PATH, index=False)
    validation_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)

    best_model_name = results_df.loc[0, "model"]
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, BEST_MODEL_PATH)

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Model metrics: {MODEL_METRICS_PATH}")
    print(f"Validation predictions: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Best model name: {best_model_name}")


def print_results(results_df: pd.DataFrame) -> None:
    """
    Print model comparison results.
    """
    print("\n" + "=" * 80)
    print("IMDB SENTIMENT MODEL COMPARISON")
    print("=" * 80)

    formatted = results_df.copy()

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for column in metric_columns:
        formatted[column] = formatted[column].round(4)

    print(formatted)


if __name__ == "__main__":
    train_reviews, validation_reviews = load_modeling_data()

    X_train, y_train, X_valid, y_valid = split_features_target(
        train_data=train_reviews,
        validation_data=validation_reviews,
    )

    model_pipelines = build_models()

    model_results, fitted_models, validation_predictions = train_and_evaluate_models(
        models=model_pipelines,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )

    print_results(model_results)

    save_outputs(
        results_df=model_results,
        trained_models=fitted_models,
        validation_predictions=validation_predictions,
    )