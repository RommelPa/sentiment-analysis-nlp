from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from train_models import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_reviews.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_reviews.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_reviews.csv"

FINAL_TEST_METRICS_PATH = REPORTS_DIR / "final_test_metrics.csv"
FINAL_TEST_PREDICTIONS_PATH = REPORTS_DIR / "final_test_predictions.csv"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "final_classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "final_confusion_matrix.csv"
FINAL_MODEL_PATH = MODELS_DIR / "final_linear_svm_model.joblib"

TEXT_COLUMN = "clean_review"
TARGET_LABEL_COLUMN = "sentiment_label"
TARGET_COLUMN = "sentiment"

FINAL_MODEL_NAME = "linear_svm_tfidf"


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test review datasets.
    """
    required_files = [
        TRAIN_DATA_PATH,
        VALIDATION_DATA_PATH,
        TEST_DATA_PATH,
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
    test_data = pd.read_csv(TEST_DATA_PATH)

    return train_data, validation_data, test_data


def get_model_scores(model, X_test: pd.Series):
    """
    Get positive-class ranking scores.
    """
    classifier = model.named_steps["model"]

    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(classifier, "decision_function"):
        return model.decision_function(X_test)

    raise AttributeError(
        "Model does not expose predict_proba or decision_function."
    )


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred,
    y_score,
) -> dict:
    """
    Evaluate final test predictions.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def train_final_models(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Train all candidate models on train + validation and evaluate on test.
    """
    train_validation_data = pd.concat(
        [train_data, validation_data],
        axis=0,
        ignore_index=True,
    )

    X_train_validation = train_validation_data[TEXT_COLUMN].astype(str)
    y_train_validation = train_validation_data[TARGET_LABEL_COLUMN]

    X_test = test_data[TEXT_COLUMN].astype(str)
    y_test = test_data[TARGET_LABEL_COLUMN]

    models = build_models()

    metrics = []
    prediction_frames = []
    final_model = None

    for model_name, model in models.items():
        print(f"Training final {model_name}...")

        model.fit(X_train_validation, y_train_validation)

        y_pred = model.predict(X_test)
        y_score = get_model_scores(model, X_test)

        metrics.append(
            evaluate_predictions(
                model_name=model_name,
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
            )
        )

        prediction_frame = pd.DataFrame(
            {
                "model": model_name,
                "test_row_id": test_data.index.values,
                "actual": y_test.values,
                "predicted_label": y_pred,
                "score": y_score,
                "sentiment": test_data[TARGET_COLUMN].values,
            }
        )
        
        prediction_frames.append(prediction_frame)

        if model_name == FINAL_MODEL_NAME:
            final_model = model

    metrics_df = (
        pd.DataFrame(metrics)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    predictions_df = pd.concat(
        prediction_frames,
        axis=0,
        ignore_index=True,
    )

    if final_model is None:
        raise ValueError(f"Final model '{FINAL_MODEL_NAME}' was not trained.")

    return metrics_df, predictions_df, final_model


def save_classification_report(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save classification report for final model.
    """
    final_predictions = predictions[predictions["model"] == FINAL_MODEL_NAME].copy()

    report = classification_report(
        final_predictions["actual"],
        final_predictions["predicted_label"],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(CLASSIFICATION_REPORT_PATH)

    return report_df


def save_confusion_matrix(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Save confusion matrix table for final model.
    """
    final_predictions = predictions[predictions["model"] == FINAL_MODEL_NAME].copy()

    matrix = confusion_matrix(
        final_predictions["actual"],
        final_predictions["predicted_label"],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Negative", "Actual Positive"],
        columns=["Predicted Negative", "Predicted Positive"],
    )

    matrix_df.to_csv(CONFUSION_MATRIX_PATH)

    return matrix_df


def save_figures(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    """
    Save final evaluation figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    final_predictions = predictions[predictions["model"] == FINAL_MODEL_NAME].copy()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics, x="f1", y="model")
    plt.title("Final Test F1 by Model")
    plt.xlabel("F1 Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_test_f1_by_model.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_df.values,
        display_labels=["negative", "positive"],
    ).plot(values_format="d", cmap="Blues")
    plt.title("Final Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(
        final_predictions["actual"],
        final_predictions["score"],
    )
    roc_auc = roc_auc_score(
        final_predictions["actual"],
        final_predictions["score"],
    )

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve — Final Linear SVM")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_roc_curve.png", dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(
        final_predictions["actual"],
        final_predictions["score"],
    )
    average_precision = average_precision_score(
        final_predictions["actual"],
        final_predictions["score"],
    )

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Average Precision = {average_precision:.3f}")
    plt.title("Precision-Recall Curve — Final Linear SVM")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_precision_recall_curve.png", dpi=300)
    plt.close()


def save_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    final_model,
    report_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> None:
    """
    Save final outputs and model artifact.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(FINAL_TEST_METRICS_PATH, index=False)
    predictions.to_csv(FINAL_TEST_PREDICTIONS_PATH, index=False)

    joblib.dump(final_model, FINAL_MODEL_PATH)

    save_figures(
        predictions=predictions,
        metrics=metrics,
        confusion_df=confusion_df,
    )

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Final test metrics: {FINAL_TEST_METRICS_PATH}")
    print(f"Final test predictions: {FINAL_TEST_PREDICTIONS_PATH}")
    print(f"Classification report: {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Final model artifact: {FINAL_MODEL_PATH}")
    print(f"F1 figure: {FIGURES_DIR / 'final_test_f1_by_model.png'}")
    print(f"Confusion matrix figure: {FIGURES_DIR / 'final_confusion_matrix.png'}")
    print(f"ROC curve: {FIGURES_DIR / 'final_roc_curve.png'}")
    print(f"Precision-recall curve: {FIGURES_DIR / 'final_precision_recall_curve.png'}")


def print_final_summary(metrics: pd.DataFrame, report_df: pd.DataFrame, confusion_df: pd.DataFrame) -> None:
    """
    Print final model summary.
    """
    print("=" * 80)
    print("FINAL IMDB SENTIMENT TEST EVALUATION")
    print("=" * 80)

    formatted_metrics = metrics.copy()

    for column in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        formatted_metrics[column] = formatted_metrics[column].round(4)

    print("\nFinal test metrics by model")
    print(formatted_metrics)

    print("\nClassification report for final model")
    print(report_df.round(4))

    print("\nConfusion matrix for final model")
    print(confusion_df)


if __name__ == "__main__":
    train_reviews, validation_reviews, test_reviews = load_processed_data()

    final_metrics, final_predictions, fitted_final_model = train_final_models(
        train_data=train_reviews,
        validation_data=validation_reviews,
        test_data=test_reviews,
    )

    classification_report_df = save_classification_report(final_predictions)
    confusion_matrix_df = save_confusion_matrix(final_predictions)

    print_final_summary(
        metrics=final_metrics,
        report_df=classification_report_df,
        confusion_df=confusion_matrix_df,
    )

    save_outputs(
        metrics=final_metrics,
        predictions=final_predictions,
        final_model=fitted_final_model,
        report_df=classification_report_df,
        confusion_df=confusion_matrix_df,
    )