from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from train_models import build_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_reviews.csv"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation_reviews.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_reviews.csv"

FINAL_TEST_PREDICTIONS_PATH = REPORTS_DIR / "final_test_predictions.csv"

TOKEN_COEFFICIENTS_PATH = REPORTS_DIR / "linear_svm_token_coefficients.csv"
TOP_POSITIVE_TOKENS_PATH = REPORTS_DIR / "top_positive_tokens.csv"
TOP_NEGATIVE_TOKENS_PATH = REPORTS_DIR / "top_negative_tokens.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"

TEXT_COLUMN = "clean_review"
TARGET_LABEL_COLUMN = "sentiment_label"
TARGET_COLUMN = "sentiment"

FINAL_MODEL_NAME = "linear_svm_tfidf"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets.
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


def load_final_predictions() -> pd.DataFrame:
    """
    Load final test predictions.
    """
    if not FINAL_TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Missing final test predictions. Run 'python src/evaluate_models.py' first."
        )

    predictions = pd.read_csv(FINAL_TEST_PREDICTIONS_PATH)

    final_predictions = predictions[
        predictions["model"] == FINAL_MODEL_NAME
    ].copy()

    if final_predictions.empty:
        raise ValueError(
            f"No predictions found for final model '{FINAL_MODEL_NAME}'."
        )

    return final_predictions


def train_final_linear_svm(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
):
    """
    Train final Linear SVM on train + validation data.
    """
    train_validation_data = pd.concat(
        [train_data, validation_data],
        axis=0,
        ignore_index=True,
    )

    X_train_validation = train_validation_data[TEXT_COLUMN].astype(str)
    y_train_validation = train_validation_data[TARGET_LABEL_COLUMN]

    models = build_models()

    if FINAL_MODEL_NAME not in models:
        raise ValueError(f"Model '{FINAL_MODEL_NAME}' not found in build_models().")

    model = models[FINAL_MODEL_NAME]
    model.fit(X_train_validation, y_train_validation)

    return model


def extract_token_coefficients(model) -> pd.DataFrame:
    """
    Extract token coefficients from final Linear SVM model.

    Positive coefficients indicate association with positive sentiment.
    Negative coefficients indicate association with negative sentiment.
    """
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["model"]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    coefficients_df = pd.DataFrame(
        {
            "token": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficients_df["absolute_coefficient"] = coefficients_df["coefficient"].abs()

    coefficients_df = (
        coefficients_df
        .sort_values("absolute_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    return coefficients_df


def build_error_analysis(
    test_data: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build error analysis table without storing review text.

    The output avoids saving raw review text to the repository.
    """
    test_features = test_data.copy().reset_index(drop=True)
    test_features["test_row_id"] = test_features.index

    test_features["clean_word_count"] = (
        test_features[TEXT_COLUMN]
        .astype(str)
        .str.split()
        .str.len()
    )

    analysis = final_predictions.merge(
        test_features[
            [
                "test_row_id",
                TARGET_COLUMN,
                TARGET_LABEL_COLUMN,
                "clean_word_count",
                "clean_character_length",
            ]
        ],
        on="test_row_id",
        how="left",
        suffixes=("", "_from_test"),
    )

    analysis["is_correct"] = analysis["actual"] == analysis["predicted_label"]
    analysis["is_error"] = ~analysis["is_correct"]
    analysis["absolute_score"] = analysis["score"].abs()

    analysis["actual_sentiment"] = analysis["actual"].map(
        {
            0: "negative",
            1: "positive",
        }
    )
    analysis["predicted_sentiment"] = analysis["predicted_label"].map(
        {
            0: "negative",
            1: "positive",
        }
    )

    analysis["error_type"] = "correct"
    analysis.loc[
        (analysis["actual"] == 0) & (analysis["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"
    analysis.loc[
        (analysis["actual"] == 1) & (analysis["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    output_columns = [
        "model",
        "test_row_id",
        "actual",
        "predicted_label",
        "actual_sentiment",
        "predicted_sentiment",
        "score",
        "absolute_score",
        "is_correct",
        "is_error",
        "error_type",
        "clean_word_count",
        "clean_character_length",
    ]

    return analysis[output_columns]


def save_tables(
    coefficients_df: pd.DataFrame,
    error_analysis: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save interpretation and error analysis tables.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    coefficients_df.to_csv(TOKEN_COEFFICIENTS_PATH, index=False)

    top_positive = (
        coefficients_df
        .sort_values("coefficient", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    top_negative = (
        coefficients_df
        .sort_values("coefficient", ascending=True)
        .head(30)
        .reset_index(drop=True)
    )

    top_positive.to_csv(TOP_POSITIVE_TOKENS_PATH, index=False)
    top_negative.to_csv(TOP_NEGATIVE_TOKENS_PATH, index=False)
    error_analysis.to_csv(ERROR_ANALYSIS_PATH, index=False)

    print("\nSaved interpretation tables")
    print("-" * 80)
    print(f"Token coefficients: {TOKEN_COEFFICIENTS_PATH}")
    print(f"Top positive tokens: {TOP_POSITIVE_TOKENS_PATH}")
    print(f"Top negative tokens: {TOP_NEGATIVE_TOKENS_PATH}")
    print(f"Error analysis: {ERROR_ANALYSIS_PATH}")

    return top_positive, top_negative


def save_figures(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    error_analysis: pd.DataFrame,
) -> None:
    """
    Save interpretation and error analysis figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_positive.head(20),
        x="coefficient",
        y="token",
    )
    plt.title("Top Positive Sentiment Tokens — Linear SVM")
    plt.xlabel("Coefficient")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_positive_sentiment_tokens.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_negative.head(20),
        x="coefficient",
        y="token",
    )
    plt.title("Top Negative Sentiment Tokens — Linear SVM")
    plt.xlabel("Coefficient")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_negative_sentiment_tokens.png", dpi=300)
    plt.close()

    error_counts = (
        error_analysis["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )

    plt.figure(figsize=(9, 6))
    sns.barplot(data=error_counts, x="error_type", y="count")
    plt.title("Final Model Prediction Outcomes")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prediction_outcomes.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=error_analysis,
        x="is_error",
        y="clean_word_count",
    )
    plt.title("Review Length by Correct vs Incorrect Predictions")
    plt.xlabel("Is Error")
    plt.ylabel("Clean Word Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "review_length_by_error_status.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=error_analysis,
        x="score",
        hue="is_error",
        bins=40,
        kde=True,
        element="step",
    )
    plt.title("Linear SVM Score Distribution by Error Status")
    plt.xlabel("Linear SVM Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "score_distribution_by_error_status.png", dpi=300)
    plt.close()

    print("\nSaved interpretation figures")
    print("-" * 80)
    print(f"Top positive tokens: {FIGURES_DIR / 'top_positive_sentiment_tokens.png'}")
    print(f"Top negative tokens: {FIGURES_DIR / 'top_negative_sentiment_tokens.png'}")
    print(f"Prediction outcomes: {FIGURES_DIR / 'prediction_outcomes.png'}")
    print(f"Review length by error status: {FIGURES_DIR / 'review_length_by_error_status.png'}")
    print(f"Score distribution by error status: {FIGURES_DIR / 'score_distribution_by_error_status.png'}")


def print_interpretation_summary(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    error_analysis: pd.DataFrame,
) -> None:
    """
    Print interpretation summary.
    """
    print("=" * 80)
    print("FINAL SENTIMENT MODEL INTERPRETATION")
    print("=" * 80)

    print("\nImportant note")
    print(
        "Linear SVM token coefficients show associations with positive or negative "
        "sentiment in the TF-IDF feature space. They are not causal explanations."
    )

    print("\nTop 20 positive sentiment tokens")
    print(top_positive[["token", "coefficient"]].head(20).round(4))

    print("\nTop 20 negative sentiment tokens")
    print(top_negative[["token", "coefficient"]].head(20).round(4))

    print("\nPrediction outcome counts")
    print(error_analysis["error_type"].value_counts())

    print("\nError rate by actual sentiment")
    error_rate_by_sentiment = (
        error_analysis
        .groupby("actual_sentiment")["is_error"]
        .mean()
        .mul(100)
        .round(2)
    )
    print(error_rate_by_sentiment)

    print("\nReview length summary by error status")
    length_summary = (
        error_analysis
        .groupby("is_error")["clean_word_count"]
        .describe()
        .round(2)
    )
    print(length_summary)


if __name__ == "__main__":
    train_reviews, validation_reviews, test_reviews = load_data()
    final_predictions = load_final_predictions()

    final_model = train_final_linear_svm(
        train_data=train_reviews,
        validation_data=validation_reviews,
    )

    token_coefficients = extract_token_coefficients(final_model)

    error_analysis_df = build_error_analysis(
        test_data=test_reviews,
        final_predictions=final_predictions,
    )

    top_positive_tokens, top_negative_tokens = save_tables(
        coefficients_df=token_coefficients,
        error_analysis=error_analysis_df,
    )

    print_interpretation_summary(
        top_positive=top_positive_tokens,
        top_negative=top_negative_tokens,
        error_analysis=error_analysis_df,
    )

    save_figures(
        top_positive=top_positive_tokens,
        top_negative=top_negative_tokens,
        error_analysis=error_analysis_df,
    )