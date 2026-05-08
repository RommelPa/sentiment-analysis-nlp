# Sentiment Analysis NLP: Classifying Movie Reviews with Machine Learning

## Overview

This project builds and evaluates NLP classification models to predict sentiment in movie reviews.

The goal is not only to classify text, but also to understand text preprocessing, TF-IDF vectorization, model comparison, error analysis, and interpretable sentiment signals.

## Business Context

Sentiment analysis helps organizations understand customer opinions at scale.

A sentiment classifier can support product reviews analysis, customer feedback monitoring, social media analysis, and market research.

However, text classification models must be evaluated carefully because they can fail on sarcasm, negation, ambiguous language, domain-specific expressions, and long reviews with mixed sentiment.

## Objectives

- Load and audit an IMDB movie review sentiment dataset.
- Clean raw review text.
- Analyze class balance and review length.
- Build baseline and TF-IDF classification models.
- Compare Logistic Regression, Linear SVM, Naive Bayes, and Random Forest.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix.
- Analyze classification errors.
- Interpret relevant words associated with positive and negative sentiment.
- Generate business-oriented recommendations and limitations.

## Project Structure

```text
sentiment-analysis-nlp/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_sentiment_analysis.ipynb
├── reports/
│   ├── executive_summary_en.md
│   ├── resumen_ejecutivo_es.md
│   └── figures/
├── src/
│   ├── load_data.py
│   ├── audit_data.py
│   ├── preprocess_text.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── interpret_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Status

Project in progress.
