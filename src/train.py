import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import build_vectorizer, clean_text_series, load_sentiment140_data


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def build_model(config):
    model_type = config["model"]["type"]

    if model_type == "logistic_regression":
        return LogisticRegression(
            C=config["model"]["C"],
            max_iter=config["model"]["max_iter"],
            class_weight=config["model"]["class_weight"],
        )

    if model_type == "linear_svc":
        return LinearSVC(
            C=config["model"]["C"],
            class_weight=config["model"]["class_weight"],
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def train_model(config):
    data_path = (
        config["data"]["sample_path"]
        if config["training"]["use_sample"]
        else config["data"]["raw_path"]
    )

    df = load_sentiment140_data(data_path)
    df["clean_text"] = clean_text_series(df["text"])

    X = df["clean_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    vectorizer = build_vectorizer(
        max_features=config["preprocessing"]["max_features"],
        ngram_range=config["preprocessing"]["ngram_range"],
    )

    model = build_model(config)

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("model", model),
        ]
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
        }

        mlflow.log_params(config["model"])
        mlflow.log_params(config["preprocessing"])
        mlflow.log_param("data_path", data_path)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model")

        Path("models").mkdir(exist_ok=True)
        joblib.dump(pipeline, "models/best_model.joblib")

        print("Evaluation metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        return pipeline, metrics


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    config = load_config(config_path)
    train_model(config)