import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.features import build_dataset
from src.config import Config


def _train_val_split(X, y, ratio: float):
    split = int(len(X) * ratio)
    return X[:split], y[:split], X[split:], y[split:]

def train_model(config: Config, logger) -> None:
    data = build_dataset(config, logger)
    X, y = data
    X_train, y_train, X_val, y_val = _train_val_split(X, y, config.train.train_ratio)

    if config.train.model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(**config.train.model_params)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, -1] if hasattr(model, "predict_proba") else None

    metrics = {
        "precision": precision_score(y_val, preds, average="macro", zero_division=0),
        "recall": recall_score(y_val, preds, average="macro", zero_division=0),
        "f1": f1_score(y_val, preds, average="macro", zero_division=0),
        "auc": roc_auc_score(y_val, proba, multi_class="ovr") if proba is not None and len(set(y_val)) > 2 else None,
    }
    logger.info("Training completed. Metrics: %s", metrics)

    out_path = config.train.model_output
    joblib.dump(model, out_path)
    logger.info("Model saved to %s", out_path)


def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {path}: {exc}") from exc
