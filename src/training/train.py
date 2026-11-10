from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing.preprocess import get_preprocessor
from src.evaluation.evaluate import calculate_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def build_pipeline(classifier):
    return Pipeline(steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", classifier)
    ])


def get_baseline_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    }


def train_and_evaluate_baselines(
    X_train,
    y_train,
    X_test,
    y_test
):
    models = get_baseline_models()
    trained_pipelines = {}
    results = {}

    for name, clf in models.items():
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        trained_pipelines[name] = pipeline
        results[name] = metrics

    return trained_pipelines, results


def save_model(pipeline, filename = "best_pipeline.joblib"):
    destination = MODELS_DIR / filename
    joblib.dump(pipeline, destination)
    return destination


def load_model(filename = "best_pipeline.joblib"):
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return joblib.load(filepath)

