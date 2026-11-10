import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }
    return metrics


def get_confusion_matrix(y_true, y_pred, labels = None):
    if labels is None:
        labels = ["Low", "Medium", "High"]
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels = None,
    title = "Confusion Matrix",
    save_filename = None
):

    if labels is None:
        labels = ["Low", "Medium", "High"]

    cm = get_confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.title(title, fontsize=12, weight="bold")
    plt.xlabel("Predicted Risk", fontsize=11)
    plt.ylabel("Actual Risk", fontsize=11)
    plt.tight_layout()

    if save_filename:
        save_path = FIGURES_DIR / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gca()


def save_metrics_to_json(metrics, filename):
    destination = METRICS_DIR / filename
    with open(destination, "w") as f:
        json.dump(metrics, f, indent=4)
    return destination

