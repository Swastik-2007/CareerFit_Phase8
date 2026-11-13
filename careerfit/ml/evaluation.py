"""Classification metrics with JSON-safe outputs."""

from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    truth: Iterable[str],
    predictions: Iterable[str],
    *,
    labels: list[str],
) -> dict[str, Any]:
    truth_values = list(truth)
    predicted_values = list(predictions)
    matrix = confusion_matrix(truth_values, predicted_values, labels=labels)
    report = classification_report(
        truth_values,
        predicted_values,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "sample_count": len(truth_values),
        "accuracy": round(float(accuracy_score(truth_values, predicted_values)), 6),
        "precision_macro": round(
            float(precision_score(truth_values, predicted_values, average="macro", zero_division=0)),
            6,
        ),
        "recall_macro": round(
            float(recall_score(truth_values, predicted_values, average="macro", zero_division=0)),
            6,
        ),
        "f1_macro": round(
            float(f1_score(truth_values, predicted_values, average="macro", zero_division=0)),
            6,
        ),
        "f1_weighted": round(
            float(f1_score(truth_values, predicted_values, average="weighted", zero_division=0)),
            6,
        ),
        "labels": labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": {
            label: {
                key: round(float(report[label][key]), 6)
                for key in ("precision", "recall", "f1-score")
            }
            | {"support": int(report[label]["support"])}
            for label in labels
        },
    }
