"""Load saved model bundles and make role predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model does not exist: {model_path}")
    bundle = joblib.load(model_path)
    required = {"schema_version", "selected_model", "pipeline", "classes"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("The file is not a compatible CareerFit Phase 3 model bundle.")
    return bundle


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def predict_probabilities(bundle: dict[str, Any], texts: list[str]) -> np.ndarray:
    pipeline = bundle["pipeline"]
    if hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict_proba(texts), dtype=float)

    decision = np.asarray(pipeline.decision_function(texts), dtype=float)
    if decision.ndim == 1:
        decision = np.column_stack((-decision, decision))
    return _softmax(decision)


def predict_text(bundle: dict[str, Any], text: str) -> dict[str, Any]:
    classes = [str(value) for value in bundle["classes"]]
    probabilities = predict_probabilities(bundle, [text])[0]
    distribution = {
        label: round(float(probability), 6)
        for label, probability in sorted(
            zip(classes, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    predicted_role = next(iter(distribution))
    return {
        "predicted_role": predicted_role,
        "confidence": distribution[predicted_role],
        "role_probabilities": distribution,
        "model_name": str(bundle["selected_model"]),
        "probability_method": str(bundle.get("probability_method", "legacy_or_unknown")),
    }
