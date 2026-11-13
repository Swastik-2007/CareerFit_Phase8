"""Reproducible machine-learning utilities for CareerFit Phase 3."""

from .dataset import DatasetValidationError, audit_dataset, load_and_validate_dataset
from .features import candidate_model_text, sanitize_model_text
from .inference import load_model_bundle, predict_text
from .similarity import text_similarity
from .training import train_and_evaluate

__all__ = [
    "DatasetValidationError",
    "audit_dataset",
    "candidate_model_text",
    "load_and_validate_dataset",
    "load_model_bundle",
    "predict_text",
    "sanitize_model_text",
    "text_similarity",
    "train_and_evaluate",
]
