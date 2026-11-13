"""Train, select, evaluate, and save CareerFit role classifiers."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .dataset import load_and_validate_dataset, read_dataset_metadata, split_dataset
from .evaluation import evaluate_predictions


MODEL_FILENAME = "careerfit_role_model.joblib"


def _models(seed: int) -> dict[str, Pipeline]:
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=12000,
        )

    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", random_state=seed),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=180,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_card(report: dict[str, Any]) -> str:
    test = report["selected_model_test_metrics"]
    metadata = report["dataset_metadata"]
    warning = metadata.get("warning") or metadata.get("limitations") or ""
    return f"""# CareerFit Phase 3 Model Card

## Intended use

This model predicts a broad target-role label from resume text. It supports CareerFit's
explainable ranking workflow; it must not make autonomous hiring decisions.

## Training data

- Source type: `{metadata.get('source_type', 'unknown')}`
- Dataset fingerprint: `{report['dataset_audit']['content_fingerprint']}`
- Unique usable samples: {report['dataset_audit']['usable_rows']}
- Classes: {', '.join(report['dataset_audit']['class_counts'])}

{warning}

## Selection protocol

Three models were trained on the training partition. The winner was selected by validation
macro-F1 (accuracy used only as a tiebreaker), refitted on training plus validation data,
and evaluated once on the untouched test partition.

## Selected model and held-out result

- Model: `{report['selected_model']}`
- Test accuracy: {test['accuracy']:.3f}
- Test macro-F1: {test['f1_macro']:.3f}
- Test samples: {test['sample_count']}

## Limitations

- Performance depends on dataset quality, labelling, representativeness, and class balance.
- TF-IDF primarily captures lexical patterns, not deep meaning.
- Linear SVM confidence is calibrated with three-fold sigmoid calibration inside the training data.
- Native Logistic Regression and Random Forest probabilities may still need calibration checks on real data.
- Identity fields must remain outside model input.
- Demo-data metrics are smoke-test evidence only and must not be presented as real-world accuracy.
"""


def train_and_evaluate(
    dataset_path: str | Path,
    artifacts_directory: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    text_column: str = "resume_text",
    label_column: str = "target_role",
) -> dict[str, Any]:
    """Run a fully reproducible train/validate/test experiment."""

    frame, audit = load_and_validate_dataset(
        dataset_path,
        text_column=text_column,
        label_column=label_column,
    )
    splits = split_dataset(
        frame,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    labels = sorted(frame["target_role"].unique().tolist())
    validation_results: dict[str, Any] = {}
    model_candidates = _models(seed)
    for name, pipeline in model_candidates.items():
        pipeline.fit(splits.train["resume_text"], splits.train["target_role"])
        predictions = pipeline.predict(splits.validation["resume_text"])
        validation_results[name] = evaluate_predictions(
            splits.validation["target_role"], predictions, labels=labels
        )

    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["f1_macro"],
            validation_results[name]["accuracy"],
            -list(model_candidates).index(name),
        ),
    )

    development = pd.concat([splits.train, splits.validation], ignore_index=True)
    selected_pipeline = _models(seed)[selected_name]
    selected_pipeline.fit(development["resume_text"], development["target_role"])
    test_predictions = selected_pipeline.predict(splits.test["resume_text"])
    test_metrics = evaluate_predictions(
        splits.test["target_role"], test_predictions, labels=labels
    )

    output = Path(artifacts_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = read_dataset_metadata(dataset_path)
    report: dict[str, Any] = {
        "careerfit_phase": 3,
        "trained_at": trained_at,
        "selection_metric": "validation_macro_f1",
        "selected_model": selected_name,
        "seed": seed,
        "dataset_audit": audit.to_dict(),
        "dataset_metadata": metadata,
        "split_manifest": splits.manifest(),
        "validation_metrics": validation_results,
        "model_probability_methods": {
            "logistic_regression": "native_predict_proba",
            "linear_svm": "three_fold_sigmoid_calibration",
            "random_forest": "native_predict_proba",
        },
        "selected_model_test_metrics": test_metrics,
        "reporting_warning": (
            "Metrics from a synthetic/demo dataset validate the pipeline only. "
            "Do not present them as real-world model performance."
            if metadata.get("source_type") == "synthetic_demo"
            else "Document dataset provenance and limitations with every reported metric."
        ),
    }

    bundle = {
        "schema_version": 2,
        "careerfit_phase": 3,
        "trained_at": trained_at,
        "selected_model": selected_name,
        "probability_method": {
            "logistic_regression": "native_predict_proba",
            "linear_svm": "three_fold_sigmoid_calibration",
            "random_forest": "native_predict_proba",
        }[selected_name],
        "classes": labels,
        "seed": seed,
        "dataset_fingerprint": audit.content_fingerprint,
        "pipeline": selected_pipeline,
    }
    joblib.dump(bundle, output / MODEL_FILENAME)
    _write_json(output / "evaluation_report.json", report)
    _write_json(output / "dataset_audit.json", audit.to_dict())
    _write_json(output / "split_manifest.json", splits.manifest())
    (output / "MODEL_CARD.md").write_text(_model_card(report), encoding="utf-8")

    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, test_metrics["confusion_matrix"], strict=True):
            writer.writerow([label, *row])

    predictions_frame = splits.test[["record_id", "target_role", "fingerprint"]].copy()
    predictions_frame["predicted_role"] = test_predictions
    predictions_frame["correct"] = predictions_frame["target_role"] == test_predictions
    predictions_frame.to_csv(output / "test_predictions.csv", index=False)

    return {
        "careerfit_phase": 3,
        "selected_model": selected_name,
        "model_path": str(output / MODEL_FILENAME),
        "evaluation_report_path": str(output / "evaluation_report.json"),
        "test_metrics": test_metrics,
        "reporting_warning": report["reporting_warning"],
    }
