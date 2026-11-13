"""Dataset validation and leakage-safe splitting for resume classification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import contains_contact_information, sanitize_model_text


class DatasetValidationError(ValueError):
    """Raised when a labelled dataset cannot be used safely."""


@dataclass(slots=True)
class DatasetAudit:
    source_file: str
    text_column: str
    label_column: str
    total_rows: int
    usable_rows: int
    missing_rows_removed: int
    duplicate_rows_removed: int
    conflicting_duplicate_groups: int
    identity_like_rows_sanitized: int
    class_counts: dict[str, int]
    content_fingerprint: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    seed: int
    ratios: dict[str, float]

    def manifest(self) -> dict[str, Any]:
        partitions: dict[str, Any] = {}
        for name, frame in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            partitions[name] = {
                "row_count": int(len(frame)),
                "class_counts": {
                    str(key): int(value)
                    for key, value in frame["target_role"].value_counts().sort_index().items()
                },
                "record_ids": frame["record_id"].astype(str).tolist(),
                "fingerprints": frame["fingerprint"].astype(str).tolist(),
            }
        return {"seed": self.seed, "ratios": self.ratios, "partitions": partitions}


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _fingerprint(value: str) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    joined = "\n".join(
        f"{row.fingerprint}:{row.target_role}"
        for row in frame.sort_values("fingerprint").itertuples()
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_and_validate_dataset(
    path: str | Path,
    *,
    text_column: str = "resume_text",
    label_column: str = "target_role",
    minimum_samples_per_class: int = 8,
) -> tuple[pd.DataFrame, DatasetAudit]:
    """Load a CSV, remove unusable rows, and block obvious leakage risks."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {source}")
    if source.suffix.lower() != ".csv":
        raise DatasetValidationError("Phase 3 currently accepts labelled CSV datasets only.")

    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing_columns = [name for name in (text_column, label_column) if name not in frame]
    if missing_columns:
        raise DatasetValidationError(
            "Dataset is missing required column(s): " + ", ".join(missing_columns)
        )

    total_rows = int(len(frame))
    working = frame.copy()
    working[text_column] = working[text_column].astype(str).str.strip()
    working[label_column] = (
        working[label_column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    usable_mask = working[text_column].ne("") & working[label_column].ne("")
    missing_rows_removed = int((~usable_mask).sum())
    working = working.loc[usable_mask].copy()
    identity_like_rows = int(working[text_column].map(contains_contact_information).sum())
    working[text_column] = working[text_column].map(sanitize_model_text)
    empty_after_sanitization = working[text_column].eq("")
    missing_rows_removed += int(empty_after_sanitization.sum())
    working = working.loc[~empty_after_sanitization].copy()
    working["fingerprint"] = working[text_column].map(_fingerprint)

    conflicts = (
        working.groupby("fingerprint")[label_column].nunique().loc[lambda series: series > 1]
    )
    if not conflicts.empty:
        raise DatasetValidationError(
            f"Found {len(conflicts)} duplicate resume text group(s) with conflicting labels. "
            "Resolve them before training."
        )

    before_deduplication = len(working)
    working = working.drop_duplicates(subset=["fingerprint"], keep="first").copy()
    duplicate_rows_removed = int(before_deduplication - len(working))

    if "record_id" not in working:
        working["record_id"] = [f"record_{index + 1:05d}" for index in range(len(working))]
    else:
        working["record_id"] = working["record_id"].astype(str).str.strip()
        empty_ids = working["record_id"].eq("")
        replacements = [f"record_{index + 1:05d}" for index in range(int(empty_ids.sum()))]
        working.loc[empty_ids, "record_id"] = replacements

    if working["record_id"].duplicated().any():
        raise DatasetValidationError("record_id values must be unique after duplicate removal.")

    working = working.rename(columns={text_column: "resume_text", label_column: "target_role"})
    class_counts = {
        str(key): int(value)
        for key, value in working["target_role"].value_counts().sort_index().items()
    }
    if len(class_counts) < 2:
        raise DatasetValidationError("At least two target-role classes are required.")
    too_small = {
        label: count for label, count in class_counts.items() if count < minimum_samples_per_class
    }
    if too_small:
        details = ", ".join(f"{label}={count}" for label, count in too_small.items())
        raise DatasetValidationError(
            f"Each class needs at least {minimum_samples_per_class} unique samples; found {details}."
        )

    warnings: list[str] = []
    largest = max(class_counts.values())
    smallest = min(class_counts.values())
    if largest / smallest >= 3:
        warnings.append(
            "Class imbalance is at least 3:1; inspect macro-F1 and per-class recall."
        )
    if duplicate_rows_removed:
        warnings.append(
            f"Removed {duplicate_rows_removed} exact/normalized duplicate row(s) before splitting."
        )
    if missing_rows_removed:
        warnings.append(f"Removed {missing_rows_removed} row(s) with missing text or labels.")
    if identity_like_rows:
        warnings.append(
            f"Removed email, phone, or URL patterns from {identity_like_rows} row(s) before training."
        )

    working = working[["record_id", "resume_text", "target_role", "fingerprint"]].reset_index(
        drop=True
    )
    audit = DatasetAudit(
        source_file=str(source),
        text_column=text_column,
        label_column=label_column,
        total_rows=total_rows,
        usable_rows=int(len(working)),
        missing_rows_removed=missing_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        conflicting_duplicate_groups=0,
        identity_like_rows_sanitized=identity_like_rows,
        class_counts=class_counts,
        content_fingerprint=_dataset_fingerprint(working),
        warnings=warnings,
    )
    return working, audit


def audit_dataset(
    path: str | Path,
    *,
    text_column: str = "resume_text",
    label_column: str = "target_role",
) -> DatasetAudit:
    _, audit = load_and_validate_dataset(
        path,
        text_column=text_column,
        label_column=label_column,
    )
    return audit


def split_dataset(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> DatasetSplits:
    """Create deterministic stratified splits after deduplication."""

    if seed < 0:
        raise DatasetValidationError("The random seed must be non-negative.")
    if any(ratio <= 0 for ratio in (train_ratio, validation_ratio, test_ratio)):
        raise DatasetValidationError("Every split ratio must be greater than zero.")
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise DatasetValidationError("Train, validation, and test ratios must sum to 1.0.")

    try:
        train, temporary = train_test_split(
            frame,
            test_size=validation_ratio + test_ratio,
            random_state=seed,
            stratify=frame["target_role"],
        )
        relative_test_ratio = test_ratio / (validation_ratio + test_ratio)
        validation, test = train_test_split(
            temporary,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=temporary["target_role"],
        )
    except ValueError as exc:
        raise DatasetValidationError(
            "Could not create stratified splits. Add more unique samples per class or adjust ratios: "
            f"{exc}"
        ) from exc

    result = DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        seed=seed,
        ratios={"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
    )
    fingerprint_sets = {
        name: set(partition["fingerprint"])
        for name, partition in (
            ("train", result.train),
            ("validation", result.validation),
            ("test", result.test),
        )
    }
    if (
        fingerprint_sets["train"] & fingerprint_sets["validation"]
        or fingerprint_sets["train"] & fingerprint_sets["test"]
        or fingerprint_sets["validation"] & fingerprint_sets["test"]
    ):
        raise DatasetValidationError("A resume fingerprint appears in more than one split.")
    return result


def read_dataset_metadata(dataset_path: str | Path) -> dict[str, Any]:
    """Read optional provenance metadata stored beside a dataset."""

    source = Path(dataset_path)
    candidates = (
        source.with_suffix(".manifest.json"),
        source.with_name(f"{source.stem}_manifest.json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(f"Invalid dataset manifest {candidate}: {exc}") from exc
    return {
        "source_type": "unknown",
        "warning": "No dataset provenance manifest was found. Document the source before reporting metrics.",
    }
