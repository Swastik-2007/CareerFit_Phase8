"""Command-line interface for CareerFit Phases 1 through 6."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from careerfit.extraction import build_candidate_profile
from careerfit.fairness import audit_fairness
from careerfit.growth import SkillGraph, SkillGraphError, analyze_growth
from careerfit.job import build_job_requirements
from careerfit.matching import rank_candidates
from careerfit.ml import (
    DatasetValidationError,
    audit_dataset,
    candidate_model_text,
    load_model_bundle,
    predict_text,
    train_and_evaluate,
)
from careerfit.parsing.document_parser import SUPPORTED_EXTENSIONS, DocumentParseError
from careerfit.scoring import build_scoring_context, score_candidates_with_context


def _add_common_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", help="Optional JSON output path.")
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include cleaned source text in JSON (off by default).",
    )
    parser.add_argument(
        "--taxonomy",
        help="Optional path to a custom skills taxonomy JSON file.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="careerfit",
        description="Parse resumes and explain candidate-to-job requirement matches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse one or more resumes (Phase 1 compatible command).",
    )
    parse_parser.add_argument("inputs", nargs="+", help="Resume files or directories.")
    _add_common_output_options(parse_parser)

    job_parser = subparsers.add_parser("parse-job", help="Parse one job description.")
    job_parser.add_argument("job", help="PDF, DOCX, or TXT job description.")
    _add_common_output_options(job_parser)

    match_parser = subparsers.add_parser(
        "match",
        help="Rank resumes against one parsed job description.",
    )
    match_parser.add_argument("--job", required=True, help="Job-description file.")
    match_parser.add_argument(
        "--resumes",
        nargs="+",
        required=True,
        help="Resume files or directories.",
    )
    _add_common_output_options(match_parser)

    audit_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a labelled resume CSV before model training.",
    )
    audit_parser.add_argument("dataset", help="CSV containing resume_text and target_role.")
    audit_parser.add_argument("-o", "--output", help="Optional JSON audit output path.")
    audit_parser.add_argument("--text-column", default="resume_text")
    audit_parser.add_argument("--label-column", default="target_role")

    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate Logistic Regression, SVM, and Random Forest.",
    )
    train_parser.add_argument("--dataset", required=True, help="Labelled resume CSV.")
    train_parser.add_argument("--artifacts", required=True, help="Model/evaluation directory.")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--train-ratio", type=float, default=0.70)
    train_parser.add_argument("--validation-ratio", type=float, default=0.15)
    train_parser.add_argument("--test-ratio", type=float, default=0.15)
    train_parser.add_argument("--text-column", default="resume_text")
    train_parser.add_argument("--label-column", default="target_role")
    train_parser.add_argument("-o", "--output", help="Optional JSON training summary path.")

    predict_parser = subparsers.add_parser(
        "predict-role",
        help="Predict target roles for resumes using a saved model.",
    )
    predict_parser.add_argument("--model", required=True, help="Saved Phase 3 .joblib file.")
    predict_parser.add_argument("--resumes", nargs="+", required=True)
    _add_common_output_options(predict_parser)

    score_parser = subparsers.add_parser(
        "score",
        help="Run hybrid ML plus requirement scoring for a job and resumes.",
    )
    score_parser.add_argument("--model", required=True, help="Saved Phase 3 .joblib file.")
    score_parser.add_argument("--job", required=True, help="Job-description file.")
    score_parser.add_argument("--resumes", nargs="+", required=True)
    score_parser.add_argument(
        "--target-role",
        help="Optional exact model role; otherwise CareerFit infers it from the job.",
    )
    score_parser.add_argument("--similarity-weight", type=float, default=0.40)
    score_parser.add_argument("--role-weight", type=float, default=0.40)
    score_parser.add_argument("--requirement-weight", type=float, default=0.20)
    _add_common_output_options(score_parser)

    growth_parser = subparsers.add_parser(
        "growth",
        help="Recommend graph-backed learning paths and simulate conservative score gains.",
    )
    growth_parser.add_argument("--model", required=True, help="Saved Phase 3+ .joblib file.")
    growth_parser.add_argument("--job", required=True, help="Job-description file.")
    growth_parser.add_argument("--resumes", nargs="+", required=True)
    growth_parser.add_argument("--graph", help="Optional custom skill-graph JSON file.")
    growth_parser.add_argument("--max-recommendations", type=int, default=5)
    growth_parser.add_argument("--target-role")
    growth_parser.add_argument("--similarity-weight", type=float, default=0.40)
    growth_parser.add_argument("--role-weight", type=float, default=0.40)
    growth_parser.add_argument("--requirement-weight", type=float, default=0.20)
    _add_common_output_options(growth_parser)

    fairness_parser = subparsers.add_parser(
        "fairness-audit",
        help="Compare standard and identity-masked scores and flag sensitivity for review.",
    )
    fairness_parser.add_argument("--model", required=True, help="Saved Phase 3+ .joblib file.")
    fairness_parser.add_argument("--job", required=True, help="Job-description file.")
    fairness_parser.add_argument("--resumes", nargs="+", required=True)
    fairness_parser.add_argument("--target-role")
    fairness_parser.add_argument("--similarity-weight", type=float, default=0.40)
    fairness_parser.add_argument("--role-weight", type=float, default=0.40)
    fairness_parser.add_argument("--requirement-weight", type=float, default=0.20)
    fairness_parser.add_argument(
        "--minimum-review-threshold",
        type=float,
        default=0.01,
        help="Minimum absolute fit-score change that can trigger human review.",
    )
    fairness_parser.add_argument(
        "--review-threshold",
        type=float,
        help="Optional reviewed threshold override on the 0-to-1 score scale.",
    )
    fairness_parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="Multiplier for the robust median absolute-deviation threshold.",
    )
    _add_common_output_options(fairness_parser)
    return parser


def _collect_files(values: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            files.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            files.append(path)
    return list(dict.fromkeys(path.resolve() for path in files))


def _render(payload: dict[str, Any], output: str | None, success_message: str) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"{success_message} {output_path}")
    else:
        print(rendered)


def _options(args: argparse.Namespace) -> dict[str, str]:
    taxonomy = getattr(args, "taxonomy", None)
    return {"taxonomy_path": taxonomy} if taxonomy else {}


def _parse_candidates(
    paths: list[Path],
    args: argparse.Namespace,
) -> tuple[list[Any], list[dict[str, str]]]:
    candidates = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            candidates.append(build_candidate_profile(path, **_options(args)))
        except (DocumentParseError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"source_file": str(path), "error": str(exc)})
    return candidates, errors


def _parse_command(args: argparse.Namespace) -> int:
    paths = _collect_files(args.inputs)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2

    candidates, errors = _parse_candidates(paths, args)
    payload = {
        "careerfit_phase": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "error_count": len(errors),
        "candidates": [
            candidate.to_dict(include_text=args.include_text) for candidate in candidates
        ],
        "errors": errors,
    }
    _render(
        payload,
        args.output,
        f"Wrote {len(candidates)} candidate profile(s) to",
    )
    return 0 if not errors else 1


def _parse_job_command(args: argparse.Namespace) -> int:
    try:
        job = build_job_requirements(args.job, **_options(args))
    except (DocumentParseError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not parse job description: {exc}", file=sys.stderr)
        return 1

    payload = {
        "careerfit_phase": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_requirements": job.to_dict(include_text=args.include_text),
    }
    _render(payload, args.output, "Wrote job requirements to")
    return 0


def _match_command(args: argparse.Namespace) -> int:
    try:
        job = build_job_requirements(args.job, **_options(args))
    except (DocumentParseError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not parse job description: {exc}", file=sys.stderr)
        return 1

    paths = _collect_files(args.resumes)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2

    candidates, errors = _parse_candidates(paths, args)
    ranking = rank_candidates(candidates, job)
    payload = {
        "careerfit_phase": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_requirements": job.to_dict(include_text=args.include_text),
        "candidate_count": len(candidates),
        "eligible_count": sum(result.eligible for result in ranking),
        "error_count": len(errors),
        "ranking": [result.to_dict() for result in ranking],
        "errors": errors,
    }
    _render(payload, args.output, f"Ranked {len(ranking)} candidate(s) and wrote results to")
    return 0 if not errors else 1


def _validate_dataset_command(args: argparse.Namespace) -> int:
    try:
        audit = audit_dataset(
            args.dataset,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    except DatasetValidationError as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1
    payload = {"careerfit_phase": 3, "valid": True, "dataset_audit": audit.to_dict()}
    _render(payload, args.output, "Wrote dataset audit to")
    return 0


def _train_command(args: argparse.Namespace) -> int:
    try:
        summary = train_and_evaluate(
            args.dataset,
            args.artifacts,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    except (DatasetValidationError, OSError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    _render(summary, args.output, "Wrote training summary to")
    return 0


def _predict_role_command(args: argparse.Namespace) -> int:
    try:
        bundle = load_model_bundle(args.model)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Could not load model: {exc}", file=sys.stderr)
        return 1
    paths = _collect_files(args.resumes)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2
    candidates, errors = _parse_candidates(paths, args)
    predictions = []
    for candidate in candidates:
        predictions.append(
            {
                "candidate_id": candidate.candidate_id,
                "source_file": candidate.source_file,
                "candidate_name": candidate.identity.name,
                **predict_text(bundle, candidate_model_text(candidate)),
            }
        )
    payload = {
        "careerfit_phase": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "predictions": predictions,
        "errors": errors,
    }
    _render(payload, args.output, f"Predicted roles for {len(predictions)} resume(s) and wrote results to")
    return 0 if not errors else 1


def _score_command(args: argparse.Namespace) -> int:
    try:
        bundle = load_model_bundle(args.model)
        job = build_job_requirements(args.job, **_options(args))
    except (DocumentParseError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not prepare hybrid scoring: {exc}", file=sys.stderr)
        return 1
    paths = _collect_files(args.resumes)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2
    candidates, errors = _parse_candidates(paths, args)
    weights = {
        "text_similarity": args.similarity_weight,
        "role_probability": args.role_weight,
        "requirement_score": args.requirement_weight,
    }
    try:
        context = build_scoring_context(
            job,
            bundle,
            target_role_override=args.target_role,
            weights=weights,
        )
        ranking = score_candidates_with_context(candidates, job, bundle, context)
    except ValueError as exc:
        print(f"Could not calculate hybrid scores: {exc}", file=sys.stderr)
        return 1
    payload = {
        "careerfit_phase": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_requirements": job.to_dict(include_text=args.include_text),
        "model_target_role": context.target_role,
        "scoring_context": context.to_dict(),
        "candidate_count": len(candidates),
        "eligible_count": sum(result.eligible for result in ranking),
        "ranking": [result.to_dict() for result in ranking],
        "errors": errors,
    }
    _render(payload, args.output, f"Scored {len(ranking)} candidate(s) and wrote results to")
    return 0 if not errors else 1


def _growth_command(args: argparse.Namespace) -> int:
    try:
        bundle = load_model_bundle(args.model)
        job = build_job_requirements(args.job, **_options(args))
        graph = SkillGraph(args.graph) if args.graph else SkillGraph()
    except (
        DocumentParseError,
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        SkillGraphError,
    ) as exc:
        print(f"Could not prepare growth analysis: {exc}", file=sys.stderr)
        return 1

    paths = _collect_files(args.resumes)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2
    candidates, errors = _parse_candidates(paths, args)
    weights = {
        "text_similarity": args.similarity_weight,
        "role_probability": args.role_weight,
        "requirement_score": args.requirement_weight,
    }
    try:
        context = build_scoring_context(
            job,
            bundle,
            target_role_override=args.target_role,
            weights=weights,
        )
        growth_results = analyze_growth(
            candidates,
            job,
            bundle,
            context,
            graph,
            max_recommendations=args.max_recommendations,
        )
    except ValueError as exc:
        print(f"Could not calculate growth plans: {exc}", file=sys.stderr)
        return 1

    payload = {
        "careerfit_phase": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_requirements": job.to_dict(include_text=args.include_text),
        "scoring_context": context.to_dict(),
        "skill_graph": graph.metadata(),
        "candidate_count": len(candidates),
        "eligible_now_count": sum(result.current_eligible for result in growth_results),
        "eligible_after_plan_count": sum(
            result.eligible_after_plan for result in growth_results
        ),
        "growth_results": [result.to_dict() for result in growth_results],
        "errors": errors,
    }
    _render(
        payload,
        args.output,
        f"Generated {len(growth_results)} candidate growth plan(s) and wrote results to",
    )
    return 0 if not errors else 1


def _fairness_audit_command(args: argparse.Namespace) -> int:
    try:
        bundle = load_model_bundle(args.model)
        job = build_job_requirements(args.job, **_options(args))
    except (
        DocumentParseError,
        FileNotFoundError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"Could not prepare fairness audit: {exc}", file=sys.stderr)
        return 1

    paths = _collect_files(args.resumes)
    if not paths:
        print("No supported resume files were found.", file=sys.stderr)
        return 2
    candidates, errors = _parse_candidates(paths, args)
    weights = {
        "text_similarity": args.similarity_weight,
        "role_probability": args.role_weight,
        "requirement_score": args.requirement_weight,
    }
    try:
        context = build_scoring_context(
            job,
            bundle,
            target_role_override=args.target_role,
            weights=weights,
        )
        audit = audit_fairness(
            candidates,
            job,
            bundle,
            context,
            minimum_threshold=args.minimum_review_threshold,
            threshold_override=args.review_threshold,
            mad_multiplier=args.mad_multiplier,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"Could not calculate fairness audit: {exc}", file=sys.stderr)
        return 1

    audit_payload = audit.to_dict()
    payload = {
        "careerfit_phase": 6,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_requirements": job.to_dict(include_text=args.include_text),
        "scoring_context": context.to_dict(),
        "candidate_count": len(candidates),
        **audit_payload,
        "errors": errors,
    }
    _render(
        payload,
        args.output,
        f"Audited {len(audit.results)} candidate(s) and wrote fairness results to",
    )
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "parse":
        return _parse_command(args)
    if args.command == "parse-job":
        return _parse_job_command(args)
    if args.command == "match":
        return _match_command(args)
    if args.command == "validate-dataset":
        return _validate_dataset_command(args)
    if args.command == "train":
        return _train_command(args)
    if args.command == "predict-role":
        return _predict_role_command(args)
    if args.command == "score":
        return _score_command(args)
    if args.command == "growth":
        return _growth_command(args)
    if args.command == "fairness-audit":
        return _fairness_audit_command(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
