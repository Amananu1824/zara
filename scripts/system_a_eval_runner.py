from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "app"
    / "knowledge"
    / "system_a_eval_cases.jsonl"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "system_a_evaluations"


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))

    from app.improvement.observer import (
        _ALLOWED_ISSUES,
        _ALLOWED_RULES,
    )

    allowed_tool_fields = {
        "tool",
        "tool_name",
        "name",
        "operation",
        "status",
        "error_code",
        "code",
        "requires_confirmation",
        "call_id",
    }

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid JSON."
                ) from exc

            if not isinstance(case, dict):
                raise ValueError(
                    f"Line {line_number}: expected JSON object."
                )

            case_id = str(case.get("id") or "").strip()

            if not case_id or case_id.casefold() in seen:
                raise ValueError(
                    f"Line {line_number}: invalid or duplicate ID."
                )

            for required_field in (
                "user_message",
                "assistant_answer",
            ):
                value = case.get(required_field)

                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{case_id}: {required_field} is required."
                    )

            for field in (
                "expected_issue_codes",
                "expected_rule_keys",
            ):
                values = case.get(field, [])

                if (
                    not isinstance(values, list)
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in values
                    )
                ):
                    raise ValueError(
                        f"{case_id}: {field} must be a list of strings."
                    )

                if len(values) != len(set(values)):
                    raise ValueError(
                        f"{case_id}: duplicate values in {field}."
                    )

            invalid_issues = (
                set(case.get("expected_issue_codes", []))
                - _ALLOWED_ISSUES
            )

            invalid_rules = (
                set(case.get("expected_rule_keys", []))
                - set(_ALLOWED_RULES)
            )

            if invalid_issues or invalid_rules:
                raise ValueError(
                    f"{case_id}: unsupported evaluation labels."
                )

            for field in (
                "decision",
                "retrieval_summary",
                "answer_repair_summary",
            ):
                if not isinstance(case.get(field, {}), dict):
                    raise ValueError(
                        f"{case_id}: {field} must be an object."
                    )

            if not isinstance(case.get("language_mode", "hinglish"), str):
                raise ValueError(
                    f"{case_id}: language_mode must be a string."
                )

            if not isinstance(case.get("output_constraint", "normal"), str):
                raise ValueError(
                    f"{case_id}: output_constraint must be a string."
                )

            response_policy_ids = case.get("response_policy_ids", [])

            if (
                not isinstance(response_policy_ids, list)
                or any(
                    not isinstance(value, str)
                    for value in response_policy_ids
                )
            ):
                raise ValueError(
                    f"{case_id}: response_policy_ids must be strings."
                )

            history = case.get("review_context", {})

            if not isinstance(history, dict):
                raise ValueError(
                    f"{case_id}: review_context must be an object."
                )

            for key, value in history.items():
                if key == "tool_trace":
                    continue

                if (
                    not isinstance(value, list)
                    or any(not isinstance(item, str) for item in value)
                ):
                    raise ValueError(
                        f"{case_id}: review_context.{key} "
                        "must be a list of strings."
                    )

            tool_trace = case.get(
                "tool_trace",
                history.get("tool_trace", []),
            )

            if not isinstance(tool_trace, list) or len(tool_trace) > 20:
                raise ValueError(
                    f"{case_id}: tool_trace must be a bounded list."
                )

            for trace_index, trace_item in enumerate(tool_trace):
                if not isinstance(trace_item, dict):
                    raise ValueError(
                        f"{case_id}: tool_trace item "
                        f"{trace_index} must be an object."
                    )

                unknown_fields = (
                    set(trace_item) - allowed_tool_fields
                )

                if unknown_fields:
                    raise ValueError(
                        f"{case_id}: unsupported tool_trace fields."
                    )

                if any(
                    not isinstance(
                        value,
                        (str, int, float, bool, type(None)),
                    )
                    for value in trace_item.values()
                ):
                    raise ValueError(
                        f"{case_id}: tool_trace values must be scalar."
                    )

            # Normalize nested legacy tool traces into one canonical field.
            case["tool_trace"] = tool_trace
            case["review_context"] = {
                key: value
                for key, value in history.items()
                if key != "tool_trace"
            }

            seen.add(case_id)
            cases.append(case)

    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    return cases

DETERMINISTIC_ISSUE_MAP = {
    "roman_script": "roman_hinglish",
    "repeated_sentence": "repetition",
    "feminine_agreement": "feminine_agreement",
    "generic_follow_up": "unnecessary_question",
    "physical_experience_claim": "physical_experience_claim",
    "unnecessary_ai_disclosure": "unnecessary_ai_disclosure",
}


def extract_deterministic_issues(
    review: dict[str, Any],
) -> set[str]:
    predicted = set()

    for item in review.get("checks", []):
        if not isinstance(item, dict):
            continue

        if item.get("result") not in {"fail", "review_needed"}:
            continue

        code = DETERMINISTIC_ISSUE_MAP.get(
            str(item.get("code") or "")
        )

        if code:
            predicted.add(code)

    return predicted

def score_sets(
    expected: set[str],
    predicted: set[str],
) -> dict[str, Any]:

    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_static_acceptance() -> dict[str, Any]:

    from app.improvement.observer import (
        _ALLOWED_RULES,
        _AUTO_PROMOTION_ALLOWED_RULES,
        _AUTO_PROMOTION_RISK_ORDER,
        _VALIDATION_THRESHOLDS,
        _lesson_conflict_signal,
        _deterministic_review,
    )

    from app.improvement_repair.answer_repair import (
        AnswerRepairer,
    )

    from app.improvement_repair.code_repair import (
        CodeRepairError,
        CodeRepairService,
        RULE_REPAIR_TARGETS,
    )

    checks = {}

    # A1/A2: rule and issue definitions.
    checks["rule_registry"] = (
        bool(_ALLOWED_RULES)
        and all(
            isinstance(key, str) and isinstance(value, tuple)
            for key, value in _ALLOWED_RULES.items()
        )
    )

    # A3: thresholds become stricter with increasing risk.
    checks["validation_threshold_order"] = (
        _VALIDATION_THRESHOLDS["low"]["min_evidence"]
        < _VALIDATION_THRESHOLDS["medium"]["min_evidence"]
        < _VALIDATION_THRESHOLDS["high"]["min_evidence"]
        and
        _VALIDATION_THRESHOLDS["low"]["min_confidence"]
        < _VALIDATION_THRESHOLDS["medium"]["min_confidence"]
        < _VALIDATION_THRESHOLDS["high"]["min_confidence"]
    )

    checks["contradiction_detection"] = (
        _lesson_conflict_signal(
            "Keep answers concise and direct.",
            "Do not keep answers concise and direct.",
        ) is not None
        and
        _lesson_conflict_signal(
            "Use Roman Hindi when requested.",
            "Preserve useful English technical terminology.",
        ) is None
    )

    # A4: automatic policy cannot include high-risk rules.
    checks["automatic_promotion_allowlist"] = (
        _AUTO_PROMOTION_ALLOWED_RULES.issubset(_ALLOWED_RULES)
        and all(
            _AUTO_PROMOTION_RISK_ORDER[
                _ALLOWED_RULES[rule][1]
            ] <= _AUTO_PROMOTION_RISK_ORDER["medium"]
            for rule in _AUTO_PROMOTION_ALLOWED_RULES
        )
    )

    # A5: deterministic checks must identify script violations.
    deterministic = _deterministic_review(
        "Roman Hinglish mein jawab do.",
        "मैं आपकी बात समझ रही हूँ।",
        "hinglish",
        {},
    )

    checks["answer_repair_script_detection"] = any(
        item.get("code") == "roman_script"
        and item.get("result") == "fail"
        for item in deterministic.get("checks", [])
    )

    # A5: fallback must preserve original answer.
    repairer = AnswerRepairer(object())

    fallback = repairer.skip(
        "Original answer remains unchanged.",
        reason_code="evaluation_fallback",
    )

    checks["answer_repair_fallback"] = (
        fallback.original_answer
        == fallback.final_answer
        and fallback.trace.get("status") == "skipped"
    )

    # A6: supported rule targets must be within the allowlist.
    checks["code_repair_rule_registry"] = (
        set(RULE_REPAIR_TARGETS).issubset(_ALLOWED_RULES)
    )

    safe_original = (
        "def preserve_existing_behavior():\n"
        "    value = 'old'\n"
        "    result = value.strip()\n"
        "    return result\n"
    )

    safe_proposed = (
        "def preserve_existing_behavior():\n"
        "    value = 'updated'\n"
        "    result = value.strip()\n"
        "    return result\n"
    )

    try:
        CodeRepairService._validate_source(
            safe_original,
            safe_proposed,
            "response_rules.py",
        )
        safe_patch_accepted = True

    except CodeRepairError:
        safe_patch_accepted = False

    checks["code_repair_safe_patch"] = safe_patch_accepted

    try:
        CodeRepairService._validate_source(
            safe_original,
            "import subprocess\n"
            + safe_proposed,
            "response_rules.py",
        )
        dangerous_import_rejected = False
    except CodeRepairError:
        dangerous_import_rejected = True

    checks["code_repair_dangerous_import_rejected"] = (
        dangerous_import_rejected
    )

    try:
        CodeRepairService._validate_source(
            safe_original,
            "def preserve_existing_behavior(\n",
            "response_rules.py",
        )
        malformed_source_rejected = False
    except CodeRepairError:
        malformed_source_rejected = True

    checks["code_repair_invalid_syntax_rejected"] = (
        malformed_source_rejected
    )

    failed_check_names = [
        name
        for name, result in checks.items()
        if not result
    ]

    passed = sum(bool(value) for value in checks.values())

    return {
        "checks": checks,
        "passed": passed,
        "failed": len(checks) - passed,
        "total": len(checks),
        "status": "passed" if passed == len(checks) else "failed",
        "failed_checks": failed_check_names,
    }


def run_model_evaluation(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:

    from app.ai.ollama_client import OllamaClient

    from app.improvement.observer import (
        _run_local_review,
        _deterministic_review,
    )

    client = OllamaClient()

    issue_totals = Counter()
    model_issue_totals = Counter()
    rule_totals = Counter()
    case_results = []
    calibration_errors = []

    try:

        for case in cases:

            user_message = case["user_message"]
            answer = case["assistant_answer"]

            decision = case.get("decision", {})
            history = case.get("review_context", {})

            deterministic = _deterministic_review(
                user_message,
                answer,
                case.get("language_mode", "hinglish"),
                decision,
            )

            job = {
                "user_message": user_message,
                "answer": answer,
                "language_mode": case.get(
                    "language_mode",
                    "hinglish",
                ),
                "output_constraint": case.get(
                    "output_constraint",
                    "normal",
                ),
                "decision": decision,
                "response_policy_ids": case.get(
                    "response_policy_ids",
                    [],
                ),
                "retrieval_summary": case.get(
                    "retrieval_summary",
                    {},
                ),
                "tool_trace": case.get(
                    "tool_trace",
                    [],
                ),
                "review_context": history,
                "deterministic_review": deterministic,
                "answer_repair_summary": case.get(
                    "answer_repair_summary",
                    {},
                ),
            }

            try:
                report, candidates = _run_local_review(
                    client,
                    job,
                )

                model_predicted_issues = {
                    item["code"]
                    for item in report.get("issues", [])
                }

                deterministic_predicted_issues = (
                    extract_deterministic_issues(deterministic)
                )

                # Combined system-level issue classification.
                predicted_issues = (
                    model_predicted_issues
                    | deterministic_predicted_issues
                )

                expected_issues = set(
                    case.get("expected_issue_codes", [])
                )

                predicted_rules = {
                    item["rule_key"]
                    for item in candidates
                }

                expected_rules = set(
                    case.get("expected_rule_keys", [])
                )

                model_only_scores = score_sets(
                    expected_issues,
                    model_predicted_issues,
                )

                model_issue_totals.update({
                    "tp": model_only_scores["true_positive"],
                    "fp": model_only_scores["false_positive"],
                    "fn": model_only_scores["false_negative"],
                })

                issue_scores = score_sets(
                    expected_issues,
                    predicted_issues,
                )

                rule_scores = score_sets(
                    expected_rules,
                    predicted_rules,
                )

                issue_totals.update({
                    "tp": issue_scores["true_positive"],
                    "fp": issue_scores["false_positive"],
                    "fn": issue_scores["false_negative"],
                })

                rule_totals.update({
                    "tp": rule_scores["true_positive"],
                    "fp": rule_scores["false_positive"],
                    "fn": rule_scores["false_negative"],
                })

                confidence_by_code = {
                    item["code"]: float(item["confidence"])
                    for item in report.get("issues", [])
                }

                for code in expected_issues | model_predicted_issues:

                    probability = confidence_by_code.get(
                        code,
                        0.0,
                    )

                    target = 1.0 if code in expected_issues else 0.0

                    calibration_errors.append(
                        (probability - target) ** 2
                    )

                case_results.append({
                    "id": case["id"],
                    "status": report.get("overall"),
                    "expected_issue_codes": sorted(expected_issues),
                    "predicted_issue_codes": sorted(predicted_issues),
                    "model_only_issue_codes": sorted(
                        model_predicted_issues
                    ),
                    "deterministic_issue_codes": sorted(
                        deterministic_predicted_issues
                    ),
                    "expected_rule_keys": sorted(expected_rules),
                    "predicted_rule_keys": sorted(predicted_rules),
                    "issue_scores": issue_scores,
                    "rule_scores": rule_scores,
                })

            except Exception as exc:
                case_results.append({
                    "id": case["id"],
                    "status": "error",
                    "error_code": type(exc).__name__,
                })

    finally:
        try:
            client.session.close()
        except Exception:
            pass

    issue_metrics = score_sets(
        {"tp"} if issue_totals["tp"] else set(),
        set(),
    )

    tp = issue_totals["tp"]
    fp = issue_totals["fp"]
    fn = issue_totals["fn"]

    issue_precision = (
        tp / (tp + fp) if tp + fp else 1.0
    )

    issue_recall = (
        tp / (tp + fn) if tp + fn else 1.0
    )

    issue_f1 = (
        2 * issue_precision * issue_recall
        / (issue_precision + issue_recall)
        if issue_precision + issue_recall
        else 0.0
    )



    rtp = rule_totals["tp"]
    rfp = rule_totals["fp"]
    rfn = rule_totals["fn"]

    rule_precision = (
        rtp / (rtp + rfp) if rtp + rfp else 1.0
    )

    rule_recall = (
        rtp / (rtp + rfn) if rtp + rfn else 1.0
    )

    rule_f1 = (
        2 * rule_precision * rule_recall
        / (rule_precision + rule_recall)
        if rule_precision + rule_recall
        else 0.0
    )

    brier = (
        sum(calibration_errors) / len(calibration_errors)
        if calibration_errors else None
    )

    return {
        "total_cases": len(cases),
        "completed_cases": sum(
            item.get("status") != "error"
            for item in case_results
        ),
        "error_cases": sum(
            item.get("status") == "error"
            for item in case_results
        ),
        "issue_classification": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(issue_precision, 4),
            "recall": round(issue_recall, 4),
            "f1": round(issue_f1, 4),
        },

        "model_only_issue_classification": {
            "tp": model_issue_totals["tp"],
            "fp": model_issue_totals["fp"],
            "fn": model_issue_totals["fn"],
            "precision": round(
                model_issue_totals["tp"] /
                max(
                    1,
                    model_issue_totals["tp"]
                    + model_issue_totals["fp"],
                ),
                4,
            ),
            "recall": round(
                model_issue_totals["tp"] /
                max(
                    1,
                    model_issue_totals["tp"]
                    + model_issue_totals["fn"],
                ),
                4,
            ),
            "f1": round(
                (
                    2 * model_issue_totals["tp"]
                ) / max(
                    1,
                    (
                        2 * model_issue_totals["tp"]
                        + model_issue_totals["fp"]
                        + model_issue_totals["fn"]
                    ),
                ),
                4,
            ),
        },
        
        "lesson_extraction": {
            "tp": rtp,
            "fp": rfp,
            "fn": rfn,
            "precision": round(rule_precision, 4),
            "recall": round(rule_recall, 4),
            "f1": round(rule_f1, 4),
        },
        "confidence_calibration": {
            "brier_score": (
                round(brier, 4)
                if brier is not None
                else None
            ),
            "interpretation": (
                "Lower is better; compare across repeat runs."
            ),
        },
        "cases": case_results,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except ValueError:
            return {}

    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except ValueError:
            return []

    return []


def run_read_only_database_audit() -> dict[str, Any]:

    import psycopg
    from psycopg.rows import dict_row

    from app.core.config import settings

    connection = psycopg.connect(
        host=settings.zara_database_host,
        port=settings.zara_database_port,
        dbname=settings.zara_database_name,
        user=settings.zara_database_user,
        password=settings.zara_database_password,
        connect_timeout=settings.zara_database_connect_timeout,
        options=(
            "-c statement_timeout="
            f"{settings.zara_database_statement_timeout_ms}"
        ),
        row_factory=dict_row,
    )

    report: dict[str, Any] = {
        "read_only": True,
        "tables": {},
        "violations": [],
    }

    try:
        with connection:

            connection.execute("SET TRANSACTION READ ONLY")

            expected_tables = [
                "zara_improvement_turn_traces",
                "zara_lesson_candidates",
                "zara_production_lessons",
                "zara_lesson_promotion_events",
                "zara_lesson_promotion_authorizations",
                "zara_code_repair_proposals",
                "zara_code_repair_events",
            ]

            for table in expected_tables:
                row = connection.execute(
                    "SELECT to_regclass(%s) AS table_name",
                    (table,),
                ).fetchone()

                report["tables"][table] = bool(
                    row and row["table_name"]
                )

            report["missing_tables"] = [
                table
                for table, exists in report["tables"].items()
                if not exists
            ]

            report["missing_tables"] = [
                table
                for table, exists in report["tables"].items()
                if not exists
            ]

            # A1: review processing health.
            if report["tables"]["zara_improvement_turn_traces"]:
                rows = connection.execute("""
                    SELECT review_status, COUNT(*) AS total
                    FROM zara_improvement_turn_traces
                    GROUP BY review_status
                """).fetchall()

                report["review_status_counts"] = {
                    str(row["review_status"]): int(row["total"])
                    for row in rows
                }

                repair_rows = connection.execute("""
                    SELECT
                        COALESCE(
                            context_summary #>> '{answer_repair,status}',
                            'missing'
                        ) AS repair_status,
                        COUNT(*) AS total
                    FROM zara_improvement_turn_traces
                    GROUP BY 1
                """).fetchall()

                report["answer_repair_status_counts"] = {
                    str(row["repair_status"]): int(row["total"])
                    for row in repair_rows
                }

            # A2/A3: candidate and validation integrity.
            if report["tables"]["zara_lesson_candidates"]:

                rows = connection.execute("""
                    SELECT
                        id,
                        rule_key,
                        risk_level,
                        status,
                        confidence,
                        evidence_trace_ids,
                        validation
                    FROM zara_lesson_candidates
                    ORDER BY id DESC
                """).fetchall()

                candidate_counts = Counter()
                validation_violations = []

                from app.improvement.observer import (
                    _VALIDATION_THRESHOLDS,
                )

                for row in rows:

                    status = str(row["status"])
                    risk = str(row["risk_level"])

                    candidate_counts[status] += 1

                    if status != "validated":
                        continue

                    validation = _as_dict(row["validation"])
                    evidence = _as_dict(validation.get("evidence"))

                    ids = _as_list(row["evidence_trace_ids"])
                    trace_ids = _as_list(evidence.get("trace_ids"))
                    matching_ids = _as_list(
                        evidence.get("matching_trace_ids")
                    )

                    thresholds = _VALIDATION_THRESHOLDS.get(risk, {})

                    if (
                        validation.get("state") != "validated"
                        or validation.get("production_approval") is not False
                        or not thresholds
                        or float(row["confidence"])
                        < thresholds["min_confidence"]
                        or len(ids) < thresholds["min_evidence"]
                        or len(trace_ids) != len(ids)
                        or len(matching_ids)
                        < thresholds["min_matches"]
                    ):
                        validation_violations.append(
                            int(row["id"])
                        )

                report["candidate_status_counts"] = dict(
                    candidate_counts
                )

                report["invalid_validated_candidates"] = (
                    validation_violations
                )

                if validation_violations:
                    report["violations"].append(
                        "validated_candidate_integrity_failure"
                    )

            # A4: active versions and authorization.
            if report["tables"]["zara_production_lessons"]:

                duplicate_rows = connection.execute("""
                    SELECT rule_key, COUNT(*) AS total
                    FROM zara_production_lessons
                    WHERE active = TRUE
                    GROUP BY rule_key
                    HAVING COUNT(*) > 1
                """).fetchall()

                report["duplicate_active_rules"] = [
                    {
                        "rule_key": str(row["rule_key"]),
                        "count": int(row["total"]),
                    }
                    for row in duplicate_rows
                ]

                status_rows = connection.execute("""
                    SELECT
                        production.id,
                        production.candidate_id,
                        production.rule_key,
                        production.active,
                        candidate.status AS candidate_status
                    FROM zara_production_lessons AS production
                    LEFT JOIN zara_lesson_candidates AS candidate
                      ON candidate.id = production.candidate_id
                    WHERE production.active = TRUE
                      AND candidate.status <> 'approved'
                """).fetchall()

                report["production_state_mismatches"] = [
                    int(row["id"])
                    for row in status_rows
                ]

                if duplicate_rows or status_rows:
                    report["violations"].append(
                        "production_lesson_state_integrity_failure"
                    )

            if report["tables"][
                "zara_lesson_promotion_authorizations"
            ]:
                authorization = connection.execute("""
                    SELECT
                        enabled,
                        allowed_rule_keys,
                        max_risk_level,
                        policy_version
                    FROM zara_lesson_promotion_authorizations
                    WHERE authorization_key = 'system_a'
                """).fetchone()

                report["automatic_promotion_authorization"] = (
                    dict(authorization)
                    if authorization else None
                )

            if report["tables"]["zara_lesson_promotion_events"]:
                rows = connection.execute("""
                    SELECT action, outcome, COUNT(*) AS total
                    FROM zara_lesson_promotion_events
                    GROUP BY action, outcome
                """).fetchall()

                report["promotion_event_counts"] = [
                    {
                        "action": str(row["action"]),
                        "outcome": str(row["outcome"]),
                        "count": int(row["total"]),
                    }
                    for row in rows
                ]

            # A6: approval, application and rollback boundaries.
            if report["tables"]["zara_code_repair_proposals"]:

                proposals = connection.execute("""
                    SELECT
                        id,
                        status,
                        reviewer,
                        review_note,
                        baseline_sha256,
                        proposed_sha256,
                        backup_path,
                        applied_sha256,
                        validation
                    FROM zara_code_repair_proposals
                """).fetchall()

                counts = Counter()
                boundary_violations = []

                for row in proposals:

                    status = str(row["status"])
                    proposal_id = int(row["id"])
                    counts[status] += 1

                    if status == "applied":

                        validation = _as_dict(row["validation"])

                        if (
                            not row["reviewer"]
                            or not row["review_note"]
                            or not row["backup_path"]
                            or row["applied_sha256"]
                            != row["proposed_sha256"]
                            or validation.get("status") != "passed"
                        ):
                            boundary_violations.append(proposal_id)

                report["code_repair_status_counts"] = dict(counts)

                report["code_repair_boundary_violations"] = (
                    boundary_violations
                )

                if boundary_violations:
                    report["violations"].append(
                        "code_repair_approval_boundary_failure"
                    )

                # Check rollback event history.
                if report["tables"].get(
                    "zara_code_repair_events",
                    False,
                ):

                    rollback_rows = connection.execute("""
                        SELECT proposal.id
                        FROM zara_code_repair_proposals AS proposal
                        WHERE proposal.status = 'rolled_back'
                          AND NOT EXISTS (
                              SELECT 1
                              FROM zara_code_repair_events AS event
                              WHERE event.proposal_id = proposal.id
                                AND event.event_type = 'rolled_back'
                          )
                    """).fetchall()

                    report["rollback_history_violations"] = [
                        int(row["id"])
                        for row in rollback_rows
                    ]

                    if rollback_rows:
                        report["violations"].append(
                            "code_repair_rollback_history_failure"
                        )

                else:
                    report["rollback_history_violations"] = []

                    report["violations"].append(
                        "code_repair_events_table_missing"
                    )

    finally:
        connection.close()

    if report["violations"]:
        report["status"] = "violations_found"

    elif report["missing_tables"]:
        report["status"] = "incomplete_schema"

    else:
        report["status"] = (
            "audit_completed_no_detected_violations"
        )

    return report

def evaluate_system_a_acceptance(
    static: dict[str, Any],
    model: dict[str, Any] | None,
    database: dict[str, Any] | None,
    dataset_count: int,
) -> dict[str, Any]:

    issue_metrics = (
        model.get("issue_classification", {})
        if isinstance(model, dict) else {}
    )

    lesson_metrics = (
        model.get("lesson_extraction", {})
        if isinstance(model, dict) else {}
    )

    database_ok = bool(
        database
        and database.get("status")
        == "audit_completed_no_detected_violations"
    )

    checks = {
        "static_checks": (
            static.get("failed", 1) == 0
            and static.get("passed", 0) == static.get("total", -1)
        ),
        "minimum_labelled_cases": dataset_count >= 30,
        "model_evaluation_completed": (
            model is not None
            and model.get("error_cases", 1) == 0
            and model.get("completed_cases", 0) == dataset_count
        ),
        "issue_classification": (
            float(issue_metrics.get("f1", 0)) >= 0.80
        ),
        "lesson_precision": (
            float(lesson_metrics.get("precision", 0)) >= 0.80
        ),
        "lesson_recall": (
            float(lesson_metrics.get("recall", 0)) >= 0.70
        ),
        "database_audit": database_ok,
    }

    passed = all(checks.values())

    return {
        "status": "accepted" if passed else "not_accepted",
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
        "failed_checks": [
            name
            for name, result in checks.items()
            if not result
        ],
        "note": (
            "Passing these gates establishes evaluated acceptance, "
            "not a guarantee of zero future defects."
        ),
    }

def compare_baseline(
    current: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:

    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )

    current_model = current.get("model_evaluation") or {}
    old_model = baseline.get("model_evaluation") or {}

    deltas = {}

    for section in (
        "issue_classification",
        "lesson_extraction",
    ):

        current_metrics = current_model.get(section) or {}
        old_metrics = old_model.get(section) or {}

        if "f1" in current_metrics and "f1" in old_metrics:
            deltas[section] = round(
                float(current_metrics["f1"])
                - float(old_metrics["f1"]),
                4,
            )

    return {
        "baseline_file": str(baseline_path),
        "metric_deltas": deltas,
        "comparison_is_valid": bool(deltas),
        "note": (
            "Compare identical datasets and compatible model/prompt "
            "versions before claiming improvement."
        ),
    }


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Zara System A evaluation and acceptance runner."
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
    )

    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Run labelled A1/A2 cases through the configured Ollama model.",
    )

    parser.add_argument(
        "--db-audit",
        action="store_true",
        help="Audit existing System A data using read-only SQL.",
    )

    parser.add_argument(
        "--output",
        help="Optional evaluation report path.",
    )

    parser.add_argument(
        "--compare",
        help="Optional previous report JSON for comparison.",
    )

    args = parser.parse_args()

    try:
        dataset_path = Path(args.dataset).expanduser().resolve()
        cases = load_cases(dataset_path)
        static = run_static_acceptance()

    except Exception as exc:
        print(
            f"[System A evaluation preparation failed] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    report = {
        "schema_version": 1,
        "evaluation_id": hashlib.sha256(
            (
                str(datetime.now().astimezone().isoformat())
                + str(dataset_path)
            ).encode("utf-8")
        ).hexdigest()[:24],
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "case_count": len(cases),
            "sha256": sha256_file(dataset_path),
        },
        "scope": {
            "database_writes": False,
            "source_file_writes": False,
            "production_promotions": False,
            "code_patch_application": False,
        },
        "static_acceptance": static,
        "model_evaluation": None,
        "database_audit": None,
        "baseline_comparison": None,
    }

    if args.with_model:
        try:
            report["model_evaluation"] = run_model_evaluation(cases)
        except Exception as exc:
            report["model_evaluation"] = {
                "status": "error",
                "error_code": type(exc).__name__,
            }

    if args.db_audit:
        try:
            report["database_audit"] = (
                run_read_only_database_audit()
            )
        except Exception as exc:
            report["database_audit"] = {
                "status": "error",
                "error_code": type(exc).__name__,
            }

    report["acceptance"] = evaluate_system_a_acceptance(
        static=static,
        model=report["model_evaluation"],
        database=report["database_audit"],
        dataset_count=len(cases),
    )

    source_files = [
        PROJECT_ROOT / "app" / "improvement" / "observer.py",
        PROJECT_ROOT / "app" / "improvement_repair" / "answer_repair.py",
        PROJECT_ROOT / "app" / "improvement_repair" / "code_repair.py",
        PROJECT_ROOT / "scripts" / "eval_runner.py",
        PROJECT_ROOT / "scripts" / "system_a_eval_runner.py",
        PROJECT_ROOT / "app" / "knowledge" / "system_a_eval_cases.jsonl",
        PROJECT_ROOT / "app" / "ai" / "chat_engine.py",
        PROJECT_ROOT / "app" / "learning_ai" / "learning_store.py",
        PROJECT_ROOT / "app" / "learning_ai" / "embedding_client.py",
        PROJECT_ROOT / "app" / "learning_ai" / "document_ingest.py",
    ]

    report["source_provenance"] = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in source_files
    }

    if args.compare:
        try:
            report["baseline_comparison"] = compare_baseline(
                report,
                Path(args.compare).expanduser().resolve(),
            )
        except Exception as exc:
            report["baseline_comparison"] = {
                "status": "error",
                "error_code": type(exc).__name__,
            }

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )

        output = RESULTS_DIR / f"system_a_{timestamp}.json"

    try:
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.exists():
            raise FileExistsError(
                "Output already exists; choose another report path."
            )

        output.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

    except OSError as exc:
        print(
            f"[Report output error] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    print("System A evaluation report")
    print(f"Cases: {len(cases)}")
    print(f"Static checks: {static['passed']}/{static['total']}")
    print(f"Static failures: {static['failed']}")
    print(
        "Static failed checks:",
        ", ".join(static.get("failed_checks", [])) or "none",
    )

    print(
        "System A acceptance:",
        report["acceptance"]["status"],
    )

    if report["acceptance"]["failed_checks"]:
        print(
            "Acceptance pending:",
            ", ".join(
                report["acceptance"]["failed_checks"]
            ),
        )
    print(f"Report: {output}")

    if report["model_evaluation"]:
        print(
            "Issue-classification F1:",
            report["model_evaluation"].get(
                "issue_classification", {}
            ).get("f1"),
        )

        print(
            "Lesson-extraction F1:",
            report["model_evaluation"].get(
                "lesson_extraction", {}
            ).get("f1"),
        )

    if report["database_audit"]:
        print(
            "Database audit:",
            report["database_audit"].get("status"),
        )

    print(
        "Model calls:",
        "enabled" if args.with_model else "not run",
    )

    print(
        "Database audit:",
        "read-only" if args.db_audit else "not run",
    )

    print(
        "No production changes or code patches were applied."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())