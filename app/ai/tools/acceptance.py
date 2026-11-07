"""Static security and integration acceptance for Zara tools."""

from __future__ import annotations

from typing import Any


SECURITY_REGRESSION_CASES = (
    ("SEC-001", "Unknown tool is rejected."),
    ("SEC-002", "Malformed JSON arguments are rejected."),
    ("SEC-003", "Unknown schema fields are rejected."),
    ("SEC-004", "Wrong interface is denied."),
    ("SEC-005", "Missing permission scope is denied."),
    ("SEC-006", "Medium-risk operation requests confirmation."),
    ("SEC-007", "High-risk operation requests confirmation."),
    ("SEC-008", "Critical-risk operation is blocked by default."),
    ("SEC-009", "Expired confirmation cannot be consumed."),
    ("SEC-010", "Cross-session confirmation is rejected."),
    ("SEC-011", "Changed arguments invalidate confirmation."),
    ("SEC-012", "Confirmation replay is rejected."),
    ("SEC-013", "Cancellation prevents pending execution."),
    ("SEC-014", "Non-idempotent mutations are not retried."),
    ("SEC-015", "Duplicate idempotency keys do not repeat actions."),
    ("SEC-016", "Unknown mutation outcomes require reconciliation."),
    ("SEC-017", "Audit storage failure blocks risky writes."),
    ("SEC-018", "Successful mutation requires independent verification."),
    ("SEC-019", "Sensitive values never appear in audit records."),
    ("SEC-020", "Tool registration cannot overwrite existing names."),
)


class ToolAcceptanceEvaluator:
    """
    Static readiness evaluator.

    It verifies architecture and contracts only.
    It does not claim that live integrations or regression tests passed.
    """

    def evaluate(
        self,
        router: Any,
    ) -> dict[str, Any]:

        checks: list[dict[str, Any]] = []

        def add(
            name: str,
            passed: bool,
            message: str,
        ) -> None:

            checks.append({
                "name": name,
                "status": "passed" if passed else "failed",
                "message": message,
            })

        registry = router.registry
        runtime = router.execution_runtime
        policy = router.permission_policy

        add(
            "shared_registry",
            (
                runtime.registry is registry
                and policy.registry is registry
            ),
            "Router, runtime and permission policy share one registry.",
        )

        audit_logger = getattr(
            router,
            "audit_logger",
            None,
        )

        audit_ready = bool(
            audit_logger is not None
            and audit_logger.is_healthy
        )

        add(
            "audit_storage",
            audit_ready,
            (
                "Persistent audit logger is initialized."
                if audit_ready
                else "Persistent audit logger is unavailable."
            ),
        )

        if audit_ready:
            integrity = audit_logger.verify_chain()

            add(
                "audit_integrity",
                bool(integrity.get("valid")),
                (
                    "Retained audit event hashes and links are valid."
                    if integrity.get("valid")
                    else "Audit chain integrity could not be verified."
                ),
            )

        specs = registry.specifications()

        for spec in specs:

            prefix = f"tool:{spec.name}"

            legacy = (
                spec.name in router.legacy_tool_names
                and router.database_service is not None
            )

            generic = (
                spec.name in router._generic_tools
            )

            add(
                f"{prefix}:handler",
                legacy or generic,
                (
                    "Trusted application execution handler is available."
                    if legacy or generic
                    else "No trusted execution handler is registered."
                ),
            )

            metadata = spec.metadata

            valid_risk = getattr(
                metadata,
                "risk_level",
                None,
            ) in {
                "low",
                "medium",
                "high",
                "critical",
            }

            add(
                f"{prefix}:risk",
                valid_risk,
                "Tool has an explicit supported risk classification.",
            )

            timeout = getattr(
                metadata,
                "timeout_seconds",
                None,
            )

            valid_timeout = (
                isinstance(timeout, (int, float))
                and not isinstance(timeout, bool)
                and 0.1 <= timeout <= 600
            )

            add(
                f"{prefix}:timeout",
                valid_timeout,
                "Tool timeout is within the execution policy.",
            )

            retries = getattr(
                metadata,
                "max_retries",
                None,
            )

            idempotent = getattr(
                metadata,
                "idempotent",
                False,
            )

            retry_safe = (
                isinstance(retries, int)
                and not isinstance(retries, bool)
                and 0 <= retries <= 5
                and not (
                    spec.permission == "mutate"
                    and retries > 0
                    and not idempotent
                )
            )

            add(
                f"{prefix}:retry_policy",
                retry_safe,
                "Retry configuration does not repeat unsafe mutations.",
            )

            if generic:

                requires_confirmation = (
                    spec.confirmation in {"required", "conditional"}
                    or metadata.risk_level in {
                        "medium",
                        "high",
                        "critical",
                    }
                )

                preview_ready = (
                    not requires_confirmation
                    or spec.name in router._confirmation_previews
                )

                add(
                    f"{prefix}:confirmation_preview",
                    preview_ready,
                    (
                        "Required action preview is registered."
                        if preview_ready
                        else "Risky tool is missing its trusted confirmation preview."
                    ),
                )

                verifier_ready = (
                    spec.permission != "mutate"
                    or router.result_verifier.has_verifier(
                        spec.name
                    )
                )

                add(
                    f"{prefix}:result_verifier",
                    verifier_ready,
                    (
                        "Required independent verifier is available."
                        if verifier_ready
                        else "Mutating tool needs an independent result verifier."
                    ),
                )

            if spec.permission == "mutate":

                add(
                    f"{prefix}:audit",
                    audit_ready,
                    "Mutations require persistent audit availability.",
                )

        failures = sum(
            item["status"] == "failed"
            for item in checks
        )

        passed = len(checks) - failures

        return {
            "schema_version": 1,
            "scope": "static_tool_readiness",
            "status": "ready" if failures == 0 else "blocked",
            "total_checks": len(checks),
            "passed_checks": passed,
            "failed_checks": failures,
            "checks": checks,
            "regression_cases": [
                {
                    "case_id": case_id,
                    "description": description,
                    "status": "not_run",
                }
                for case_id, description
                in SECURITY_REGRESSION_CASES
            ],
            "execution_note": (
                "No tool, model, database or external service was executed "
                "by this static acceptance evaluator."
            ),
        }