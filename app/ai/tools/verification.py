"""Independent verification contracts for Zara tool outcomes."""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from typing import Any, Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VerificationReport:
    status: str
    verified: bool | None

    method: str
    message: str

    error_code: str | None = None

    def __post_init__(self) -> None:

        if self.status not in {
            "verified",
            "failed",
            "not_applicable",
            "contract_only",
            "missing_verifier",
        }:
            raise ValueError("Invalid verification status.")

        if (
            self.verified is not None
            and not isinstance(self.verified, bool)
        ):
            raise TypeError(
                "Verification decision must be boolean or None."
            )

        if (
            not isinstance(self.method, str)
            or not self.method.strip()
            or len(self.method) > 100
        ):
            raise ValueError("Invalid verification method.")

        if (
            not isinstance(self.message, str)
            or not self.message.strip()
            or len(self.message) > 1000
        ):
            raise ValueError("Invalid verification message.")
        
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verified": self.verified,
            "method": self.method,
            "message": self.message,
            "error_code": self.error_code,
        }


VerifierHandler = Callable[
    [dict[str, Any], dict[str, Any]],
    bool | VerificationReport,
]


class ToolOutcomeVerifier:
    """
    Application-owned independent verification callbacks.

    Verifiers must be trusted Python code. The model cannot
    register or modify them.
    """

    def __init__(self) -> None:
        self._verifiers: dict[str, VerifierHandler] = {}

    def register(
        self,
        tool_name: str,
        verifier: VerifierHandler,
        *,
        replace: bool = False,
    ) -> None:

        if (
            not isinstance(tool_name, str)
            or not re.fullmatch(
                r"[a-z][a-z0-9_]{0,63}",
                tool_name,
            )
        ):
            raise ValueError("Invalid verifier tool name.")

        if not callable(verifier):
            raise TypeError("Verifier must be callable.")

        if tool_name in self._verifiers and not replace:
            raise ValueError(
                f"A verifier already exists for {tool_name!r}."
            )

        self._verifiers[tool_name] = verifier

    def has_verifier(self, tool_name: str) -> bool:
        return tool_name in self._verifiers

    def verify(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        *,
        permission: str,
    ) -> VerificationReport:

        if result.get("status") != "success":
            return VerificationReport(
                status="not_applicable",
                verified=None,
                method="execution_status",
                message="No successful result requires verification.",
            )

        verifier = self._verifiers.get(tool_name)

        if verifier is None:

            if permission == "mutate":
                return VerificationReport(
                    status="missing_verifier",
                    verified=False,
                    method="none",
                    message=(
                        "A mutating tool cannot report verified success "
                        "without an independent postcondition verifier."
                    ),
                    error_code="result_verifier_missing",
                )

            return VerificationReport(
                status="contract_only",
                verified=None,
                method="schema_validation",
                message=(
                    "The result passed its registered schema contract, "
                    "but no independent verifier is registered."
                ),
            )

        try:
            outcome = verifier(
                dict(arguments),
                dict(result),
            )

            if isinstance(outcome, VerificationReport):
                report = outcome

            elif isinstance(outcome, bool):
                report = VerificationReport(
                    status=(
                        "verified"
                        if outcome
                        else "failed"
                    ),
                    verified=outcome,
                    method="application_postcondition",
                    message=(
                        "The tool's independent postcondition was verified."
                        if outcome
                        else "The tool's postcondition could not be verified."
                    ),
                    error_code=(
                        None
                        if outcome
                        else "postcondition_not_verified"
                    ),
                )

            else:
                report = VerificationReport(
                    status="failed",
                    verified=False,
                    method="invalid_verifier_output",
                    message="The verifier returned an invalid decision.",
                    error_code="invalid_verifier_output",
                )

            if report.verified not in {True, False, None}:
                raise ValueError("Invalid verifier outcome.")

            return report

        except Exception as exc:
            logger.error(
                "Tool verifier '%s' failed (%s).",
                tool_name,
                type(exc).__name__,
            )

            return VerificationReport(
                status="failed",
                verified=False,
                method="verifier_exception",
                message="Independent result verification failed.",
                error_code="result_verification_failed",
            )