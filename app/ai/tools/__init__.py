"""Public entry point for Zara's application tool system."""

from app.ai.tools.contracts import (
    ToolResult,
    ToolStatus,
)

from app.ai.tools.confirmation import (
    ConfirmationChallenge,
    ConfirmationError,
    ConfirmationGrant,
    ToolConfirmationManager,
)

from app.ai.tools.database import (
    DATABASE_TOOL_DEFINITIONS,
    DATABASE_TOOL_REGISTRY,
    DatabaseToolService,
    should_route_database_request,
)
from app.ai.tools.router import (
    ActionRequest,
    UnifiedActionRouter,
)

from app.ai.tools.permissions import (
    PermissionDecision,
    ToolPermissionPolicy,
)

from app.ai.tools.registry import (
    ToolMetadata,
    ToolRegistry,
    ToolRegistryError,
    ToolSpec,
    ToolValidationError,
)
from app.ai.tools.execution import (
    CancellationToken,
    ExecutionPolicy,
    IdempotencyStore,
    InMemoryIdempotencyStore,
    ToolCancelledError,
    ToolDeadlineExceeded,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionRuntime,
)

from app.ai.tools.audit import (
    ToolAuditError,
    ToolAuditLogger,
)

from app.ai.tools.verification import (
    ToolOutcomeVerifier,
    VerificationReport,
)

from app.ai.tools.acceptance import (
    ToolAcceptanceEvaluator,
)

TOOL_REGISTRY = DATABASE_TOOL_REGISTRY


__all__ = [
    "ConfirmationChallenge",
    "ConfirmationError",
    "ConfirmationGrant",
    "DATABASE_TOOL_DEFINITIONS",
    "DATABASE_TOOL_REGISTRY",
    "DatabaseToolService",
    "PermissionDecision",
    "TOOL_REGISTRY",
    "ToolConfirmationManager",
    "ToolMetadata",
    "ToolPermissionPolicy",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolResult",
    "ToolSpec",
    "ActionRequest",
    "UnifiedActionRouter",
    "ToolStatus",
    "ToolValidationError",
    "should_route_database_request",
    "CancellationToken",
    "ExecutionPolicy",
    "IdempotencyStore",
    "InMemoryIdempotencyStore",
    "ToolCancelledError",
    "ToolDeadlineExceeded",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolExecutionRuntime",
    "ToolAuditError",
    "ToolAuditLogger",
    "ToolAcceptanceEvaluator",
    "ToolOutcomeVerifier",
    "VerificationReport",
]