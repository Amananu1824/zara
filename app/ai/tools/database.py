"""Bounded personal-memory database tools for Zara."""
from __future__ import annotations

import logging
import hashlib
import json
import re
import secrets
import time

from dataclasses import dataclass
from threading import RLock, Event
from typing import Any

from app.learning_ai.document_ingest import DocumentIngestor
from app.core.task_result import TaskResult, TaskStatus
from app.ai.tools.permissions import ToolPermissionPolicy
from app.memory_management.manager import MemoryOperationCancelled
from app.ai.tools.registry import (
    ToolMetadata,
    ToolRegistry,
    ToolValidationError,
)
from app.learning_ai.learning_store import (
    DatabaseOperationCancelled,
    DatabaseOperationOutcomeUnknown,
    LearningStore,
)


logger = logging.getLogger(__name__)


PROFILE_FIELD_KEYS = frozenset({
    "name",
    "preferred_address",
    "education",
    "current_learning",
    "current_goal",
    "long_term_goals",
    "interests",
    "hobbies",
    "business_focus",
    "learning_style",
    "communication_style",
    "name_usage",
    "humor_and_mood",
    "emotional_support",
    "advice_preference",
    "follow_up_preference",
    "response_structure",
    "emoji_preference",
    "language_preference",
    "profile_usage",
})


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


DATABASE_TOOL_DEFINITIONS = [
    _tool(
        "memory_search",
        (
            "Search saved personal profile fields and Zara-created notes. "
            "Use this when the user asks what is saved or asks to find a "
            "specific saved item. Return record IDs for later references."
        ),
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "scope": {
                "type": "string",
                "enum": ["all", "profile", "notes"],
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
        },
        ["query"],
    ),
    _tool(
        "memory_latest",
        (
            "List the most recently updated saved profile fields and notes. "
            "Use this for requests such as 'what did I save most recently?'"
        ),
        {
            "scope": {
                "type": "string",
                "enum": ["all", "profile", "notes"],
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
        },
        [],
    ),
    _tool(
        "memory_save",
        (
            "Save a new personal note. Only use after an explicit user "
            "request to remember/save/store the information. This inserts "
            "a new note and never overwrites an existing record."
        ),
        {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1500,
            },
            "label": {
                "type": "string",
                "maxLength": 64,
            },
        },
        ["content"],
    ),
    _tool(
        "profile_save",
        (
            "Save a new approved profile field or request an update to an "
            "existing field. Only use after an explicit request to save or "
            "remember the information. Updating an existing value always "
            "requires a separate confirmation command."
        ),
        {
            "memory_key": {
                "type": "string",
                "enum": sorted(PROFILE_FIELD_KEYS),
            },
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 600,
            },
        },
        ["memory_key", "content"],
    ),
    _tool(
        "memory_update",
        (
            "Prepare an update for one saved profile/note record by its "
            "record ID. This never applies the change immediately. The "
            "application will show the current value and proposed value "
            "and require an explicit confirmation token."
        ),
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
            "new_content": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1500,
            },
        },
        ["memory_id", "new_content"],
    ),
    _tool(
        "memory_delete",
        (
            "Prepare deletion of one saved profile/note record by its "
            "record ID. This never deletes immediately. The application "
            "will show which record would be deleted and require a "
            "separate confirmation token."
        ),
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
]


_TOOL_SCHEMA_BY_NAME = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in DATABASE_TOOL_DEFINITIONS
}

_MUTATING_TOOL_NAMES = {
    "memory_save",
    "profile_save",
    "memory_update",
    "memory_delete",
    "feedback_review",
}

_READ_ONLY_TOOL_NAMES = {
    "memory_search",
    "memory_latest",
    "document_list",
    "document_search",
    "feedback_stats",
    "feedback_list",
    "feedback_get",
}


_READ_ONLY_DATABASE_TOOLS = [
    _tool(
        "document_list",
        "List indexed documents and their indexing status.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
        },
        [],
    ),
    _tool(
        "document_search",
        (
            "Search indexed document chunks. Return source filename, "
            "page, section, similarity and a short evidence excerpt."
        ),
        {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
        },
        ["query"],
    ),
    _tool(
        "feedback_stats",
        "Read feedback counts: total, good, bad and reviewed.",
        {},
        [],
    ),
    _tool(
        "feedback_list",
        "List recent feedback records with rating, review status and note.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
        },
        [],
    ),
    _tool(
        "feedback_get",
        "Read one feedback record by exact numeric ID.",
        {
            "feedback_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["feedback_id"],
    ),
    _tool(
        "feedback_review",
        (
            "Prepare approval for one existing bad-feedback record with a "
            "written note. This only creates a confirmation preview; it "
            "never approves or updates the feedback immediately."
        ),
        {
            "feedback_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["feedback_id"],
    ),
]

_TOPIC_MEMORY_READ_TOOLS = [
    _tool(
        "topic_memory_list",
        "List active approved or pending personal topic memories. "
        "Use only for questions about the user's saved personal memories.",
        {
            "status": {
                "type": "string",
                "enum": ["approved", "candidate", "all"],
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
            },
        },
        [],
    ),
    _tool(
        "topic_memory_candidates",
        "List automatic personal-memory candidates awaiting review.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
            },
        },
        [],
    ),
    _tool(
        "topic_memory_review_candidate",
        "Inspect one pending candidate by exact record ID.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_history",
        "Read the version history of one personal memory.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_audit",
        "Read lifecycle audit events for one personal memory.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_consolidation_preview",
        "Preview exact duplicate and conflicting memory candidates. "
        "This tool never changes records.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
        },
        [],
    ),
    _tool(
        "topic_memory_repair_preview",
        "Diagnose personal-memory integrity, privacy risks, stale embeddings, "
        "version relationships and safe repair opportunities. Read-only.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
            },
        },
        [],
    ),
]

_TOPIC_MEMORY_MUTATING_TOOLS = [
    _tool(
        "topic_memory_correct",
        "Prepare a corrected new version of one approved memory. "
        "Never overwrite the old version. Requires explicit confirmation.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
            "new_summary": {
                "type": "string",
                "minLength": 12,
                "maxLength": 320,
            },
        },
        ["memory_id", "new_summary"],
    ),
    _tool(
        "topic_memory_delete",
        "Prepare a reversible soft deletion of one approved memory. "
        "Requires explicit confirmation.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_restore",
        "Prepare restoration of one deleted or superseded memory as a "
        "new version. Requires explicit confirmation.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_candidate_approve",
        "Prepare explicit promotion of a pending candidate to approved "
        "personal memory. Existing conflicting approved memory is never "
        "silently replaced. Requires confirmation.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_candidate_reject",
        "Prepare rejection of a pending personal-memory candidate. "
        "The candidate remains in history. Requires confirmation.",
        {
            "memory_id": {
                "type": "integer",
                "minimum": 1,
            },
            "reason_code": {
                "type": "string",
                "enum": [
                    "incorrect",
                    "temporary",
                    "duplicate",
                    "not_useful",
                    "sensitive",
                    "other",
                ],
            },
        },
        ["memory_id"],
    ),
    _tool(
        "topic_memory_consolidate",
        "Prepare consolidation of exact duplicate pending candidates. "
        "Different summaries are never automatically combined. "
        "Requires confirmation.",
        {
            "topic_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 80,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
        },
        [],
    ),
    _tool(
        "topic_memory_repair_apply",
        "Prepare a confirmation-protected repair of verified memory metadata, "
        "embeddings and privacy-risk records.",
        {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
            },
        },
        [],
    ),
    _tool(
        "topic_memory_repair_rollback",
        "Prepare rollback of a previous memory repair using its saved snapshot.",
        {
            "repair_id": {
                "type": "string",
                "minLength": 32,
                "maxLength": 32,
            },
        },
        ["repair_id"],
    ),
]

DATABASE_TOOL_DEFINITIONS.extend(_READ_ONLY_DATABASE_TOOLS)
DATABASE_TOOL_DEFINITIONS.extend(_TOPIC_MEMORY_READ_TOOLS)
DATABASE_TOOL_DEFINITIONS.extend(_TOPIC_MEMORY_MUTATING_TOOLS)

_TOOL_SCHEMA_BY_NAME = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in DATABASE_TOOL_DEFINITIONS
}

_MUTATING_TOOL_NAMES.update({
    "topic_memory_correct",
    "topic_memory_delete",
    "topic_memory_restore",
    "topic_memory_candidate_approve",
    "topic_memory_candidate_reject",
    "topic_memory_consolidate",
    "topic_memory_repair_apply",
    "topic_memory_repair_rollback",
})

_READ_ONLY_TOOL_NAMES.update({
    "topic_memory_list",
    "topic_memory_candidates",
    "topic_memory_review_candidate",
    "topic_memory_history",
    "topic_memory_audit",
    "topic_memory_consolidation_preview",
    "topic_memory_repair_preview",
})

DATABASE_TOOL_REGISTRY = ToolRegistry(
    DATABASE_TOOL_DEFINITIONS,

    read_only_tools=_READ_ONLY_TOOL_NAMES,
    mutating_tools=_MUTATING_TOOL_NAMES,

    confirmation_tools={
        "memory_update",
        "memory_delete",
        "feedback_review",
        "topic_memory_correct",
        "topic_memory_delete",
        "topic_memory_restore",
        "topic_memory_candidate_approve",
        "topic_memory_candidate_reject",
        "topic_memory_consolidate",
        "topic_memory_repair_apply",
        "topic_memory_repair_rollback",
    },

    conditional_confirmation_tools={
        "profile_save",
    },

    default_metadata=ToolMetadata(
        category="database",
        version="1.0.0",
        provider="application",
        supported_interfaces=(
            "terminal",
            "api",
        ),
        risk_level="low",
        required_scopes=(),
    ),
)

_REVIEW_FEEDBACK_PATTERN = re.compile(
    r"^\s*/feedback-review\b"
    r"|"
    r"\b(?:approve|review)\s+(?:the\s+)?feedback\b"
    r"|"
    r"\bfeedback\b.{0,50}\b(?:approve|review)\b",
    re.IGNORECASE,
)

_DOCUMENT_FEEDBACK_OPERATION_PATTERN = re.compile(
    r"^\s*/(?:documents|doc-search|feedback-list|feedback-stats|"
    r"feedback-review)\b"
    r"|"
    r"\b(?:show|list|search|find|read)\b.{0,70}"
    r"\b(?:indexed\s+documents?|saved\s+documents?|document\s+chunks?)\b"
    r"|"
    r"\b(?:feedback\s+(?:stats|statistics|history|records?|list|summary))\b"
    r"|"
    r"\b(?:list|show|search|find)\s+(?:recent\s+)?feedback\b"
    r"|"
    r"\b(?:read|show|inspect)\s+feedback\s+#?\d+\b",
    re.IGNORECASE,
)

_DOCUMENT_COLLECTION_TARGET_PATTERN = re.compile(
    r"\b(?:indexed|saved|uploaded|available)\s+"
    r"(?:documents?|files?)\b"
    r"|"
    r"\b(?:document|file)\s+(?:list|inventory)\b",
    re.IGNORECASE,
)

_DOCUMENT_COLLECTION_ACTION_PATTERN = re.compile(
    r"\b(?:show|list|display|view|fetch|get|"
    r"dikhao|dikhaiye|dikha(?:o|iye)?|"
    r"batao|bataiye|dikha\s+do|"
    r"list\s+karo|nikaalo|nikal\s+do)\b",
    re.IGNORECASE,
)


def _is_document_collection_listing_request(
    message: str,
) -> bool:
    """Detect document-list requests regardless of word order."""
    return bool(
        _DOCUMENT_COLLECTION_TARGET_PATTERN.search(message)
        and _DOCUMENT_COLLECTION_ACTION_PATTERN.search(message)
    )


_LOOKUP_SAVED_PATTERN = re.compile(
    r"\b(?:what|kya|which|latest|last|recent|show|list|search|find|"
    r"retrieve|fetch|read|batao|nikalo)\b.{0,100}"
    r"\b(?:did\s+i\s+(?:ask\s+you\s+to\s+)?save|save|saved|store|"
    r"stored|remember|memory|note|profile|database|record|item)\b",
    re.IGNORECASE,
)

_EXPLICIT_SAVE_PATTERN = re.compile(
    r"\b(?:save|store|note\s+down|yaad\s+rakh(?:na|o)?)\b"
    r"|"
    r"\bremember\s+(?:this|that|my\b|i\b|i'm\b|we\b|mera|meri|mere)\b",
    re.IGNORECASE,
)

_UPDATE_REQUEST_PATTERN = re.compile(
    r"\b(?:update|change|edit|modify|replace|revise|badal|badlo|"
    r"badalna|badal\s+do|change\s+kar)\b",
    re.IGNORECASE,
)

_DELETE_REQUEST_PATTERN = re.compile(
    r"\b(?:delete|remove|forget|clear|hatao|mitao|mita\s+do|"
    r"delete\s+kar|remove\s+kar)\b",
    re.IGNORECASE,
)

_DATABASE_OPERATION_PATTERN = re.compile(
    r"\b(?:database|db)\b"
    r"|"
    r"\b(?:saved|stored)\s+(?:memory|note|fact|profile|data|item|record)\b"
    r"|"
    r"\b(?:latest|last|recent)\b.{0,70}"
    r"\b(?:save|saved|store|stored|memory|note|record|data|item)\b"
    r"|"
    r"\b(?:show|list|search|find|retrieve|fetch|read|lookup|look\s+up|"
    r"nikal|batao|dikhao)\b.{0,80}"
    r"\b(?:saved|stored|memory|note|profile|record|data|database)\b"
    r"|"
    r"\b(?:save|store|remember|note\s+down|yaad\s+rakh)\b.{0,90}"
    r"\b(?:my|this|that|it|profile|memory|note|fact|preference|goal|"
    r"interest|database|record|data|baat|pasand|mera|meri|mere)\b"
    r"|"
    r"\b(?:update|edit|change|modify|replace|delete|remove|forget|"
    r"hatao|mitao)\b.{0,80}"
    r"\b(?:memory|note|profile|saved|stored|record|database|data|field|value)\b"
    r"|"
    r"\b(?:what|kya)\b.{0,90}"
    r"\b(?:did\s+i\s+(?:ask\s+you\s+to\s+)?save|save\s+(?:karwaya|krwaya)|"
    r"saved|stored|remember|memory|note|profile)\b",
    re.IGNORECASE,
)

_TOPIC_MEMORY_OPERATION_PATTERN = re.compile(
    r"\b(?:"
    r"approve|reject|restore|recover|correct|update|change|edit|"
    r"delete|remove|forget|consolidate|deduplicate|merge"
    r")\b.{0,110}\b(?:"
    r"memory|memories|candidate|candidates|saved\s+fact|saved\s+note"
    r")\b"
    r"|"
    r"\b(?:pending|automatic|saved|personal)\s+"
    r"(?:memory\s+)?candidates?\b"
    r"|"
    r"\b(?:memory|memories)\s+"
    r"(?:history|audit|versions?|candidates?)\b"
    r"|"
    r"\b(?:list|show|review|inspect)\b.{0,100}"
    r"\b(?:personal\s+)?memories\b",
    re.IGNORECASE,
)

_DATABASE_DEFINITION_PATTERN = re.compile(
    r"\b(?:what\s+is|what's|explain|define|meaning\s+of|"
    r"kya\s+hai|kya\s+hota\s+hai|kya\s+hoti\s+hai)\b.{0,50}"
    r"\b(?:database|db)\b",
    re.IGNORECASE,
)


def should_route_database_request(user_message: str) -> bool:
    """Route real database operations, not general questions about databases."""
    message = (user_message or "").strip()

    if not message:
        return False

    lowered = message.casefold()

    if lowered.startswith((
        "/db-confirm",
        "/db-cancel",
        "/profile-set",
        "/profile-delete",
        "/memory-latest",
        "/memory-search",
    )):
        return True

    has_mutation_intent = bool(
        _EXPLICIT_SAVE_PATTERN.search(message)
        or _UPDATE_REQUEST_PATTERN.search(message)
        or _DELETE_REQUEST_PATTERN.search(message)
    )
    has_lookup_intent = bool(_LOOKUP_SAVED_PATTERN.search(message))

    if (
        _DATABASE_DEFINITION_PATTERN.search(message)
        and not has_mutation_intent
        and not has_lookup_intent
    ):
        return False

    return bool(
        _DATABASE_OPERATION_PATTERN.search(message)
        or _TOPIC_MEMORY_OPERATION_PATTERN.search(message)
        or _DOCUMENT_FEEDBACK_OPERATION_PATTERN.search(message)
        or _is_document_collection_listing_request(message)
    )


@dataclass(frozen=True)
class _PendingMutation:
    token: str
    action: str
    memory_id: int
    expected_content: str
    new_content: str | None
    label: str
    expires_at: float
    scope: str = "legacy"
    payload: dict[str, Any] | None = None

class DatabaseToolService:
    CONFIRMATION_TTL_SECONDS = 120

    def __init__(
        self,
        store: LearningStore,
        *,
        memory_manager: Any | None = None,
        allow_mutations: bool = False,
        allow_tools: bool = True,
        permission_policy: ToolPermissionPolicy | None = None,
    ) -> None:
        self.store = store
        self.memory_manager = memory_manager
        self.registry = DATABASE_TOOL_REGISTRY

        self.permission_policy = permission_policy or ToolPermissionPolicy(
            self.registry,
            tools_enabled=allow_tools,
            mutations_enabled=allow_mutations,
        )

        if self.permission_policy.registry is None:
            self.permission_policy.registry = self.registry

        self.allow_tools = self.permission_policy.tools_enabled
        self.allow_mutations = self.permission_policy.mutations_enabled

        self._pending: dict[str, _PendingMutation] = {}
        self._lock = RLock()
        self.document_ingestor = DocumentIngestor(store)

    @staticmethod
    def confirmation_intent(message: str) -> str | None:
        """Recognise explicit spoken or typed confirmation phrases."""

        text = " ".join(str(message or "").casefold().split())

        if re.fullmatch(
            r"(?:(?:yes|yeah|yep|haan|han|ha|okay|ok|theek\s+hai)\s+)?"
            r"(?:confirm|approve|apply)"
            r"(?:\s+(?:it|this|the\s+change))?[.!]?",
            text,
        ):
            return "confirm"

        if re.fullmatch(
            r"(?:(?:no|nahi|nahin|nope)\s+)?"
            r"(?:cancel|reject)"
            r"(?:\s+(?:it|this|the\s+change))?[.!]?",
            text,
        ):
            return "cancel"

        return None

    def confirm_latest_action(
        self,
        *,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        """Confirm the only currently pending action in this session."""

        with self._lock:
            self._purge_expired_pending()
            tokens = list(self._pending)

        if len(tokens) != 1:
            return self._confirmation_result(
                "database_confirm",
                "failed",
                (
                    "No single unambiguous pending action is available "
                    "for confirmation."
                ),
                started_at=time.monotonic(),
                data={
                    "verified": False,
                    "confirmation_consumed": False,
                    "outcome": "no_action_applied",
                },
                error_code="no_unambiguous_pending_action",
            )

        return self.confirm_action(
            tokens[0],
            cancel_event=cancel_event,
        )

    def cancel_latest_action(self) -> dict[str, Any]:
        """Cancel the only currently pending action in this session."""

        with self._lock:
            self._purge_expired_pending()
            tokens = list(self._pending)

        if len(tokens) != 1:
            return self._confirmation_result(
                "database_cancel",
                "failed",
                "No single pending action is available to cancel.",
                started_at=time.monotonic(),
                data={
                    "verified": False,
                    "confirmation_consumed": False,
                    "outcome": "no_action_applied",
                },
                error_code="no_unambiguous_pending_action",
            )

        return self.cancel_action(tokens[0])
    
    def clear_pending(self) -> int:
        """Invalidate outstanding confirmation previews on session reset."""
        with self._lock:
            count = len(self._pending)
            self._pending.clear()
            return count

    def _mutation_denial(self) -> dict[str, str] | None:
        decision = self.permission_policy.authorize_mutation()

        if decision.allowed:
            return None

        return {
            "status": "denied",
            "message": decision.message,
            "error_code": decision.code,
        }

    @staticmethod
    def _public_record(row: dict[str, Any]) -> dict[str, Any]:
        def serialize_date(value: Any) -> str | None:
            if value is None:
                return None
            if hasattr(value, "isoformat"):
                return value.isoformat()
            return str(value)

        return {
            "id": row.get("id"),
            "memory_key": row.get("memory_key"),
            "memory_type": row.get("memory_type"),
            "content": str(row.get("content") or "")[:600],
            "source": row.get("source"),
            "created_at": serialize_date(row.get("created_at")),
            "updated_at": serialize_date(row.get("updated_at")),
        }

    @staticmethod
    def _public_feedback(row: dict[str, Any]) -> dict[str, Any]:
        created_at = row.get("created_at")
        if hasattr(created_at, "isoformat"):
            created_at = created_at.isoformat()
        elif created_at is not None:
            created_at = str(created_at)

        return {
            "id": int(row["id"]),
            "created_at": created_at,
            "rating": int(row["rating"]),
            "reviewed": bool(row["reviewed"]),
            "note": str(row.get("note") or "")[:300],
            "user_message": str(row.get("user_message") or "")[:240],
            "assistant_response": str(
                row.get("assistant_response") or ""
            )[:300],    
        }

    @staticmethod
    def _validate_arguments(
        tool_name: str,
        args: Any,
    ) -> dict[str, Any]:
        if tool_name not in _TOOL_SCHEMA_BY_NAME:
            raise ValueError("Unknown database tool.")

        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object.")

        schema = _TOOL_SCHEMA_BY_NAME[tool_name]
        properties = schema["properties"]
        required = schema.get("required", [])

        unknown = set(args) - set(properties)
        if unknown:
            raise ValueError(
                "Unexpected tool argument(s): " + ", ".join(sorted(unknown))
            )

        missing = [
            key for key in required
            if key not in args
        ]
        if missing:
            raise ValueError(
                "Missing required argument(s): " + ", ".join(missing)
            )

        clean: dict[str, Any] = {}

        for key, value in args.items():
            spec = properties[key]
            value_type = spec["type"]

            if value_type == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be text.")

                value = value.strip()

                if len(value) < spec.get("minLength", 0):
                    raise ValueError(f"{key} cannot be empty.")

                if len(value) > spec.get("maxLength", 10000):
                    raise ValueError(f"{key} is too long.")

            elif value_type == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{key} must be an integer.")

                if value < spec.get("minimum", 0):
                    raise ValueError(f"{key} is below the allowed minimum.")

                if value > spec.get("maximum", 1000000):
                    raise ValueError(f"{key} exceeds the allowed maximum.")

            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"{key} is not an allowed value.")

            clean[key] = value

        return clean

    @staticmethod
    def _has_explicit_save_request(message: str) -> bool:
        if _LOOKUP_SAVED_PATTERN.search(message or ""):
            return False
        return bool(_EXPLICIT_SAVE_PATTERN.search(message or ""))

    def _purge_expired_pending(self) -> None:
        now = time.monotonic()
        expired = [
            token
            for token, pending in self._pending.items()
            if pending.expires_at <= now
        ]
        for token in expired:
            self._pending.pop(token, None)

    @staticmethod
    def _record_label(row: dict[str, Any]) -> str:
        key = str(row.get("memory_key") or "").strip()
        kind = str(row.get("memory_type") or "record")
        record_id = row.get("id", "?")

        if key:
            return f"{kind} '{key}' (record #{record_id})"
        return f"{kind} record #{record_id}"

    def _request_mutation(
        self,
        action: str,
        row: dict[str, Any],
        new_content: str | None = None,
    ) -> dict[str, Any]:
        # if not self.allow_mutations:
        #     return {
        #         "status": "denied",
        #         "message": (
        #             "Database changes are disabled for this interface."
        #         ),
        #     }
        denial = self._mutation_denial()

        if denial:
            return denial

        with self._lock:
            self._purge_expired_pending()

            if self._pending:
                return {
                    "status": "error",
                    "message": (
                        "A database change is already awaiting confirmation. "
                        "Confirm or cancel it before requesting another change."
                    ),
                }

            token = secrets.token_urlsafe(9)
            pending = _PendingMutation(
                token=token,
                action=action,
                memory_id=int(row["id"]),
                expected_content=str(row.get("content") or ""),
                new_content=new_content,
                label=self._record_label(row),
                expires_at=(
                    time.monotonic() + self.CONFIRMATION_TTL_SECONDS
                ),
            )
            self._pending[token] = pending

        return {
            "status": "needs_confirmation",
            "action": action,
            "token": token,
            "memory_id": pending.memory_id,
            "label": pending.label,
            "before": pending.expected_content[:240],
            "after": (
                str(new_content)[:240]
                if new_content is not None
                else None
            ),
            "expires_in_seconds": self.CONFIRMATION_TTL_SECONDS,
            "message": "No database change has been applied yet.",
        }

    def _profile_save(
        self,
        memory_key: str,
        content: str,
    ) -> dict[str, Any]:
        # if not self.allow_mutations:
        #     return {
        #         "status": "denied",
        #         "message": "Database changes are disabled for this interface.",
        #     }
        denial = self._mutation_denial()

        if denial:
            return denial
        
        key = str(memory_key or "").strip().casefold()
        value = " ".join(str(content or "").split())

        if key not in PROFILE_FIELD_KEYS:
            return {
                "status": "error",
                "message": (
                    "That profile key is not in Zara's approved profile-field "
                    "list. Search/list the profile to use a supported key."
                ),
            }

        if not value or len(value) > 600:
            return {
                "status": "error",
                "message": "Profile content must contain 1–600 characters.",
            }

        try:
            existing = self.store.get_tool_managed_memory_by_key(key)

            if len(existing) > 1:
                return {
                    "status": "error",
                    "message": (
                        "Multiple active profile records use this key. "
                        "No change was made because the target is ambiguous."
                    ),
                }

            if existing:
                row = existing[0]

                if str(row.get("content") or "") == value:
                    return {
                        "status": "unchanged",
                        "message": "The profile field already has that value.",
                        "record": self._public_record(row),
                    }

                # Existing profile values are never silently overwritten.
                return self._request_mutation("update", row, value)

            created = self.store.create_tool_profile_entry_if_missing(
                key,
                value,
            )

            if created.get("status") == "created":
                return {
                    "status": "success",
                    "operation": "profile_created",
                    "message": "A new profile field was saved.",
                    "record": self._public_record(created["record"]),
                }

            concurrent_matches = created.get("records", [])
            if len(concurrent_matches) != 1:
                return {
                    "status": "error",
                    "message": (
                        "The profile field changed during saving. "
                        "No overwrite was performed."
                    ),
                }

            row = concurrent_matches[0]
            if str(row.get("content") or "") == value:
                return {
                    "status": "unchanged",
                    "message": "The profile field already has that value.",
                    "record": self._public_record(row),
                }

            return self._request_mutation("update", row, value)

        except Exception as exc:
            logger.warning(
                "Profile save tool failed (%s).",
                type(exc).__name__,
            )
            return {
                "status": "error",
                "message": "The profile change could not be verified.",
            }

    def request_profile_save(
        self,
        memory_key: str,
        content: str,
    ) -> dict[str, Any]:
        """Entry point for the explicit /profile-set terminal command."""
        return self._profile_save(memory_key, content)

    def request_profile_delete_by_key(
        self,
        memory_key: str,
    ) -> dict[str, Any]:
        """Prepare profile deletion by key; never delete immediately."""
        # if not self.allow_mutations:
        #     return {
        #         "status": "denied",
        #         "message": "Database changes are disabled for this interface.",
        #     }

        denial = self._mutation_denial()

        if denial:
            return denial

        key = str(memory_key or "").strip().casefold()

        if key not in PROFILE_FIELD_KEYS:
            return {
                "status": "error",
                "message": "That profile key is not in the approved profile list.",
            }

        try:
            records = self.store.get_tool_managed_memory_by_key(key)
        except Exception as exc:
            logger.warning(
                "Profile lookup for deletion failed (%s).",
                type(exc).__name__,
            )
            return {
                "status": "error",
                "message": "The profile record could not be read.",
            }

        if not records:
            return {
                "status": "error",
                "message": f"No active profile record was found for '{key}'.",
            }

        if len(records) != 1:
            return {
                "status": "error",
                "message": (
                    "Multiple records match this profile key. "
                    "No deletion was prepared."
                ),
            }

        return self._request_mutation("delete", records[0])

    @staticmethod
    def _topic_preview_signature(preview: dict[str, Any]) -> str:
        """Create a deterministic fingerprint of a consolidation preview."""

        groups = preview.get("exact_duplicate_groups", [])
        conflicts = preview.get("conflicting_topics", [])

        canonical = {
            "groups": sorted(
                (
                    str(item.get("topic_key") or ""),
                    str(item.get("content_hash") or ""),
                    tuple(sorted(
                        int(value)
                        for value in item.get("memory_ids", [])
                    )),
                )
                for item in groups
            ),
            "conflicts": sorted(
                (
                    str(item.get("topic_key") or ""),
                    int(item.get("distinct_versions", 0)),
                    tuple(sorted(
                        int(value)
                        for value in item.get("memory_ids", [])
                    )),
                )
                for item in conflicts
            ),
        }

        raw = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _request_topic_memory_mutation(
        self,
        *,
        action: str,
        memory_id: int,
        label: str,
        before: str = "",
        after: str | None = None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        denial = self._mutation_denial()

        if denial:
            return {
                "status": "denied",
                "message": denial["message"],
                "error_code": denial["error_code"],
            }

        with self._lock:
            self._purge_expired_pending()

            if self._pending:
                return {
                    "status": "error",
                    "message": (
                        "Another database or memory action is already "
                        "awaiting confirmation. Confirm or cancel it first."
                    ),
                    "error_code": "pending_confirmation_exists",
                }

            token = secrets.token_urlsafe(18)

            pending = _PendingMutation(
                token=token,
                action=action,
                memory_id=memory_id,
                expected_content=str(before),
                new_content=after,
                label=label[:160],
                expires_at=(
                    time.monotonic() + self.CONFIRMATION_TTL_SECONDS
                ),
                scope="topic_memory",
                payload=dict(payload),
            )

            self._pending[token] = pending

        return {
            "status": "needs_confirmation",
            "action": action,
            "token": token,
            "memory_id": memory_id or None,
            "label": pending.label,
            "before": str(before)[:320],
            "after": str(after)[:320] if after is not None else None,
            "expires_in_seconds": self.CONFIRMATION_TTL_SECONDS,
            "message": "No personal-memory change has been applied yet.",
        }

    @staticmethod
    def _topic_memory_action_is_explicit(
        tool_name: str,
        user_message: str,
    ) -> bool:
        message = str(user_message or "")

        memory_target = bool(re.search(
            r"\b(?:memories|memory|saved\s+fact|saved\s+note|"
            r"personal\s+fact|candidate|candidates)\b",
            message,
            re.IGNORECASE,
        ))

        if tool_name == "topic_memory_correct":
            return (
                bool(re.search(
                    r"\b(?:correct|update|change|edit|fix|modify|"
                    r"badal|badlo|badalna)\b",
                    message,
                    re.IGNORECASE,
                ))
                and memory_target
            )

        if tool_name == "topic_memory_delete":
            return (
                bool(re.search(
                    r"\b(?:delete|remove|forget|clear|hatao|mitao)\b",
                    message,
                    re.IGNORECASE,
                ))
                and memory_target
            )

        if tool_name == "topic_memory_restore":
            return (
                bool(re.search(
                    r"\b(?:restore|recover|reinstate|wapas\s+lao)\b",
                    message,
                    re.IGNORECASE,
                ))
                and memory_target
            )

        if tool_name == "topic_memory_candidate_approve":
            return (
                bool(re.search(
                    r"\b(?:approve|accept|promote)\b",
                    message,
                    re.IGNORECASE,
                ))
                and bool(re.search(
                    r"\b(?:candidate|candidates|pending\s+memory)\b",
                    message,
                    re.IGNORECASE,
                ))
            )

        if tool_name == "topic_memory_candidate_reject":
            return (
                bool(re.search(
                    r"\b(?:reject|decline|discard)\b",
                    message,
                    re.IGNORECASE,
                ))
                and bool(re.search(
                    r"\b(?:candidate|candidates|pending\s+memory)\b",
                    message,
                    re.IGNORECASE,
                ))
            )

        if tool_name == "topic_memory_consolidate":
            return (
                bool(re.search(
                    r"\b(?:consolidate|deduplicate|merge)\b",
                    message,
                    re.IGNORECASE,
                ))
                and bool(re.search(
                    r"\b(?:duplicate|duplicates|memories|candidates)\b",
                    message,
                    re.IGNORECASE,
                ))
            )
        if tool_name == "topic_memory_repair_apply":
            return bool(re.search(
                r"\b(?:repair|fix|theek|sudhar)\b.{0,100}"
                r"\b(?:memory|memories|memory\s+records|"
                r"personal\s+memory)\b",
                message,
                re.IGNORECASE,
            ))

        if tool_name == "topic_memory_repair_rollback":
            return bool(re.search(
                r"\b(?:rollback|undo|reverse)\b.{0,100}"
                r"\b(?:repair|memory|memories)\b",
                message,
                re.IGNORECASE,
            ))
        
        return False

    def _confirm_topic_memory_mutation(
        self,
        pending: _PendingMutation,
        *,
        started_at: float,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        manager = self.memory_manager
        payload = dict(pending.payload or {})

        base_data = {
            "action": pending.action,
            "record_id": pending.memory_id or None,
            "confirmation_consumed": True,
            "verified": False,
        }

        if manager is None:
            return self._confirmation_result(
                "database_confirm",
                "failed",
                "Personal memory service is unavailable.",
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": "operation_error",
                },
                error_code="memory_manager_unavailable",
            )

        try:
            action = pending.action

            if action == "repair":
                limit = int(payload.get("limit", 500))

                report = manager.preview_memory_repair(
                    limit=limit,
                )

                if (
                    report["plan_fingerprint"]
                    != payload.get("plan_fingerprint")
                ):
                    raise ValueError(
                        "memory_repair_preview_stale"
                    )

                result = manager.apply_memory_repairs(
                    payload["plan_fingerprint"],
                    limit=limit,
                    trace_id=pending.token,
                    cancel_event=cancel_event,
                )

                if result.get("status") == "unchanged":
                    return self._confirmation_result(
                        "database_confirm",
                        "unchanged",
                        (
                            "No repairable memory change was necessary. "
                            "Existing memory state was preserved."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "verified": True,
                            "repaired_records": 0,
                            "repaired_actions": 0,
                            "remaining_records": 0,
                            "scan_may_be_incomplete": (
                                result.get(
                                    "scan_may_be_incomplete"
                                ) is True
                            ),
                            "verification_method": "transaction_result",
                            "outcome": "no_action_applied",
                        },
                    )

                if result.get("status") != "completed":
                    raise ValueError(
                        "memory_repair_not_completed"
                    )

                remaining_records = result.get(
                    "remaining_records",
                    0,
                )

                if (
                    isinstance(remaining_records, bool)
                    or not isinstance(remaining_records, int)
                    or remaining_records < 0
                ):
                    raise ValueError(
                        "invalid_memory_repair_result"
                    )

                scan_incomplete = (
                    result.get("scan_may_be_incomplete") is True
                )

                repaired_records = int(
                    result["repaired_records"]
                )
                repaired_actions = int(
                    result["repaired_actions"]
                )

                system_message = (
                    "Memory repair batch completed. "
                    f"{repaired_records} record(s) processed."
                )

                if remaining_records:
                    system_message += (
                        f" {remaining_records} repairable "
                        "record(s) remain in the current preview."
                    )

                if scan_incomplete:
                    system_message += (
                        " The scan reached its record limit; "
                        "additional records may remain uninspected."
                    )

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    system_message,
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "repair_id": result["repair_id"],
                        "repaired_records": repaired_records,
                        "repaired_actions": repaired_actions,
                        "remaining_records": remaining_records,
                        "scan_may_be_incomplete": scan_incomplete,
                        "recoverable": True,
                        "verification_method": "transaction_result",
                        "outcome": "committed",
                    },
                )

            if action == "repair_rollback":
                repair_id = str(payload["repair_id"])

                preview = manager.preview_memory_repair_rollback(
                    repair_id
                )

                if preview["rollback_available"] < 1:
                    raise ValueError(
                        "repair_snapshot_unavailable"
                    )

                result = manager.rollback_memory_repair(
                    repair_id,
                    trace_id=pending.token,
                    cancel_event=cancel_event,
                )

                if result.get("status") == "unchanged":
                    return self._confirmation_result(
                        "database_confirm",
                        "unchanged",
                        (
                            "No unapplied repair snapshots remain. "
                            "No additional rollback was necessary."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "verified": True,
                            "repair_id": repair_id,
                            "rolled_back": 0,
                            "conflicts": 0,
                            "outcome": "no_action_applied",
                        },
                    )

                if result.get("status") not in {
                    "completed",
                    "partial",
                }:
                    raise ValueError(
                        "memory_repair_rollback_not_completed"
                    )

                rollback_complete = (
                    result["status"] == "completed"
                )

                rolled_back = int(result["rolled_back"])
                conflicts = int(result["conflicts"])

                return self._confirmation_result(
                    "database_confirm",
                    (
                        "success"
                        if rollback_complete
                        else "failed"
                    ),
                    (
                        (
                            "Memory repair rollback completed. "
                            f"{rolled_back} record(s) restored."
                        )
                        if rollback_complete
                        else (
                            f"Rollback was partial. {rolled_back} "
                            f"record(s) restored and {conflicts} "
                            "record(s) safely skipped because "
                            "their state changed."
                        )
                    ),
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": rollback_complete,
                        "repair_id": repair_id,
                        "rolled_back": rolled_back,
                        "conflicts": conflicts,
                        "verification_method": "transaction_result",
                        "outcome": (
                            "rolled_back"
                            if rollback_complete
                            else "rollback_conflict"
                        ),
                    },
                )
            
            if action not in {
                "consolidate",
                "repair",
                "repair_rollback",
            }:
                current = manager.get_memory_record(
                    pending.memory_id
                )

                if current is None:
                    raise ValueError("memory_target_not_found")

                expected_version = payload.get("expected_version")

                expected_summary_hash = payload.get(
                    "expected_summary_hash"
                )

                actual_summary_hash = hashlib.sha256(
                    str(current.get("summary") or "").encode("utf-8")
                ).hexdigest()

                if (
                    current.get("version") != expected_version
                    or actual_summary_hash != expected_summary_hash
                ):
                    raise ValueError("stale_confirmation_target")

                expected_states = {
                    "correct": ("approved", True),
                    "delete": ("approved", True),
                    "restore": ({"deleted", "superseded"}, False),
                    "approve_candidate": ("candidate", True),
                    "reject_candidate": ("candidate", True),
                }

                expected = expected_states.get(action)

                if expected is not None:
                    valid = (
                        (
                            current.get("status") in expected[0]
                            if isinstance(expected[0], set)
                            else current.get("status") == expected[0]
                        )
                        and current.get("active") is expected[1]
                    )

                    if not valid:
                        raise ValueError("memory_stale_target")

            if action == "correct":
                result = manager.correct_memory(
                    pending.memory_id,
                    str(pending.new_content or ""),
                    trace_id=pending.token,
                )

                if result.get("status") != "corrected":
                    raise ValueError(
                        str(result.get("status") or "correction_failed")
                    )

                verified = manager.get_memory_record(
                    int(result["memory_id"])
                )

                if (
                    verified is None
                    or verified.get("status") != "approved"
                    or verified.get("active") is not True
                    or verified.get("summary") != pending.new_content
                ):
                    raise ValueError("memory_correction_not_verified")

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    "Personal memory corrected and new version verified.",
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "record_id": int(result["memory_id"]),
                        "previous_memory_id": pending.memory_id,
                        "version": result.get("version"),
                        "verification_method": "read_after_write",
                        "outcome": "updated",
                    },
                )

            if action == "delete":
                result = manager.delete_memory(
                    pending.memory_id,
                    trace_id=pending.token,
                )

                verified = manager.get_memory_record(
                    pending.memory_id
                )

                if (
                    result.get("status") != "deleted"
                    or verified is None
                    or verified.get("status") != "deleted"
                    or verified.get("active") is not False
                ):
                    raise ValueError("memory_delete_not_verified")

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    (
                        "Personal memory soft-deleted and its inactive "
                        "state verified. Historical versions remain recoverable."
                    ),
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "verification_method": "read_after_write",
                        "outcome": "deleted",
                        "recoverable": True,
                    },
                )

            if action == "restore":
                result = manager.restore_memory(
                    pending.memory_id,
                    trace_id=pending.token,
                )

                if result.get("status") != "restored":
                    raise ValueError(
                        str(result.get("status") or "restore_failed")
                    )

                verified = manager.get_memory_record(
                    int(result["memory_id"])
                )

                if (
                    verified is None
                    or verified.get("status") != "approved"
                    or verified.get("active") is not True
                    or verified.get("summary") != pending.expected_content
                ):
                    raise ValueError("memory_restore_not_verified")

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    "Historical memory restored as a new verified version.",
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "record_id": int(result["memory_id"]),
                        "restored_from_id": pending.memory_id,
                        "version": result.get("version"),
                        "verification_method": "read_after_write",
                        "outcome": "restored",
                    },
                )

            if action == "approve_candidate":
                result = manager.approve_memory_candidate(
                    pending.memory_id,
                    trace_id=pending.token,
                )

                if result.get("status") not in {
                    "approved",
                    "merged_into_approved",
                    "already_approved",
                }:
                    raise ValueError(
                        str(result.get("status") or "approval_failed")
                    )

                approved_record = manager.get_memory_record(
                    int(result["memory_id"])
                )

                if (
                    approved_record is None
                    or approved_record.get("status") != "approved"
                    or approved_record.get("active") is not True
                ):
                    raise ValueError("memory_approval_not_verified")

                if result.get("status") == "merged_into_approved":
                    retired_candidate = manager.get_memory_record(
                        pending.memory_id
                    )

                    if (
                        retired_candidate is None
                        or retired_candidate.get("status") != "superseded"
                        or retired_candidate.get("active") is not False
                    ):
                        raise ValueError(
                            "candidate_consolidation_not_verified"
                        )

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    "Candidate approval/consolidation verified.",
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "record_id": int(result["memory_id"]),
                        "candidate_id": pending.memory_id,
                        "verification_method": "read_after_write",
                        "outcome": "approved",
                    },
                )

            if action == "reject_candidate":
                result = manager.reject_memory_candidate(
                    pending.memory_id,
                    reason_code=str(
                        payload.get("reason_code") or "not_useful"
                    ),
                    trace_id=pending.token,
                )

                verified = manager.get_memory_record(
                    pending.memory_id
                )

                if (
                    result.get("status") != "rejected"
                    or verified is None
                    or verified.get("status") != "rejected"
                    or verified.get("active") is not False
                ):
                    raise ValueError("candidate_rejection_not_verified")

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    "Memory candidate rejected and state verified.",
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "verification_method": "read_after_write",
                        "outcome": "rejected",
                    },
                )
            
            if action == "consolidate":
                before = manager.preview_candidate_consolidation(
                    limit=int(payload.get("limit", 100))
                )

                current_signature = self._topic_preview_signature(before)

                if current_signature != payload.get("preview_signature"):
                    raise ValueError(
                        "consolidation_preview_stale"
                    )

                result = manager.consolidate_memory_candidates(
                    topic_key=payload.get("topic_key"),
                    limit=int(payload.get("limit", 100)),
                    trace_id=pending.token,
                )

                if (
                    result.get("status") != "completed"
                    or int(result.get("records_retired", 0)) < 1
                ):
                    raise ValueError(
                        "memory_consolidation_not_verified"
                    )

                return self._confirmation_result(
                    "database_confirm",
                    "success",
                    "Exact duplicate candidate consolidation completed.",
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "groups_merged": int(
                            result.get("groups_merged", 0)
                        ),
                        "records_retired": int(
                            result.get("records_retired", 0)
                        ),
                        "verification_method": "transaction_result",
                        "outcome": "consolidated",
                    },
                )

            raise ValueError("unsupported_topic_memory_action")

        except DatabaseOperationOutcomeUnknown:
            logger.error(
                "Topic memory confirmation ended with unknown outcome."
            )

            return self._confirmation_result(
                "database_confirm",
                "failed",
                (
                    "Memory transaction outcome is uncertain. "
                    "Inspect memory history before retrying."
                ),
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": "unknown",
                },
                error_code="database_outcome_unknown",
            )

        except (DatabaseOperationCancelled, MemoryOperationCancelled):
            return self._confirmation_result(
                "database_confirm",
                "cancelled",
                "Memory operation was cancelled; final state needs verification.",
                started_at=started_at,
                data={
                    **base_data,
                    "cancelled": True,
                    "outcome": "unverified",
                },
                error_code="memory_operation_cancelled",
            )

        except Exception as exc:
            logger.warning(
                "Topic memory confirmation failed (%s).",
                type(exc).__name__,
            )

            stale = (
                str(exc) in {
                    "memory_target_not_found",
                    "stale_confirmation_target",
                    "memory_stale_target",
                    "consolidation_preview_stale",
                    "memory_repair_preview_stale",
                }
            )

            return self._confirmation_result(
                "database_confirm",
                "failed",
                (
                    "The memory changed after its preview or is no longer "
                    "eligible. No success is being reported."
                    if stale
                    else "Memory operation could not be verified. "
                         "Inspect memory history before retrying."
                ),
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": (
                        "stale_target" if stale else "operation_error"
                    ),
                },
                error_code=(
                    "stale_confirmation_target"
                    if stale
                    else "memory_operation_failed"
                ),
            )

    def _execute_topic_memory_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_message: str,
    ) -> dict[str, Any]:
        """Execute registered personal-memory operations through Python."""

        manager = self.memory_manager

        if manager is None:
            return {
                "status": "error",
                "message": "Personal memory service is unavailable.",
                "error_code": "memory_manager_unavailable",
            }

        try:
            if tool_name == "topic_memory_list":
                status = args.get("status", "approved")
                limit = args.get("limit", 20)

                rows = manager.list_memories(
                    status=status,
                    limit=limit,
                )

                return {
                    "status": "success",
                    "operation": "topic_memory_list",
                    "count": len(rows),
                    "records": rows,
                    "message": f"Retrieved {len(rows)} personal memory record(s).",
                }

            if tool_name == "topic_memory_candidates":
                rows = manager.list_memory_candidates(
                    limit=args.get("limit", 30),
                )

                return {
                    "status": "success",
                    "operation": "topic_memory_candidates",
                    "count": len(rows),
                    "records": rows,
                    "message": f"Found {len(rows)} pending candidate(s).",
                }

            if tool_name == "topic_memory_review_candidate":
                record = manager.review_memory_candidate(
                    args["memory_id"]
                )

                if record.get("status") != "pending_review":
                    return {
                        "status": (
                            "not_found"
                            if record.get("status") == "not_found"
                            else "failed"
                        ),
                        "message": (
                            "Candidate is not available for review."
                        ),
                        "error_code": str(
                            record.get("status") or "candidate_unavailable"
                        ),
                    }

                return {
                    "status": "success",
                    "operation": "topic_memory_review_candidate",
                    "record": record,
                    "message": (
                        f"Candidate #{args['memory_id']} is ready for review."
                    ),
                }

            if tool_name in {
                "topic_memory_history",
                "topic_memory_audit",
            }:
                memory_id = args["memory_id"]
                target = manager.get_memory_record(memory_id)

                if target is None:
                    return {
                        "status": "not_found",
                        "message": "Memory record was not found.",
                    }

                if target.get("sensitivity") == "restricted":
                    return {
                        "status": "denied",
                        "message": (
                            "This memory requires separate privacy review."
                        ),
                    }

                if tool_name == "topic_memory_history":
                    rows = manager.get_memory_history(memory_id)

                    rows = [
                        row for row in rows
                        if row.get("sensitivity") != "restricted"
                    ]
                    for row in rows:
                        if row.get("status") == "quarantined":
                            row["summary"] = (
                                "[Content withheld for privacy review.]"
                            )
                            
                    return {
                        "status": "success",
                        "operation": "topic_memory_history",
                        "count": len(rows),
                        "records": rows,
                        "message": (
                            f"Retrieved {len(rows)} memory version(s)."
                        ),
                    }

                rows = manager.get_memory_audit(
                    memory_id,
                    limit=args.get("limit", 50),
                )

                return {
                    "status": "success",
                    "operation": "topic_memory_audit",
                    "count": len(rows),
                    "records": rows,
                    "message": (
                        f"Retrieved {len(rows)} memory audit event(s)."
                    ),
                }

            if tool_name == "topic_memory_consolidation_preview":
                preview = manager.preview_candidate_consolidation(
                    limit=args.get("limit", 100),
                )

                return {
                    "status": "success",
                    "operation": "topic_memory_consolidation_preview",
                    "duplicate_groups": preview.get(
                        "exact_duplicate_groups", []
                    ),
                    "conflicting_topics": preview.get(
                        "conflicting_topics", []
                    ),
                    "automatic_content_rewrite": False,
                    "message": (
                        "Consolidation preview completed. "
                        "No memory was changed."
                    ),
                }

            if tool_name == "topic_memory_repair_preview":
                report = manager.preview_memory_repair(
                    limit=args.get("limit", 500),
                )

                public_findings = [
                    {
                        "memory_id": item["memory_id"],
                        "topic_key": item["topic_key"],
                        "issue_code": item["issue_code"],
                        "severity": item["severity"],
                        "repairable": item["repairable"],
                        "description": item["description"],
                    }
                    for item in report["findings"][:100]
                ]

                return {
                    "status": "success",
                    "operation": tool_name,
                    "records_scanned": report["records_scanned"],
                    "scan_limit": report["scan_limit"],
                    "scan_may_be_incomplete": (
                        report["scan_may_be_incomplete"]
                    ),
                    "issue_count": report["issue_count"],
                    "repairable_action_count": (
                        report["repairable_action_count"]
                    ),
                    "manual_review_count": (
                        report["manual_review_count"]
                    ),
                    "privacy_issue_count": (
                        report["privacy_issue_count"]
                    ),
                    "findings": public_findings,
                    "message": (
                        "Memory integrity diagnosis complete. "
                        "No database changes were made."
                    ),
                }

            if tool_name == "topic_memory_repair_apply":
                if not self._topic_memory_action_is_explicit(
                    tool_name,
                    user_message,
                ):
                    return {
                        "status": "denied",
                        "message": (
                            "Explicit user repair intent is required."
                        ),
                        "error_code": "explicit_memory_repair_required",
                    }

                denial = self._mutation_denial()

                if denial:
                    return {
                        "status": "denied",
                        "message": denial["message"],
                        "error_code": denial["error_code"],
                    }

                report = manager.preview_memory_repair(
                    limit=args.get("limit", 500),
                )

                if report["repairable_action_count"] == 0:
                    message = (
                        "No automatically repairable memory issues "
                        "were found in the inspected records. "
                        "Manual-review issues remain unchanged."
                    )

                    if report["scan_may_be_incomplete"]:
                        message += (
                            " The scan reached its limit; "
                            "additional records may remain uninspected."
                        )

                    return {
                        "status": "unchanged",
                        "records_scanned": report["records_scanned"],
                        "scan_limit": report["scan_limit"],
                        "scan_may_be_incomplete": (
                            report["scan_may_be_incomplete"]
                        ),
                        "message": message,
                    }

                return self._request_topic_memory_mutation(
                    action="repair",
                    memory_id=0,
                    label="Personal memory integrity repair",
                    before=(
                        f"{report['records_scanned']} records scanned; "
                        f"{report['issue_count']} issue(s)"
                    ),
                    after=(
                        f"Repair {report['repairable_action_count']} "
                        "safe action(s)."
                    ),
                    payload={
                        "plan_fingerprint": report["plan_fingerprint"],
                        "limit": args.get("limit", 500),
                    },
                )

            if tool_name == "topic_memory_repair_rollback":
                if not self._topic_memory_action_is_explicit(
                    tool_name,
                    user_message,
                ):
                    return {
                        "status": "denied",
                        "message": (
                            "Explicit rollback intent is required."
                        ),
                        "error_code": "explicit_rollback_required",
                    }

                denial = self._mutation_denial()

                if denial:
                    return {
                        "status": "denied",
                        "message": denial["message"],
                        "error_code": denial["error_code"],
                    }

                repair_id = args["repair_id"]

                preview = manager.preview_memory_repair_rollback(
                    repair_id
                )

                if preview["rollback_available"] == 0:
                    return {
                        "status": "unchanged",
                        "message": (
                            "No repair snapshots are available "
                            "for rollback."
                        ),
                    }

                return self._request_topic_memory_mutation(
                    action="repair_rollback",
                    memory_id=0,
                    label=f"Memory repair #{repair_id[:12]}",
                    before=(
                        f"{preview['rollback_available']} repair "
                        "snapshot(s) available"
                    ),
                    after="Restore the original snapshot state.",
                    payload={
                        "repair_id": repair_id,
                    },
                )
            
            mutation_names = {
                "topic_memory_correct",
                "topic_memory_delete",
                "topic_memory_restore",
                "topic_memory_candidate_approve",
                "topic_memory_candidate_reject",
                "topic_memory_consolidate",
                "topic_memory_repair_apply",
                "topic_memory_repair_rollback",
            }

            if tool_name not in mutation_names:
                return {
                    "status": "error",
                    "message": "Unknown personal-memory operation.",
                    "error_code": "unknown_memory_operation",
                }

            if not self._topic_memory_action_is_explicit(
                tool_name,
                user_message,
            ):
                return {
                    "status": "denied",
                    "message": (
                        "The user's message does not explicitly authorize "
                        "this type of memory change."
                    ),
                    "error_code": "explicit_memory_intent_required",
                }

            denial = self._mutation_denial()

            if denial:
                return {
                    "status": "denied",
                    "message": denial["message"],
                    "error_code": denial["error_code"],
                }

            if tool_name == "topic_memory_consolidate":
                limit = args.get("limit", 100)
                topic_key = args.get("topic_key")

                preview = manager.preview_candidate_consolidation(
                    limit=limit,
                )

                duplicate_groups = preview.get(
                    "exact_duplicate_groups",
                    [],
                )

                if not duplicate_groups:
                    return {
                        "status": "unchanged",
                        "message": (
                            "No exact duplicate candidate groups were found."
                        ),
                    }

                return self._request_topic_memory_mutation(
                    action="consolidate",
                    memory_id=0,
                    label="Exact duplicate memory candidates",
                    before=(
                        f"{len(duplicate_groups)} duplicate group(s) "
                        f"and {len(preview.get('conflicting_topics', []))} "
                        "conflicting topic(s)"
                    ),
                    after="Consolidate exact duplicates only.",
                    payload={
                        "topic_key": topic_key,
                        "limit": limit,
                        "preview_signature": (
                            self._topic_preview_signature(preview)
                        ),
                    },
                )

            memory_id = args["memory_id"]
            current = manager.get_memory_record(memory_id)

            if current is None:
                return {
                    "status": "not_found",
                    "message": "Personal memory record was not found.",
                }

            if current.get("sensitivity") == "restricted":
                return {
                    "status": "denied",
                    "message": (
                        "This memory requires separate privacy review."
                    ),
                }

            expected_state = {
                "topic_memory_correct": ("approved", True),
                "topic_memory_delete": ("approved", True),
                "topic_memory_restore": (None, False),
                "topic_memory_candidate_approve": ("candidate", True),
                "topic_memory_candidate_reject": ("candidate", True),
            }

            expected_status, expected_active = expected_state[tool_name]

            if expected_status is None:
                if current.get("status") not in {"deleted", "superseded"}:
                    return {
                        "status": "failed",
                        "message": "This record is not a restorable historical version.",
                        "error_code": "memory_not_restorable",
                    }
            elif current.get("status") != expected_status:
                return {
                    "status": "failed",
                    "message": "Memory is no longer in the required state.",
                    "error_code": "memory_stale_target",
                }

            if current.get("active") is not expected_active:
                return {
                    "status": "failed",
                    "message": "Memory target is no longer active in the expected state.",
                    "error_code": "memory_stale_target",
                }

            expected_summary = str(current.get("summary") or "")
            expected_version = int(current["version"])

            expected_summary_hash = hashlib.sha256(
                expected_summary.encode("utf-8")
            ).hexdigest()

            common_payload = {
                "expected_version": expected_version,
                "expected_summary_hash": expected_summary_hash,
            }

            label = (
                f"{current.get('topic') or 'Memory'} "
                f"(record #{memory_id}, version {expected_version})"
            )

            if tool_name == "topic_memory_correct":
                new_summary = str(args["new_summary"]).strip()

                if new_summary == expected_summary:
                    return {
                        "status": "unchanged",
                        "message": "The memory already has this exact value.",
                    }

                return self._request_topic_memory_mutation(
                    action="correct",
                    memory_id=memory_id,
                    label=label,
                    before=expected_summary,
                    after=new_summary,
                    payload=common_payload,
                )

            if tool_name == "topic_memory_delete":
                return self._request_topic_memory_mutation(
                    action="delete",
                    memory_id=memory_id,
                    label=label,
                    before=expected_summary,
                    payload=common_payload,
                )

            if tool_name == "topic_memory_restore":
                return self._request_topic_memory_mutation(
                    action="restore",
                    memory_id=memory_id,
                    label=label,
                    before=expected_summary,
                    after="Restore as a new approved version.",
                    payload=common_payload,
                )

            if tool_name == "topic_memory_candidate_approve":
                review = manager.review_memory_candidate(memory_id)

                if review.get("status") != "pending_review":
                    return {
                        "status": "failed",
                        "message": "Candidate is not eligible for approval.",
                        "error_code": "candidate_not_reviewable",
                    }

                return self._request_topic_memory_mutation(
                    action="approve_candidate",
                    memory_id=memory_id,
                    label=label,
                    before=expected_summary,
                    after="Promote to approved personal memory.",
                    payload=common_payload,
                )

            if tool_name == "topic_memory_candidate_reject":
                review = manager.review_memory_candidate(memory_id)

                if review.get("status") != "pending_review":
                    return {
                        "status": "failed",
                        "message": "Candidate is not eligible for rejection.",
                        "error_code": "candidate_not_reviewable",
                    }

                reason_code = args.get("reason_code", "not_useful")

                return self._request_topic_memory_mutation(
                    action="reject_candidate",
                    memory_id=memory_id,
                    label=label,
                    before=expected_summary,
                    after=f"Reject candidate: {reason_code}.",
                    payload={
                        **common_payload,
                        "reason_code": reason_code,
                    },
                )

            return {
                "status": "error",
                "message": "Memory action could not be prepared.",
            }

        except (ValueError, TypeError) as exc:
            return {
                "status": "error",
                "message": str(exc),
                "error_code": "invalid_memory_operation",
            }

        except (DatabaseOperationCancelled, MemoryOperationCancelled):
            return {
                "status": "cancelled",
                "message": "Memory operation was cancelled.",
                "error_code": "memory_operation_cancelled",
            }

        except DatabaseOperationOutcomeUnknown:
            return {
                "status": "failed",
                "message": (
                    "Memory database outcome is uncertain. "
                    "Check memory history before retrying."
                ),
                "error_code": "database_outcome_unknown",
                "outcome": "unknown",
            }

        except Exception as exc:
            logger.warning(
                "Topic memory tool failed (%s).",
                type(exc).__name__,
            )

            return {
                "status": "error",
                "message": "Personal-memory operation could not be verified.",
                "error_code": "memory_tool_failed",
            }
        
    def _execute_tool_validated(
        self,
        tool_name: str,
        raw_arguments: Any,
        user_message: str,
    ) -> dict[str, Any]:
        """Validate every model-requested call and execute only allowlisted tools."""
        try:
            args = self._validate_arguments(tool_name, raw_arguments)
            if tool_name.startswith("topic_memory_"):
                return self._execute_topic_memory_tool(
                    tool_name,
                    args,
                    user_message,
                )
            
            if tool_name == "memory_latest":
                scope = args.get("scope", "all")
                limit = args.get("limit", 5)
                rows = self.store.list_tool_managed_memory(scope, limit)
                records = [self._public_record(row) for row in rows]
                return {
                    "status": "success",
                    "operation": "list_latest",
                    "scope": scope,
                    "count": len(records),
                    "records": records,
                    "message": (
                        f"Found {len(records)} saved record(s)."
                        if records
                        else "No saved records were found in that scope."
                    ),
                }

            if tool_name == "memory_search":
                scope = args.get("scope", "all")
                limit = args.get("limit", 5)
                rows = self.store.search_tool_managed_memory(
                    args["query"],
                    scope,
                    limit,
                )
                records = [self._public_record(row) for row in rows]
                return {
                    "status": "success",
                    "operation": "search",
                    "scope": scope,
                    "count": len(records),
                    "records": records,
                    "message": (
                        f"Found {len(records)} matching record(s)."
                        if records
                        else "No matching saved records were found."
                    ),
                }

            if tool_name == "memory_save":
                if not self._has_explicit_save_request(user_message):
                    return {
                        "status": "denied",
                        "message": (
                            "The current message does not clearly ask to save "
                            "information. No record was written."
                        ),
                    }

                if not self.allow_mutations:
                    return {
                        "status": "denied",
                        "message": "Database changes are disabled for this interface.",
                    }

                saved = self.store.insert_tool_memory_note(
                    args["content"],
                    args.get("label", ""),
                )
                row = saved.get("record")
                return {
                    "status": "success",
                    "operation": saved["status"],
                    "message": (
                        "The note was saved."
                        if saved["status"] == "created"
                        else "The same note was already saved; no duplicate was created."
                    ),
                    "record": self._public_record(row) if row else None,
                }

            if tool_name == "profile_save":
                explicit_profile_set = bool(
                    re.fullmatch(
                        r"/profile-set\s+\S+\s+.+",
                        user_message.strip(),
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                )

                if (
                    not self._has_explicit_save_request(user_message)
                    and not explicit_profile_set
                ):
                    return {
                        "status": "denied",
                        "message": (
                            "The current message does not clearly ask to save "
                            "a profile value. No record was changed."
                        ),
                    }

                return self._profile_save(
                    args["memory_key"],
                    args["content"],
                )

            if tool_name == "memory_update":
                if not _UPDATE_REQUEST_PATTERN.search(user_message):
                    return {
                        "status": "denied",
                        "message": (
                            "The user did not clearly request an update. "
                            "No change was prepared."
                        ),
                    }

                if not self.allow_mutations:
                    return {
                        "status": "denied",
                        "message": "Database changes are disabled for this interface.",
                    }

                row = self.store.get_tool_managed_memory_by_id(
                    args["memory_id"]
                )
                if row is None:
                    return {
                        "status": "error",
                        "message": (
                            "The record was not found or is outside the "
                            "database tool's allowed scope."
                        ),
                    }

                new_content = str(args["new_content"]).strip()
                if str(row.get("content") or "") == new_content:
                    return {
                        "status": "unchanged",
                        "message": "The record already has that value.",
                        "record": self._public_record(row),
                    }

                return self._request_mutation(
                    "update",
                    row,
                    new_content,
                )

            if tool_name == "memory_delete":
                explicit_profile_delete = bool(
                    re.fullmatch(
                        r"/profile-delete\s+[a-z0-9_]+",
                        user_message.strip(),
                        flags=re.IGNORECASE,
                    )
                )

                if (
                    not _DELETE_REQUEST_PATTERN.search(user_message)
                    and not explicit_profile_delete
                ):
                    return {
                        "status": "denied",
                        "message": (
                            "The user did not clearly request deletion. "
                            "No deletion was prepared."
                        ),
                    }

                if not self.allow_mutations:
                    return {
                        "status": "denied",
                        "message": "Database changes are disabled for this interface.",
                    }

                row = self.store.get_tool_managed_memory_by_id(
                    args["memory_id"]
                )
                if row is None:
                    return {
                        "status": "error",
                        "message": (
                            "The record was not found or is outside the "
                            "database tool's allowed scope."
                        ),
                    }

                return self._request_mutation("delete", row)

            if tool_name == "document_list":
                documents = self.document_ingestor.list_documents(
                    args.get("limit", 10)
                )

                records = []
                for document in documents:
                    created_at = document.get("created_at")
                    if hasattr(created_at, "isoformat"):
                        created_at = created_at.isoformat()
                    elif created_at is not None:
                        created_at = str(created_at)

                    records.append({
                        "id": int(document["id"]),
                        "file_name": str(document["file_name"]),
                        "category": str(document["category"]),
                        "language": str(document["language"]),
                        "ingestion_status": str(document["ingestion_status"]),
                        "created_at": created_at,
                        "chunk_count": int(document["chunk_count"]),
                    })

                return {
                    "status": "success",
                    "operation": "document_list",
                    "count": len(records),
                    "records": records,
                    "message": (
                        f"Found {len(records)} indexed document(s)."
                        if records
                        else "No indexed documents were found."
                    ),
                }

            if tool_name == "document_search":
                results = self.store.semantic_search(
                    args["query"],
                    limit=30,
                )

                matches = [
                    row for row in results
                    if row.get("source") == "document_chunk"
                ]

                matches.sort(
                    key=lambda row: float(row.get("similarity", -1.0)),
                    reverse=True,
                )

                records = []
                for row in matches[:args.get("limit", 3)]:
                    try:
                        score = float(row["similarity"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    records.append({
                        "source_file": str(
                            row.get("source_file") or row.get("title") or "Unknown source"
                        ),
                        "page_number": row.get("page_number"),
                        "section": str(row.get("section") or ""),
                        "similarity": round(score, 4),
                        "content": str(row.get("content") or "")[:500],
                    })

                return {
                    "status": "success",
                    "operation": "document_search",
                    "query": args["query"],
                    "count": len(records),
                    "records": records,
                    "message": (
                        "Retrieved the best-matching indexed document excerpts."
                        if records
                        else "No matching indexed document excerpts were found."
                    ),
                }

            if tool_name == "feedback_stats":
                return {
                    "status": "success",
                    "operation": "feedback_stats",
                    "stats": self.store.feedback_stats(),
                    "message": "Feedback statistics retrieved.",
                }

            if tool_name == "feedback_list":
                rows = self.store.list_feedback(args.get("limit", 5))
                records = [self._public_feedback(row) for row in rows]

                return {
                    "status": "success",
                    "operation": "feedback_list",
                    "count": len(records),
                    "records": records,
                    "message": (
                        f"Found {len(records)} feedback record(s)."
                        if records
                        else "No feedback records were found."
                    ),
                }

            if tool_name == "feedback_get":
                row = self.store.get_feedback_by_id(args["feedback_id"])

                if row is None:
                    return {
                        "status": "error",
                        "message": f"Feedback #{args['feedback_id']} was not found.",
                    }

                return {
                    "status": "success",
                    "operation": "feedback_get",
                    "record": self._public_feedback(row),
                    "message": "Feedback record retrieved.",
                }

            if tool_name == "feedback_review":
                if not _REVIEW_FEEDBACK_PATTERN.search(user_message):
                    return {
                        "status": "denied",
                        "message": (
                            "The current message does not explicitly request feedback "
                            "approval. No update was prepared."
                    ),
                }

                row = self.store.get_feedback_by_id(args["feedback_id"])

                if row is None:
                    return {
                        "status": "error",
                        "message": f"Feedback #{args['feedback_id']} was not found.",
                    }

                if int(row.get("rating", 0)) != -1 or not str(
                    row.get("note") or ""
                ).strip():
                    return {
                        "status": "error",
                        "message": (
                            "Only bad feedback with a written correction can be approved."
                        ),
                    }

                if bool(row.get("reviewed")):
                    return {
                        "status": "unchanged",
                        "message": f"Feedback #{args['feedback_id']} is already reviewed.",
                    }

                preview_row = {
                    "id": int(row["id"]),
                    "memory_key": f"feedback-{row['id']}",
                    "memory_type": "feedback",
                    "content": str(row["note"]),
                    "source": "feedback",
                }

                return self._request_mutation("review_feedback", preview_row)

            return {
                "status": "error",
                "message": "Unknown database tool; nothing was executed.",
            }

        except (ValueError, TypeError) as exc:
            return {
                "status": "error",
                "message": str(exc),
                "error_code": "invalid_database_operation",
            }

        except DatabaseOperationOutcomeUnknown:
            logger.error(
                "Database tool '%s' ended with an uncertain transaction outcome.",
                tool_name,
            )

            return {
                "status": "error",
                "operation": tool_name,
                "message": (
                    "The database transaction outcome is uncertain. "
                    "Verify the affected record before attempting the action again."
                ),
                "error_code": "database_outcome_unknown",
                "outcome": "unknown",
                "retryable": False,
            }

        except DatabaseOperationCancelled:
            logger.info(
                "Database tool '%s' was cancelled.",
                tool_name,
            )

            return {
                "status": "cancelled",
                "operation": tool_name,
                "message": (
                    "The database operation was cancelled. "
                    "No successful commit is being reported."
                ),
                "error_code": "database_cancelled",
                "outcome": "not_committed",
                "retryable": False,
            }
        
        except Exception as exc:
            logger.warning(
                "Database tool '%s' failed (%s).",
                tool_name,
                type(exc).__name__,
            )

            return {
                "status": "error",
                "message": (
                    "The database operation could not be verified. "
                    "No success is being reported."
                ),
                "error_code": (
                    "database_timeout"
                    if (
                        getattr(exc, "sqlstate", None) == "57014"
                        or type(exc).__name__ in {
                            "QueryCanceled",
                            "QueryCanceledError",
                        }
                    )
                    else "database_operation_failed"
                ),
                "retryable": False,
            }

    def execute_tool(
        self,
        tool_name: str,
        raw_arguments: Any,
        user_message: str,
    ) -> dict[str, Any]:
        """Authorize, validate, execute and validate one registered tool."""

        decision = self.permission_policy.authorize(tool_name)

        if not decision.allowed:
            return {
                "status": "denied",
                "message": decision.message,
                "error_code": decision.code,
            }

        try:
            clean_arguments = self.registry.validate_arguments(
                tool_name,
                raw_arguments,
            )

            result = self._execute_tool_validated(
                tool_name,
                clean_arguments,
                user_message,
            )

            return self.registry.validate_result(
                tool_name,
                result,
            )

        except ToolValidationError as exc:
            logger.warning(
                "Tool request/result rejected (%s).",
                type(exc).__name__,
            )

            return {
                "status": "error",
                "message": str(exc),
                "error_code": "tool_contract_violation",
            }

        except Exception as exc:
            logger.warning(
                "Tool execution boundary failed (%s).",
                type(exc).__name__,
            )

            return {
                "status": "error",
                "message": (
                    "The tool operation could not be verified."
                ),
                "error_code": "tool_execution_failed",
            }

    def format_confirmation_message(
        self,
        result: dict[str, Any],
    ) -> str:
        action = str(result.get("action", "change")).upper()
        label = str(result.get("label", "selected record"))
        token = str(result.get("token", ""))
        seconds = int(result.get("expires_in_seconds", 120))
        before = str(result.get("before") or "")
        after = result.get("after")

        lines = [
            f"Database {action} pending — abhi koi change apply nahi hua.",
            f"Record: {label}",
            f"Current value: {before or '(empty)'}",
        ]

        if after is not None:
            lines.append(
                f"Proposed value: {str(after)[:320]}"
            )

        lines.extend([
            f"Confirm within {seconds} seconds: /db-confirm {token}",
            f"Cancel instead: /db-cancel {token}",
        ])

        return "\n".join(lines)

    def render_result(self, result: dict[str, Any]) -> str:
        status = str(result.get("status", "error"))

        stats = result.get("stats")
        if isinstance(stats, dict):
            return (
                f"Feedback total={stats.get('total', 0)}, "
                f"good={stats.get('good', 0)}, "
                f"bad={stats.get('bad', 0)}, "
                f"reviewed={stats.get('reviewed', 0)}."
            )   

        if status == "needs_confirmation":
            return self.format_confirmation_message(result)

        if status in {"error", "denied"}:
            return "Database: " + str(
                result.get("message") or "Operation was not performed."
            )

        records = result.get("records")
        if records is not None:
            if not records:
                return str(
                    result.get("message") or "No matching records found."
                )

            lines = [str(result.get("message") or "Saved records:")]

            for row in records[:10]:
                # label = self._record_label(row)
                # content = str(row.get("content") or "")[:240]
                # lines.append(f"#{row.get('id')} — {label}: {content}")
                if "file_name" in row:
                    lines.append(
                        f"Document #{row.get('id')}: {row.get('file_name')} "
                        f"[{row.get('ingestion_status')}] — "
                        f"{row.get('chunk_count', 0)} chunks"
                    )

                elif "source_file" in row:
                    page = row.get("page_number")
                    page_text = f", page {page}" if page is not None else ""
                    lines.append(
                        f"{row.get('source_file')}{page_text} "
                        f"[score={row.get('similarity', 0)}]: "
                        f"{str(row.get('content') or '')[:250]}"
                    )

                elif "rating" in row:
                    rating = "GOOD" if row["rating"] == 1 else "BAD"
                    lines.append(
                    f"Feedback #{row.get('id')} [{rating}] "
                    f"[reviewed={row.get('reviewed')}]: "
                    f"{str(row.get('note') or '')[:180]}"
                )

                elif "topic_key" in row:
                    lines.append(
                        f"Memory #{row.get('id')} — "
                        f"{row.get('topic')} "
                        f"[{row.get('status')}]: "
                        f"{str(row.get('summary') or '')[:240]}"
                    )

                else:
                    label = self._record_label(row)
                    content = str(row.get("content") or "")[:240]
                    lines.append(f"#{row.get('id')} — {label}: {content}")

            return "\n".join(lines)

        row = result.get("record")

        if isinstance(row, dict):
            if "topic_key" in row:
                return (
                    f"{result.get('message', 'Memory operation completed.')} "
                    f"Memory #{row.get('id')} — "
                    f"{row.get('topic')}: "
                    f"{str(row.get('summary') or '')[:320]}"
                )

            return (
                f"{result.get('message', 'Database operation completed.')} "
                f"Record #{row.get('id')} "
                f"({row.get('memory_key') or row.get('memory_type')})."
            )

        return str(
            result.get("message") or "No verified database result."
        )

    @staticmethod
    def _confirmation_result(
        operation: str,
        status: TaskStatus,
        system_message: str,
        *,
        started_at: float,
        data: dict[str, Any] | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        result_data = dict(data or {})

        result_data["duration_ms"] = max(
            0,
            int(round(
                (time.monotonic() - started_at) * 1000
            )),
        )

        return TaskResult(
            operation=operation,
            status=status,
            system_message=system_message,
            data=result_data,
            error_code=error_code,
        ).to_dict()
    
    def confirm_action(
        self,
        token: str,
        *,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        operation = "database_confirm"

        if cancel_event is not None and cancel_event.is_set():
            return self._confirmation_result(
                operation,
                "cancelled",
                "Confirmation was cancelled before execution.",
                started_at=started_at,
                data={
                    "verified": False,
                    "confirmation_consumed": False,
                    "outcome": "not_executed",
                },
                error_code="memory_operation_cancelled",
            )

        denial = self._mutation_denial()

        if denial:
            return self._confirmation_result(
                "database_confirm",
                "denied",
                denial["message"],
                started_at=started_at,
                data={
                    "verified": False,
                    "confirmation_consumed": False,
                    "outcome": "no_action_applied",
                },
                error_code=denial["error_code"],
            )
        token = str(token or "").strip()

        with self._lock:
            pending = self._pending.get(token)

            if pending is None or not secrets.compare_digest(
                token,
                pending.token,
            ):
                return self._confirmation_result(
                    operation,
                    "failed",
                    (
                        "Confirmation token invalid, expired, or "
                        "already used. No database change was applied."
                    ),
                    started_at=started_at,
                    data={
                        "verified": False,
                        "confirmation_consumed": False,
                        "outcome": "no_action_applied",
                    },
                    error_code="confirmation_token_invalid_or_expired",
                )

            if pending.expires_at <= time.monotonic():
                self._pending.pop(token, None)

                return self._confirmation_result(
                    operation,
                    "failed",
                    (
                        "Confirmation expired. "
                        "No database change was applied."
                    ),
                    started_at=started_at,
                    data={
                        "action": pending.action,
                        "record_id": pending.memory_id,
                        "verified": False,
                        "outcome": "confirmation_expired",
                        "confirmation_consumed": True,
                    },
                    error_code="confirmation_expired",
                )

            # One-time token: consume it before performing the action.
            self._pending.pop(token, None)

        if pending.scope == "topic_memory":
            return self._confirm_topic_memory_mutation(
                pending,
                started_at=started_at,
                cancel_event=cancel_event,
            )
        
        base_data = {
            "action": pending.action,
            "record_id": pending.memory_id,
            "verified": False,
            "confirmation_consumed": True,
        }

        try:
            if pending.action == "review_feedback":
                feedback = self.store.get_feedback_by_id(
                    pending.memory_id
                )

                if (
                    feedback is None
                    or str(feedback.get("note") or "")
                    != pending.expected_content
                    or int(feedback.get("rating", 0)) != -1
                    or bool(feedback.get("reviewed"))
                ):
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Feedback approval cancelled: the record "
                            "changed, disappeared, or is already reviewed. "
                            "No approval was applied."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "stale_target",
                        },
                        error_code="stale_confirmation_target",
                    )

                approved = self.store.review_feedback(
                    pending.memory_id
                )

                if not approved:
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Feedback approval could not be verified. "
                            "No successful approval is being reported."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "approval_not_verified",
                        },
                        error_code="feedback_approval_not_verified",
                    )

                # Read back the state before reporting success.
                verified_feedback = self.store.get_feedback_by_id(
                    pending.memory_id
                )

                if (
                    not isinstance(verified_feedback, dict)
                    or not bool(verified_feedback.get("reviewed"))
                ):
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Feedback approval was attempted, but its "
                            "saved state could not be verified."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "readback_verification_failed",
                        },
                        error_code="feedback_readback_failed",
                    )

                return self._confirmation_result(
                    operation,
                    "success",
                    (
                        f"Feedback #{pending.memory_id} approved. "
                        "The reviewed correction was stored successfully."
                    ),
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "verification_method": "read_after_write",
                        "outcome": "approved",
                    },
                )

            if pending.action == "update":
                updated = self.store.update_tool_managed_memory_by_id(
                    pending.memory_id,
                    pending.expected_content,
                    str(pending.new_content or ""),
                )

                if updated is None:
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Update apply nahi hua: record preview ke baad "
                            "badal gaya hai ya ab accessible nahi hai. "
                            "No new value was verified."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "stale_target",
                        },
                        error_code="stale_confirmation_target",
                    )

                expected_value = str(
                    pending.new_content or ""
                ).strip()

                returned_value = str(
                    updated.get("content") or ""
                )

                record_matches = (
                    str(updated.get("id")) == str(pending.memory_id)
                    and returned_value == expected_value
                )

                if not record_matches:
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Update was attempted, but the returned record "
                            "did not match the requested value. "
                            "The final state could not be verified."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "readback_verification_failed",
                        },
                        error_code="update_result_mismatch",
                    )

                return self._confirmation_result(
                    operation,
                    "success",
                    (
                        "Database update confirmed and completed for "
                        f"{self._record_label(updated)}."
                    ),
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "verification_method": "returned_row_match",
                        "outcome": "updated",
                    },
                )

            if pending.action == "delete":
                deleted = self.store.delete_tool_managed_memory_by_id(
                    pending.memory_id,
                    pending.expected_content,
                )

                if not deleted:
                    return self._confirmation_result(
                        operation,
                        "failed",
                        (
                            "Delete apply nahi hua: record preview ke baad "
                            "badal gaya hai ya ab accessible nahi hai."
                        ),
                        started_at=started_at,
                        data={
                            **base_data,
                            "outcome": "delete_not_verified",
                        },
                        error_code="delete_not_verified",
                    )

                # The store reports success only when one matching row
                # was deleted using the expected record content.
                return self._confirmation_result(
                    operation,
                    "success",
                    (
                        "Database deletion confirmed and completed for "
                        f"{pending.label}."
                    ),
                    started_at=started_at,
                    data={
                        **base_data,
                        "verified": True,
                        "verification_method": "delete_rowcount",
                        "outcome": "deleted",
                    },
                )

            return self._confirmation_result(
                operation,
                "failed",
                "Unsupported pending action. No database change was applied.",
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": "unsupported_action",
                },
                error_code="unsupported_pending_action",
            )

        except DatabaseOperationCancelled:
            logger.info(
                "Confirmed database action was cancelled (%s).",
                pending.action,
            )

            return self._confirmation_result(
                operation,
                "cancelled",
                (
                    "Database confirmation was interrupted. The final "
                    "record state could not be verified. The one-time "
                    "confirmation token has been consumed. Check the "
                    "affected record before requesting another change."
                ),
                started_at=started_at,
                data={
                    **base_data,
                    "cancelled": True,
                    "outcome": "unverified",
                },
                error_code="database_cancelled",
            )

        except DatabaseOperationOutcomeUnknown:
            logger.error(
                "Confirmed database action has an uncertain "
                "transaction outcome (%s).",
                pending.action,
            )

            return self._confirmation_result(
                operation,
                "failed",
                (
                    "The PostgreSQL transaction outcome is uncertain. "
                    "The one-time confirmation token has been consumed. "
                    "Verify the affected record before attempting "
                    "another change."
                ),
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": "unknown",
                },
                error_code="database_outcome_unknown",
            )

        except Exception as exc:
            logger.warning(
                "Confirmed database action failed (%s).",
                type(exc).__name__,
            )
            return self._confirmation_result(
                operation,
                "failed",
                (
                    "Database action could not be verified. Check the "
                    "record before attempting another change."
                ),
                started_at=started_at,
                data={
                    **base_data,
                    "outcome": "operation_error",
                },
                error_code="database_action_failed",
            )                   

    def cancel_action(self, token: str) -> dict[str, Any]:
        started_at = time.monotonic()
        operation = "database_cancel"
        token = str(token or "").strip()

        with self._lock:
            pending = self._pending.get(token)

            if pending is None or not secrets.compare_digest(
                token,
                pending.token,
            ):
                return self._confirmation_result(
                    operation,
                    "failed",
                    (
                        "Confirmation token invalid or already used; "
                        "no action was applied."
                    ),
                    started_at=started_at,
                    data={
                        "verified": False,
                        "confirmation_consumed": False,
                        "outcome": "no_action_applied",
                    },
                    error_code="confirmation_token_invalid_or_used",
                )

            if pending.expires_at <= time.monotonic():
                self._pending.pop(token, None)

                return self._confirmation_result(
                    operation,
                    "failed",
                    (
                        "Confirmation expired; the pending action "
                        "was removed without applying it."
                    ),
                    started_at=started_at,
                    data={
                        "action": pending.action,
                        "record_id": pending.memory_id,
                        "verified": True,
                        "cancelled": False,
                        "outcome": "confirmation_expired",
                        "confirmation_consumed": True,
                    },
                    error_code="confirmation_expired",
                )

            # Consume the one-time pending entry before returning.
            self._pending.pop(token, None)

        return self._confirmation_result(
            operation,
            "cancelled",
            (
                f"Pending {pending.action} cancelled. "
                "No database change was applied."
            ),
            started_at=started_at,
            data={
                "action": pending.action,
                "record_id": pending.memory_id,
                "verified": True,
                "cancelled": True,
                "confirmation_consumed": True,
                "verification_method": "pending_registry_removal",
                "outcome": "pending_action_cancelled",
            },
        )