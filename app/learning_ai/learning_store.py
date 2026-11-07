"""PostgreSQL-backed learning store for Zara."""
from __future__ import annotations

import re
import logging
import hashlib 
import math
from typing import Any, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from contextvars import ContextVar
from threading import Lock

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.learning_ai.embedding_client import EmbeddingClient


logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}|[\u0900-\u097F]{2,}")


INITIAL_GUIDES = [
    {
        "key": "intent_context",
        "category": "intent",
        "title": "Intent and conversation context",
        "keywords": (
            "intent meaning goal request context followup previous "
            "conversation correction clarify usme isko phir"
        ),
        "content": (
            "Understand the user's immediate goal using the current message "
            "and relevant conversation context. Resolve references from prior "
            "turns when possible. Do not answer an assumed question instead "
            "of the actual one. Ask for clarification only when ambiguity "
            "materially changes the response."
        ),
    },
    {
        "key": "emotional_response",
        "category": "emotion",
        "title": "Emotion and helpful response",
        "keywords": (
            "emotion emotional frustration frustrated irritate angry anger "
            "gussa ladai upset sad sadness worried worry excited mood "
            "feeling stress coding fail problem"
        ),
        "content": (
            "Consider the whole message and context before interpreting an "
            "emotion. Treat uncertain interpretations as estimates. When "
            "appropriate, acknowledge the emotion and adapt the response to "
            "the user's actual need. Do not assume the cause of a problem or "
            "replace useful help with generic empathy."
        ),
    },
    {
        "key": "natural_conversation",
        "category": "conversation",
        "title": "Natural conversation flow",
        "keywords": (
            "hello hi greeting namaste friend friendship friendly dosti "
            "repeat repeated filler emoji awkward conversation social "
            "robotic cold generic"
        ),
        "content": (
            "Respond to the kind of message the user actually sent. A greeting "
            "usually needs a simple greeting; personal sharing may need "
            "acknowledgement; a technical request needs relevant help. Do not "
            "restart the conversation, repeat greetings, or default to a "
            "generic help offer. Avoid habitual filler, unnecessary questions, "
            "and repetitive emojis. Be friendly without claiming to be human."
        ),
    },
    {
        "key": "grammar_and_roles",
        "category": "language",
        "title": "Grammar and language roles",
        "keywords": (
            "grammar spelling sentence language hinglish hindi english "
            "feminine masculine pronoun words writing typo"
        ),
        "content": (
            "Use natural, grammatically consistent language in the requested "
            "language and script. Keep tense, pronouns, and sentence structure "
            "consistent. Feminine self-reference describes Zara, not the user. "
            "Do not infer the user's gender. Preserve technical names and "
            "spellings accurately."
        ),
    },
    {
        "key": "addressing_preference",
        "category": "preference",
        "title": "Addressing preferences",
        "keywords": (
            "boss name title address preference call me refer addressing "
            "personal name"
        ),
        "content": (
            "Follow an explicit application-level addressing preference "
            "naturally and sparingly. Do not force a title into every sentence. "
            "Do not use a personal name unless explicitly requested. A preferred "
            "title is not identity verification."
        ),
    },
    {
        "key": "answer_review",
        "category": "quality",
        "title": "Construct and review the response",
        "keywords": (
            "answer response quality relevant correct concise short words "
            "length format accurate repeat mistake correction"
        ),
        "content": (
            "Choose the response based on the user's intent, context, emotional "
            "tone, and explicit constraints. Check relevance, tone, grammar, "
            "spelling, requested format, and repetition. Never add a question "
            "or advice merely out of habit."
        ),
    },
]


_FEEDBACK_CATEGORY_TERMS = {
    "natural_conversation": {
        "robotic", "robot", "natural", "cold", "awkward", "generic",
        "filler", "ending", "unnecessary", "repeated", "repeat",
        "friend", "friendship", "friendly", "dosti", "baat", "baate",
        "baatein", "bate", "baten", "chat", "talk", "hello", "hi",
        "greeting", "greetings", "conversation", "social", "warmth",
    },
    "language_quality": {
        "grammar", "spelling", "hinglish", "hindi", "english",
        "sentence", "word", "typo", "language", "roman",
        "pronoun", "pronouns",
    },
    "emotion_handling": {
        "emotion", "empathy", "empathetic", "frustrated",
        "frustration", "irritate", "irritated", "angry", "anger",
        "gussa", "sad", "dukhi", "mood", "tension", "pareshan",
        "upset", "dismissive", "hurt", "worried", "worry",
    },
    "intent_context": {
        "intent", "context", "unrelated", "misunderstood", "ignored",
        "question", "request", "followup", "continue", "usme",
        "isko", "wahi", "phir", "clarify", "meaning",
    },
}


_FEEDBACK_CATEGORY_LESSONS = {
    "natural_conversation": (
        "Respond directly to the user's latest conversational intent. "
        "Do not repeat a greeting in place of a relevant reply, copy an "
        "unhelpful previous answer, or default to a generic help offer. "
        "Friendly conversation is welcome. Avoid unnecessary closing questions."
    ),
    "language_quality": (
        "Use natural, grammatically consistent language in the requested "
        "language. Check spelling, sentence structure, tense, and pronouns. "
        "Zara's feminine grammar does not determine the user's gender."
    ),
    "emotion_handling": (
        "When emotion is relevant, acknowledge it proportionately and shape "
        "the response around the user's actual need. Do not assume the cause "
        "or substitute generic empathy for useful understanding."
    ),
    "intent_context": (
        "Use relevant conversation context to understand the latest request "
        "and its follow-ups. Answer the actual intent instead of a guessed "
        "question. Ask for clarification only when necessary."
    ),
    "general_quality": (
        "Apply reviewed response-quality corrections when relevant. Avoid "
        "repeating the identified mistake, but do not overgeneralize a single "
        "rating to unrelated conversations."
    ),
}


def _tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value or "")
    }


def _feedback_categories(text: str) -> set[str]:
    tokens = _tokens(text)
    return {
        category
        for category, terms in _FEEDBACK_CATEGORY_TERMS.items()
        if tokens & terms
    }

def _vector_literal(values: list[float]) -> str:
    """Serialize a validated vector for a parameterized PostgreSQL cast."""
    if len(values) != settings.zara_embedding_dimensions:
        raise ValueError(
            f"Expected {settings.zara_embedding_dimensions} vector dimensions."
        )

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Vector contains NaN or infinity.")

    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

_TOOL_MANAGED_SCOPE_SQL = {
    "profile": (
        "(memory_type = 'profile' "
        "AND source = 'user_profile_cli')"
    ),
    "notes": (
        "(memory_type = 'note' "
        "AND source = 'zara_memory_tool')"
    ),
    "all": (
        "((memory_type = 'profile' "
        "AND source = 'user_profile_cli') "
        "OR (memory_type = 'note' "
        "AND source = 'zara_memory_tool'))"
    ),
}

class DatabaseOperationCancelled(RuntimeError):
    """The request was cancelled before its transaction committed."""


class DatabaseOperationOutcomeUnknown(RuntimeError):
    """The final transaction outcome could not be established."""


@dataclass(slots=True)
class _ConnectionLease:
    connection: psycopg.Connection
    phase: str = "active"
    cancel_requested_during_commit: bool = False


@dataclass(slots=True)
class _OperationLease:
    cancelled: bool = False
    finished: bool = False
    scope_count: int = 0
    cancel_requests: int = 0
    connections: dict[int, _ConnectionLease] = field(
        default_factory=dict
    )


class _TrackedCursor:
    """Guard SQL execution through an operation-owned connection."""

    def __init__(
        self,
        store: "LearningStore",
        connection: psycopg.Connection,
        operation_id: str | None,
        cursor: Any,
    ) -> None:
        self._store = store
        self._connection = connection
        self._operation_id = operation_id
        self._cursor = cursor

    def __enter__(self) -> "_TrackedCursor":
        self._cursor.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> Any:
        return self._cursor.__exit__(
            exc_type,
            exc,
            traceback,
        )

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._store._execute_statement(
            self._operation_id,
            self._connection,
            self._cursor.execute,
            *args,
            **kwargs,
        )

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        return self._store._execute_statement(
            self._operation_id,
            self._connection,
            self._cursor.executemany,
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _TrackedConnection:
    """Track SQL dispatch, transaction finalization and cleanup."""

    def __init__(
        self,
        store: "LearningStore",
        connection: psycopg.Connection,
        operation_id: str | None,
    ) -> None:
        self._store = store
        self._connection = connection
        self._operation_id = operation_id

    def __enter__(self) -> "_TrackedConnection":
        self._connection.__enter__()
        return self

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._store._execute_statement(
            self._operation_id,
            self._connection,
            self._connection.execute,
            *args,
            **kwargs,
        )

    def cursor(self, *args: Any, **kwargs: Any) -> _TrackedCursor:
        self._store._ensure_operation_active(
            self._operation_id,
            self._connection,
        )

        cursor = self._connection.cursor(*args, **kwargs)

        return _TrackedCursor(
            self._store,
            self._connection,
            self._operation_id,
            cursor,
        )

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> Any:
        operation_id = self._operation_id

        if operation_id is None:
            return self._connection.__exit__(
                exc_type,
                exc,
                traceback,
            )

        try:
            force_rollback = self._store._prepare_connection_exit(
                operation_id,
                self._connection,
                has_exception=exc_type is not None,
            )

            if exc_type is not None:
                try:
                    result = self._connection.__exit__(
                        exc_type,
                        exc,
                        traceback,
                    )
                except BaseException as cleanup_error:
                    if self._store._commit_cancel_raced(
                        operation_id,
                        self._connection,
                    ):
                        raise DatabaseOperationOutcomeUnknown(
                            "Cancellation overlapped transaction finalization."
                        ) from cleanup_error
                    raise

                if self._store._operation_cancelled(operation_id):
                    raise DatabaseOperationCancelled(
                        "PostgreSQL transaction was cancelled and rolled back."
                    ) from exc

                return result

            if force_rollback:
                cancellation = DatabaseOperationCancelled(
                    "PostgreSQL operation cancelled before commit."
                )

                try:
                    self._connection.__exit__(
                        DatabaseOperationCancelled,
                        cancellation,
                        traceback,
                    )
                except BaseException as rollback_error:
                    raise DatabaseOperationOutcomeUnknown(
                        "Cancellation occurred but rollback could not be confirmed."
                    ) from rollback_error

                raise cancellation

            try:
                result = self._connection.__exit__(
                    None,
                    None,
                    None,
                )
            except BaseException as commit_error:
                if self._store._commit_cancel_raced(
                    operation_id,
                    self._connection,
                ):
                    raise DatabaseOperationOutcomeUnknown(
                        "Cancellation raced with transaction finalization."
                    ) from commit_error
                raise

            self._store._mark_connection_committed(
                operation_id,
                self._connection,
            )

            return result

        finally:
            self._store._unregister_connection(
                operation_id,
                self._connection,
            )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class LearningStore:
    """PostgreSQL source of truth for behavior guidance and reviewed feedback."""

    def __init__(
        self,
        *,
        initialize: bool = True,
    ) -> None:
        if not isinstance(initialize, bool):
            raise TypeError(
                "initialize must be boolean."
            )

        self._operation_id: ContextVar[str | None] = ContextVar(
            "zara_learning_store_operation_id",
            default=None,
        )

        self._operations: dict[str, _OperationLease] = {}
        self._active_connections_lock = Lock()

        self.embedding_client = EmbeddingClient()

        if initialize:
            self._initialize()

    def begin_operation(self, operation_id: str) -> None:
        """Register a unique request before it can access PostgreSQL."""
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or operation_id != operation_id.strip()
            or len(operation_id) > 128
        ):
            raise ValueError("Invalid PostgreSQL operation ID.")

        with self._active_connections_lock:
            if operation_id in self._operations:
                raise RuntimeError(
                    "PostgreSQL operation ID is already active."
                )

            self._operations[operation_id] = _OperationLease()

    def _retire_operation_locked(
        self,
        operation_id: str,
        state: _OperationLease,
    ) -> None:
        if (
            state.finished
            and state.scope_count == 0
            and not state.connections
            and state.cancel_requests == 0
            and self._operations.get(operation_id) is state
        ):
            self._operations.pop(operation_id, None)

    def end_operation(self, operation_id: str) -> None:
        """Close an operation after its scopes and connections unwind."""
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return

            state.finished = True

            self._retire_operation_locked(
                operation_id,
                state,
            )

    @contextmanager
    def operation_scope(self, operation_id: str) -> Iterator[None]:
        """Associate database connections with one active request."""
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None or state.finished:
                raise RuntimeError(
                    "PostgreSQL operation was not registered or is finished."
                )

            state.scope_count += 1

        token = self._operation_id.set(operation_id)

        try:
            yield
        finally:
            self._operation_id.reset(token)

            with self._active_connections_lock:
                state = self._operations.get(operation_id)

                if state is not None:
                    state.scope_count -= 1

                    self._retire_operation_locked(
                        operation_id,
                        state,
                    )

    def _ensure_operation_active(
        self,
        operation_id: str | None,
        connection: psycopg.Connection,
    ) -> None:
        if operation_id is None:
            return

        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if (
                state is None
                or state.finished
                or state.cancelled
            ):
                raise DatabaseOperationCancelled(
                    "PostgreSQL operation is no longer active."
                )

            lease = state.connections.get(id(connection))

            if lease is None or lease.phase != "active":
                raise RuntimeError(
                    "PostgreSQL connection is not accepting SQL commands."
                )

    def _begin_statement(
        self,
        operation_id: str | None,
        connection: psycopg.Connection,
    ) -> bool:
        if operation_id is None:
            return False

        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if (
                state is None
                or state.finished
                or state.cancelled
            ):
                raise DatabaseOperationCancelled(
                    "PostgreSQL operation was cancelled before SQL execution."
                )

            lease = state.connections.get(id(connection))

            if lease is None or lease.phase != "active":
                raise RuntimeError(
                    "Invalid or overlapping PostgreSQL statement."
                )

            lease.phase = "executing"
            return True

    def _end_statement(
        self,
        operation_id: str,
        connection: psycopg.Connection,
    ) -> None:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return

            lease = state.connections.get(id(connection))

            if lease is not None and lease.phase == "executing":
                lease.phase = "active"

    def _execute_statement(
        self,
        operation_id: str | None,
        connection: psycopg.Connection,
        executor: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        tracked = self._begin_statement(
            operation_id,
            connection,
        )

        try:
            return executor(*args, **kwargs)
        finally:
            if tracked and operation_id is not None:
                self._end_statement(
                    operation_id,
                    connection,
                )

    def _register_connection(
        self,
        operation_id: str,
        connection: psycopg.Connection,
    ) -> bool:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if (
                state is None
                or state.finished
                or state.cancelled
            ):
                return False

            state.connections[id(connection)] = _ConnectionLease(
                connection=connection,
            )

            return True

    def _prepare_connection_exit(
        self,
        operation_id: str,
        connection: psycopg.Connection,
        *,
        has_exception: bool,
    ) -> bool:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            lease = (
                state.connections.get(id(connection))
                if state is not None
                else None
            )

            # Missing tracking state must never permit a commit.
            if state is None or lease is None:
                return True

            if state.cancelled:
                lease.phase = "rolling_back"
                return True

            if has_exception:
                lease.phase = "rolling_back"
                return False

            lease.phase = "committing"
            return False

    def _operation_cancelled(self, operation_id: str) -> bool:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            # Fail closed if the operation unexpectedly disappeared.
            return state is None or state.cancelled

    def _commit_cancel_raced(
        self,
        operation_id: str,
        connection: psycopg.Connection,
    ) -> bool:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return True

            lease = state.connections.get(id(connection))

            return bool(
                lease is None
                or (
                    lease.phase == "committing"
                    and lease.cancel_requested_during_commit
                )
            )

    def _mark_connection_committed(
        self,
        operation_id: str,
        connection: psycopg.Connection,
    ) -> None:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return

            lease = state.connections.get(id(connection))

            if lease is not None:
                lease.phase = "committed"

    def _unregister_connection(
        self,
        operation_id: str,
        connection: psycopg.Connection,
    ) -> None:
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return

            state.connections.pop(id(connection), None)

            self._retire_operation_locked(
                operation_id,
                state,
            )

    def cancel_operation(self, operation_id: str) -> int:
        """
        Make cancellation sticky, then request cancellation of active SQL.

        Return value means client-side cancellation requests accepted,
        not proof that PostgreSQL stopped the query.
        """
        with self._active_connections_lock:
            state = self._operations.get(operation_id)

            if state is None:
                return 0

            state.cancelled = True
            state.cancel_requests += 1

            connections: list[psycopg.Connection] = []

            for lease in state.connections.values():
                if lease.phase in {"executing", "committing"}:
                    if lease.phase == "committing":
                        lease.cancel_requested_during_commit = True

                    connections.append(lease.connection)

        requested = 0

        try:
            for connection in connections:
                try:
                    cancel_safe = getattr(
                        connection,
                        "cancel_safe",
                        None,
                    )

                    if callable(cancel_safe):
                        cancel_safe(timeout=1.0)
                    else:
                        cancel = getattr(
                            connection,
                            "cancel",
                            None,
                        )

                        if not callable(cancel):
                            continue

                        cancel()

                    requested += 1

                except Exception as exc:
                    logger.warning(
                        "PostgreSQL cancellation request failed (%s).",
                        type(exc).__name__,
                    )

        finally:
            with self._active_connections_lock:
                current = self._operations.get(operation_id)

                if current is state:
                    state.cancel_requests -= 1

                    self._retire_operation_locked(
                        operation_id,
                        state,
                    )

        return requested

    def _connect(self) -> _TrackedConnection:
        operation_id = self._operation_id.get()

        if operation_id is not None:
            with self._active_connections_lock:
                state = self._operations.get(operation_id)

                if (
                    state is None
                    or state.finished
                    or state.cancelled
                ):
                    raise DatabaseOperationCancelled(
                        "PostgreSQL operation is no longer active."
                    )

        password = settings.zara_database_password

        if not password:
            raise RuntimeError(
                "ZARA_DATABASE_PASSWORD is missing. "
                "Set it in the local .env file."
            )

        try:
            connection = psycopg.connect(
                host=settings.zara_database_host,
                port=settings.zara_database_port,
                dbname=settings.zara_database_name,
                user=settings.zara_database_user,
                password=password,
                connect_timeout=settings.zara_database_connect_timeout,
                options=(
                    "-c statement_timeout="
                    f"{settings.zara_database_statement_timeout_ms}"
                ),
                row_factory=dict_row,
            )

        except psycopg.OperationalError as exc:
            raise RuntimeError(
                "Could not connect to the Zara PostgreSQL database. "
                "Check the PostgreSQL service and ZARA_DATABASE_* settings."
            ) from exc

        if operation_id is not None:
            try:
                registered = self._register_connection(
                    operation_id,
                    connection,
                )

                if not registered:
                    raise DatabaseOperationCancelled(
                        "PostgreSQL operation was cancelled during connection setup."
                    )

            except BaseException:
                try:
                    connection.close()
                except Exception as close_error:
                    logger.warning(
                        "PostgreSQL connection cleanup failed (%s).",
                        type(close_error).__name__,
                    )
                raise

        return _TrackedConnection(
            self,
            connection,
            operation_id,
        )

    def _initialize(self) -> None:
        dimensions = settings.zara_embedding_dimensions

        with self._connect() as connection:
            extension = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()

            if extension is None:
                raise RuntimeError(
                    f"pgvector is not enabled in database "
                    f"{settings.zara_database_name!r}. "
                    "Run CREATE EXTENSION IF NOT EXISTS vector "
                    "in the Zara database using pgAdmin."
                )

            extension_version = str(extension["extversion"])

            version_match = re.match(
                r"^(\d+)\.(\d+)",
                extension_version,
            )

            vector_extension_version = (
                tuple(int(part) for part in version_match.groups())
                if version_match else (0, 0)
            )

            # Current structured guidance.
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS behavior_guides (
                    id BIGSERIAL PRIMARY KEY,
                    guide_key TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    keywords TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    approved BOOLEAN NOT NULL DEFAULT TRUE,
                    source_type TEXT NOT NULL DEFAULT 'built_in',
                    version INTEGER NOT NULL DEFAULT 1,
                    embedding vector({dimensions}),
                    embedding_model TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # User ratings and exact historical answers.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_message TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
                    note TEXT NOT NULL DEFAULT '',
                    reviewed BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )

            # General lessons derived from reviewed feedback.
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS feedback_lessons (
                    id BIGSERIAL PRIMARY KEY,
                    feedback_id BIGINT NOT NULL
                        REFERENCES feedback(id) ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    lesson_text TEXT NOT NULL,
                    approved BOOLEAN NOT NULL DEFAULT FALSE,
                    embedding vector({dimensions}),
                    embedding_model TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (feedback_id, category)
                )
                """
            )

            connection.execute(
                """
                ALTER TABLE behavior_guides
                ADD COLUMN IF NOT EXISTS embedding_text_hash TEXT
                """
            )

            connection.execute(
                """
                ALTER TABLE feedback_lessons
                ADD COLUMN IF NOT EXISTS embedding_text_hash TEXT
                """
            )

            # Repeatable evaluation cases.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_cases (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    criteria TEXT NOT NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )

            # Document ingestion foundation.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    source_path TEXT NOT NULL DEFAULT '',
                    file_name TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL DEFAULT 'general',
                    language TEXT NOT NULL DEFAULT 'unknown',
                    ingestion_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # R1: stable document source identity and immutable versions.
            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS source_key TEXT NOT NULL DEFAULT ''
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS source_version INTEGER NOT NULL DEFAULT 1
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS supersedes_document_id BIGINT
                    REFERENCES documents(id) ON DELETE SET NULL
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS duplicate_of_document_id BIGINT
                    REFERENCES documents(id) ON DELETE SET NULL
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS extracted_text_sha256 TEXT NOT NULL DEFAULT ''
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT NOT NULL DEFAULT 0
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS media_type TEXT NOT NULL
                    DEFAULT 'application/octet-stream'
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS source_mtime_ns BIGINT
            """)
            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS extraction_method TEXT
                    NOT NULL DEFAULT 'legacy'
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS extraction_status TEXT
                    NOT NULL DEFAULT 'unknown'
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS extraction_warnings JSONB
                    NOT NULL DEFAULT '[]'::jsonb
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS extraction_metadata JSONB
                    NOT NULL DEFAULT '{}'::jsonb
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS source_encoding TEXT
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS page_count INTEGER
            """)

            # R7: auditable deletion, retention and integrity lifecycle.

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS deleted_by TEXT
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS purged_at TIMESTAMPTZ
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS integrity_status TEXT
                    NOT NULL DEFAULT 'unchecked'
            """)

            connection.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS last_integrity_check_at
                    TIMESTAMPTZ
            """)

            # content_sha256 identifies content, not source identity.
            # Remove the previous global uniqueness constraint.
            connection.execute("""
                DO $zara_document_identity$
                DECLARE constraint_name TEXT;
                BEGIN
                    FOR constraint_name IN
                        SELECT conname
                        FROM pg_constraint
                        WHERE conrelid = 'documents'::regclass
                          AND contype = 'u'
                          AND pg_get_constraintdef(oid)
                              = 'UNIQUE (content_sha256)'
                    LOOP
                        EXECUTE format(
                            'ALTER TABLE documents DROP CONSTRAINT %I',
                            constraint_name
                        );
                    END LOOP;
                END
                $zara_document_identity$;
            """)

            # Backfill existing path-based document identities.
            connection.execute("""
                UPDATE documents
                SET source_key = MD5(
                    LOWER(
                        REPLACE(TRIM(source_path), chr(92), '/')
                    )
                )
                WHERE source_key = ''
                  AND COALESCE(source_path, '') <> ''
            """)

            # Assign deterministic versions to legacy documents.
            connection.execute("""
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_key
                            ORDER BY created_at ASC, id ASC
                        ) AS new_version
                    FROM documents
                    WHERE source_key <> ''
                )
                UPDATE documents AS document
                SET source_version = ranked.new_version
                FROM ranked
                WHERE document.id = ranked.id
            """)

            # Keep only the newest ready version active per source.
            connection.execute("""
                WITH ranked AS (
                    SELECT
                        id,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_key
                            ORDER BY updated_at DESC, id DESC
                        ) AS version_rank
                    FROM documents
                    WHERE source_key <> ''
                      AND ingestion_status = 'ready'
                )
                UPDATE documents AS document
                SET ingestion_status = 'superseded'
                FROM ranked
                WHERE document.id = ranked.id
                  AND ranked.version_rank > 1
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_content_sha256
                ON documents(content_sha256)
            """)

            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_documents_source_version
                ON documents(source_key, source_version)
                WHERE source_key <> ''
            """)

            connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_documents_one_active_source
                ON documents(source_key)
                WHERE source_key <> ''
                  AND ingestion_status = 'ready'
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_source_status
                ON documents(source_key, ingestion_status)
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_retention_due
                ON documents(retention_until, id)
                WHERE ingestion_status = 'deleted'
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_integrity_status
                ON documents(integrity_status, id)
                WHERE ingestion_status IN (
                    'ready',
                    'archived',
                    'superseded'
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_retention_due
                ON documents(retention_until, id)
                WHERE ingestion_status = 'deleted'
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_documents_integrity_status
                ON documents(integrity_status, id)
                WHERE ingestion_status IN (
                    'ready',
                    'archived',
                    'superseded'
                )
            """)
            
            # R1: auditable document lifecycle.
            connection.execute("""
                CREATE TABLE IF NOT EXISTS document_ingestion_events (
                    id BIGSERIAL PRIMARY KEY,

                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE RESTRICT,

                    previous_document_id BIGINT
                        REFERENCES documents(id) ON DELETE RESTRICT,

                    source_key TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'document_ingestor',

                    details JSONB NOT NULL DEFAULT '{}'::jsonb,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_document_ingestion_events_document
                ON document_ingestion_events(
                    document_id,
                    created_at DESC
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_document_ingestion_events_type
                ON document_ingestion_events(
                    event_type,
                    created_at DESC
                )
            """)

            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    page_number INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({dimensions}),
                    embedding_model TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (document_id, chunk_index)
                )
                """
            )

            # R4: keep embedding provenance explicit and comparable.
            # Existing vectors remain marked as legacy-unverified.
            for table_name in (
                "behavior_guides",
                "feedback_lessons",
                "document_chunks",
            ):
                connection.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS embedding_model_version
                        TEXT NOT NULL DEFAULT 'legacy-unverified'
                """)

                connection.execute(f"""
                    ALTER TABLE {table_name}
                    ALTER COLUMN embedding_model_version
                    SET DEFAULT 'unverified'
                """)

                connection.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS embedding_dimensions
                        INTEGER NOT NULL DEFAULT {dimensions}
                """)

                connection.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS embedding_format_version
                        TEXT NOT NULL DEFAULT '{self.embedding_client.format_version}'
                """)

            connection.execute("""
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS embedding_text_hash
                    TEXT NOT NULL DEFAULT ''
            """)

            # Durable cache for completed document embedding batches.
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS document_embedding_cache (
                    embedding_text_hash TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_model_version TEXT NOT NULL,
                    embedding_dimensions INTEGER NOT NULL
                        CHECK (embedding_dimensions > 0),
                    embedding_format_version TEXT NOT NULL,
                    embedding vector({dimensions}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (
                        embedding_text_hash,
                        embedding_model,
                        embedding_model_version,
                        embedding_dimensions,
                        embedding_format_version
                    )
                )
            """)

            # Existing vector typmods must match the configured dimensions.
            for table_name in (
                "behavior_guides",
                "feedback_lessons",
                "document_chunks",
                "document_embedding_cache",
            ):
                vector_type = connection.execute("""
                    SELECT format_type(atttypid, atttypmod) AS declared_type
                    FROM pg_attribute
                    WHERE attrelid = %s::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped
                """, (table_name,)).fetchone()

                expected_type = f"vector({dimensions})"

                actual_type = (
                    str(vector_type["declared_type"])
                    if vector_type is not None else "missing"
                )

                if actual_type != expected_type:
                    raise RuntimeError(
                        f"Embedding dimension mismatch in {table_name}: "
                        f"database has {actual_type}, configuration expects "
                        f"{expected_type}. A deliberate vector migration "
                        "is required; Zara will not silently cast or "
                        "truncate vectors."
                    )
                
            # Approved, persistent user memory; not yet auto-populated.
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id BIGSERIAL PRIMARY KEY,
                    memory_key TEXT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    approved BOOLEAN NOT NULL DEFAULT FALSE,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    embedding vector({dimensions}),
                    embedding_model TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
                """
            )

            # Separate long-term topic memory with explicit provenance.
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS user_topic_memories (
                    id BIGSERIAL PRIMARY KEY,
                    topic_key TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK (
                        memory_type IN (
                            'preference', 'goal', 'project', 'interest',
                            'learning', 'important_experience', 'ongoing_topic'
                        )
                    ),
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'candidate' CHECK (
                        status IN (
                            'candidate', 'approved', 'rejected',
                            'superseded', 'deleted'
                        )
                    ),
                    source_kind TEXT NOT NULL CHECK (
                        source_kind IN (
                            'user_requested',
                            'automatic_extraction',
                            'user_corrected'
                        )
                    ),
                    provenance JSONB NOT NULL DEFAULT '[]'::jsonb,
                    importance SMALLINT NOT NULL
                        CHECK (importance BETWEEN 1 AND 5),
                    confidence DOUBLE PRECISION NOT NULL
                        CHECK (confidence BETWEEN 0 AND 1),
                    sensitivity TEXT NOT NULL DEFAULT 'normal' CHECK (
                        sensitivity IN (
                            'normal', 'personal', 'sensitive', 'restricted'
                        )
                    ),
                    content_hash TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    supersedes_id BIGINT
                        REFERENCES user_topic_memories(id),
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    embedding vector({dimensions}),
                    embedding_model TEXT,
                    embedding_text_hash TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_used_at TIMESTAMPTZ,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    expires_at TIMESTAMPTZ,
                    UNIQUE (topic_key, version)
                )
                """
            )

            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    idx_topic_memory_one_current_version
                ON user_topic_memories(topic_key)
                WHERE status = 'approved' AND active = TRUE
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_topic_memory_status
                ON user_topic_memories(status, active, updated_at DESC)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_memory_hash
                ON user_topic_memories(topic_key, content_hash)
                WHERE active = TRUE
                """
            )          

            # Auditable lifecycle events for versioned personal memories.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_topic_memory_events (
                    id BIGSERIAL PRIMARY KEY,

                    topic_key TEXT NOT NULL,

                    memory_id BIGINT
                        REFERENCES user_topic_memories(id)
                        ON DELETE RESTRICT,

                    previous_memory_id BIGINT
                        REFERENCES user_topic_memories(id)
                        ON DELETE RESTRICT,

                    action TEXT NOT NULL CHECK (
                        action IN (
                            'created',
                            'candidate_created',
                            'auto_approved',
                            'merged',
                            'corrected',
                            'deleted',
                            'restored'
                        )
                    ),

                    source_kind TEXT NOT NULL CHECK (
                        source_kind IN (
                            'user_requested',
                            'automatic_extraction',
                            'user_corrected'
                        )
                    ),

                    trace_id TEXT NOT NULL DEFAULT '',

                    details JSONB NOT NULL DEFAULT '{}'::jsonb,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_topic_memory_events_memory
                ON user_topic_memory_events(
                    memory_id,
                    created_at DESC
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_topic_memory_events_topic
                ON user_topic_memory_events(
                    topic_key,
                    created_at DESC
                )
                """
            )

            # B6: Extend memory status without removing existing states.
            connection.execute(
                """
                DO $migration$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'user_topic_memories'::regclass
                          AND conname =
                            'user_topic_memories_status_check'
                    ) THEN
                        ALTER TABLE user_topic_memories
                        DROP CONSTRAINT user_topic_memories_status_check;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'user_topic_memories'::regclass
                          AND conname =
                            'user_topic_memories_status_check_b6'
                    ) THEN
                        ALTER TABLE user_topic_memories
                        ADD CONSTRAINT
                            user_topic_memories_status_check_b6
                        CHECK (
                            status IN (
                                'candidate',
                                'approved',
                                'rejected',
                                'superseded',
                                'deleted',
                                'quarantined'
                            )
                        );
                    END IF;

                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'user_topic_memory_events'::regclass
                          AND conname =
                            'user_topic_memory_events_action_check'
                    ) THEN
                        ALTER TABLE user_topic_memory_events
                        DROP CONSTRAINT
                            user_topic_memory_events_action_check;
                    END IF;

                    IF EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'user_topic_memory_events'::regclass
                          AND conname =
                            'user_topic_memory_events_action_check_v2'
                    ) THEN
                        ALTER TABLE user_topic_memory_events
                        DROP CONSTRAINT
                            user_topic_memory_events_action_check_v2;
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid =
                            'user_topic_memory_events'::regclass
                          AND conname =
                            'user_topic_memory_events_action_check_b6'
                    ) THEN
                        ALTER TABLE user_topic_memory_events
                        ADD CONSTRAINT
                            user_topic_memory_events_action_check_b6
                        CHECK (
                            action IN (
                                'created',
                                'candidate_created',
                                'auto_approved',
                                'candidate_approved',
                                'candidate_rejected',
                                'candidates_consolidated',
                                'merged',
                                'corrected',
                                'deleted',
                                'restored',
                                'repair_applied',
                                'repair_rolled_back'
                            )
                        );
                    END IF;
                END
                $migration$;
                """
            )

            # Before/after snapshots enable guarded repair rollback.
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS
                    user_topic_memory_repair_snapshots (
                    id BIGSERIAL PRIMARY KEY,
                    repair_id TEXT NOT NULL,
                    memory_id BIGINT NOT NULL
                        REFERENCES user_topic_memories(id)
                        ON DELETE RESTRICT,

                    actions JSONB NOT NULL DEFAULT '[]'::jsonb,

                    before_status TEXT NOT NULL,
                    before_active BOOLEAN NOT NULL,
                    before_embedding vector({dimensions}),
                    before_embedding_model TEXT,
                    before_embedding_text_hash TEXT,
                    before_content_hash TEXT NOT NULL,
                    before_version INTEGER NOT NULL,

                    after_status TEXT,
                    after_active BOOLEAN,
                    after_embedding_model TEXT,
                    after_embedding_text_hash TEXT,
                    after_content_hash TEXT,
                    after_version INTEGER,

                    status TEXT NOT NULL DEFAULT 'applied'
                        CHECK (
                            status IN (
                                'applied',
                                'rolled_back',
                                'rollback_conflict'
                            )
                        ),

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    rolled_back_at TIMESTAMPTZ,

                    UNIQUE(repair_id, memory_id)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_topic_memory_repairs
                ON user_topic_memory_repair_snapshots(
                    repair_id,
                    status
                )
                """
            )

           
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_topic_memory_candidate_review
                ON user_topic_memories (
                    importance DESC,
                    confidence DESC,
                    created_at ASC
                )
                WHERE status = 'candidate'
                  AND active = TRUE
                """
            )
            
            # Persistent conversation-history foundation.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id BIGSERIAL PRIMARY KEY,
                    session_key TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id BIGINT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (
                        role IN ('system', 'user', 'assistant', 'tool')
                    ),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    UNIQUE (conversation_id, message_index)
                )
                """
            )

            # One-time migration markers.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS data_migrations (
                    migration_key TEXT PRIMARY KEY,
                    details TEXT NOT NULL DEFAULT '',
                    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_rating_created
                ON feedback (rating, created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_lessons_approved
                ON feedback_lessons (approved, category)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document
                ON document_chunks (document_id, chunk_index)
                """
            )
            vector_index_tables = (
                ("behavior_guides", "idx_behavior_guides_embedding"),
                ("feedback_lessons", "idx_feedback_lessons_embedding"),
                ("document_chunks", "idx_document_chunks_embedding"),
            )

            for table_name, index_prefix in vector_index_tables:
                if vector_extension_version >= (0, 5):
                    connection.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_prefix}_hnsw
                        ON {table_name}
                        USING hnsw (embedding vector_cosine_ops)
                        WHERE embedding IS NOT NULL
                    """)
                else:
                    # HNSW requires pgvector 0.5 or later.
                    connection.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_prefix}_ivfflat
                        ON {table_name}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        WHERE embedding IS NOT NULL
                    """)
            # R5: native PostgreSQL full-text search indexes.
            # The expressions must match keyword_search() exactly.

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_behavior_guides_fts
                ON behavior_guides
                USING GIN (
                    to_tsvector(
                        'simple',
                        COALESCE(title, '') || ' ' ||
                        COALESCE(keywords, '') || ' ' ||
                        COALESCE(category, '') || ' ' ||
                        COALESCE(content, '')
                    )
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_lessons_fts
                ON feedback_lessons
                USING GIN (
                    to_tsvector(
                        'simple',
                        COALESCE(category, '') || ' ' ||
                        COALESCE(lesson_text, '')
                    )
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_fts
                ON document_chunks
                USING GIN (
                    to_tsvector(
                        'simple',
                        COALESCE(content, '') || ' ' ||
                        COALESCE(section, '')
                    )
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_fts
                ON documents
                USING GIN (
                    to_tsvector(
                        'simple',
                        COALESCE(file_name, '') || ' ' ||
                        COALESCE(category, '')
                    )
                )
            """)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memory_active
                ON user_memory (active, approved, memory_type)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON conversation_messages (conversation_id, message_index)
                """
            )

            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO behavior_guides
                    (guide_key, category, title, keywords, content, approved, source_type)
                    VALUES (%s, %s, %s, %s, %s, TRUE, 'built_in')
                    ON CONFLICT (guide_key) DO NOTHING
                    """,
                    [
                        (
                            guide["key"],
                            guide["category"],
                            guide["title"],
                            guide["keywords"],
                            guide["content"],
                        )
                        for guide in INITIAL_GUIDES
                    ],
                )

    def retrieve_guidance(
        self,
        user_message: str,
        limit: int = 2,
    ) -> list[str]:
        """Retrieve approved behavior guides and System A lessons."""
        query_tokens = _tokens(user_message)

        if not query_tokens or limit < 1:
            return []

        with self._connect() as connection:
            rows = connection.execute("""
                SELECT title, keywords, content
                FROM behavior_guides
                WHERE approved = TRUE
            """).fetchall()

            production_rows = []

            if getattr(
                self,
                "_system_a_promotion_schema_ready",
                False,
            ):
                production_rows = connection.execute("""
                    SELECT
                        rule_key,
                        category,
                        scope,
                        priority,
                        lesson_text
                    FROM zara_production_lessons
                    WHERE active = TRUE
                    ORDER BY priority DESC, updated_at DESC
                """).fetchall()

        ranked: list[tuple[float, str]] = []

        for row in rows:
            keyword_tokens = _tokens(row["keywords"])
            title_tokens = _tokens(row["title"])
            content_tokens = _tokens(row["content"])

            score = (
                3.0 * len(query_tokens & keyword_tokens)
                + 1.5 * len(query_tokens & title_tokens)
                + 0.15 * len(query_tokens & content_tokens)
            )

            if score > 0:
                ranked.append((
                    score,
                    f"{row['title']}: {row['content']}",
                ))

        for row in production_rows:
            rule_tokens = _tokens(
                f"{row['rule_key']} {row['category']} {row['scope']}"
            )

            lesson_tokens = _tokens(str(row["lesson_text"]))

            try:
                priority = max(
                    1,
                    min(int(row["priority"]), 100),
                )
            except (TypeError, ValueError):
                priority = 1

            score = (
                4.0 * len(query_tokens & rule_tokens)
                + 0.25 * len(query_tokens & lesson_tokens)
                + priority * 0.001
            )

            if score > 0:
                ranked.append((
                    score,
                    "Approved System A guidance "
                    f"({row['scope']}/{row['rule_key']}): "
                    f"{row['lesson_text']}",
                ))

        ranked.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            text for _, text in ranked[:limit]
        ]

    def save_feedback(
        self,
        user_message: str,
        assistant_response: str,
        rating: int,
        note: str = "",
    ) -> int:
        """Save explicit feedback for a completed answer."""
        if rating not in {-1, 1}:
            raise ValueError("rating must be 1 (good) or -1 (bad)")

        with self._connect() as connection:
            row = connection.execute(
                """
                INSERT INTO feedback
                    (user_message, assistant_response, rating, note)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_message,
                    assistant_response,
                    rating,
                    note.strip(),
                ),
            ).fetchone()

            return int(row["id"])

    def feedback_stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE rating = 1) AS good,
                    COUNT(*) FILTER (WHERE rating = -1) AS bad,
                    COUNT(*) FILTER (WHERE reviewed = TRUE) AS reviewed
                FROM feedback
                """
            ).fetchone()

        return {
            "total": int(row["total"] or 0),
            "good": int(row["good"] or 0),
            "bad": int(row["bad"] or 0),
            "reviewed": int(row["reviewed"] or 0),
        }

    def list_feedback(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, user_message, assistant_response,
                       rating, note, reviewed
                FROM feedback
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_feedback_by_id(
        self,
        feedback_id: int,
    ) -> dict[str, Any] | None:
        """Fetch one feedback record for display or pre-update verification."""
        if (        
            isinstance(feedback_id, bool)
            or not isinstance(feedback_id, int)
            or feedback_id <= 0
        ):
            raise ValueError("Feedback ID must be a positive integer.")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id, created_at, user_message, assistant_response,
                    rating, note, reviewed
                FROM feedback
                WHERE id = %s
                """,
                (feedback_id,),
            ).fetchone()

        return dict(row) if row else None

    @staticmethod
    def _upsert_feedback_lessons(
        connection: psycopg.Connection,
        feedback_id: int,
        user_message: str,
        note: str,
    ) -> None:
        categories = _feedback_categories(user_message + " " + note)
        if not categories:
            categories = {"general_quality"}

        for category in categories:
            lesson = _FEEDBACK_CATEGORY_LESSONS[category]
            connection.execute(
                """
                INSERT INTO feedback_lessons
                    (feedback_id, category, lesson_text, approved)
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT (feedback_id, category)
                DO UPDATE SET
                    lesson_text = EXCLUDED.lesson_text,
                    approved = TRUE,
                    updated_at = NOW()
                """,
                (feedback_id, category, lesson),
            )

    def review_feedback(self, feedback_id: int) -> bool:
        """Approve a bad-feedback note and create generalized lessons."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT rating, note, user_message
                FROM feedback
                WHERE id = %s
                """,
                (feedback_id,),
            ).fetchone()

            if row is None:
                raise ValueError(f"Feedback #{feedback_id} was not found.")

            if int(row["rating"]) != -1 or not str(row["note"]).strip():
                raise ValueError(
                    "Only bad feedback with a written reason can be approved."
                )

            connection.execute(
                "UPDATE feedback SET reviewed = TRUE WHERE id = %s",
                (feedback_id,),
            )

            self._upsert_feedback_lessons(
                connection,
                feedback_id,
                str(row["user_message"]),
                str(row["note"]),
            )

        return True

    def retrieve_feedback_guidance(
        self,
        user_message: str,
        limit: int = 1,
    ) -> list[str]:
        """Temporary category/keyword retrieval from approved lessons."""
        query_tokens = _tokens(user_message)
        if not query_tokens or limit < 1:
            return []

        query_categories = _feedback_categories(user_message)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    fl.category,
                    fl.lesson_text,
                    f.user_message,
                    f.note
                FROM feedback_lessons AS fl
                JOIN feedback AS f ON f.id = fl.feedback_id
                WHERE fl.approved = TRUE
                  AND f.rating = -1
                  AND f.reviewed = TRUE
                ORDER BY fl.updated_at DESC
                LIMIT 200
                """
            ).fetchall()

        ranked: list[tuple[int, str]] = []

        for row in rows:
            category = str(row["category"])
            source_tokens = _tokens(str(row["user_message"]))
            note_tokens = _tokens(str(row["note"]))

            score = (
                100 if category in query_categories else 0
            ) + (
                3 * len(query_tokens & source_tokens)
                + len(query_tokens & note_tokens)
            )

            if score > 0:
                ranked.append((score, str(row["lesson_text"])))

        ranked.sort(key=lambda item: item[0], reverse=True)

        results: list[str] = []
        for _, lesson in ranked:
            if lesson not in results:
                results.append(lesson)
            if len(results) >= limit:
                break

        return results

    def index_pending_embeddings(self, batch_size: int = 8) -> dict[str, Any]:
        """Create or refresh embeddings for approved behavior guides and lessons."""
        batch_size = max(1, min(int(batch_size), 16))
        embedding_client = self.embedding_client
        model = embedding_client.model
        model_version = embedding_client.model_version
        dimensions = embedding_client.dimensions
        format_version = embedding_client.format_version
        pending: list[dict[str, Any]] = []

        with self._connect() as connection:
            guides = connection.execute(
                """
                SELECT
                    id, category, title, keywords, content,
                    embedding_model, embedding_text_hash,
                    embedding_model_version,
                    embedding_dimensions,
                    embedding_format_version,
                    embedding IS NULL AS embedding_missing
                FROM behavior_guides
                WHERE approved = TRUE
                ORDER BY id
                """
            ).fetchall()

            lessons = connection.execute(
                """
                SELECT
                    id, category, lesson_text,
                    embedding_model, embedding_text_hash,
                    embedding_model_version,
                    embedding_dimensions,
                    embedding_format_version,
                    embedding IS NULL AS embedding_missing
                FROM feedback_lessons
                WHERE approved = TRUE
                ORDER BY id
                """
            ).fetchall()

            production_lessons = []

            if getattr(
                self,
                "_system_a_promotion_schema_ready",
                False,
            ):
                production_lessons = connection.execute("""
                    SELECT
                        id,
                        rule_key,
                        category,
                        scope,
                        priority,
                        lesson_text,
                        embedding_model,
                        embedding_text_hash,
                        embedding_model_version,
                        embedding_dimensions,
                        embedding_format_version,
                        embedding IS NULL AS embedding_missing
                    FROM zara_production_lessons
                    ORDER BY id
                """).fetchall()

        def add_if_needed(
            table: str,
            row: dict[str, Any],
            title: str,
            body: str,
        ) -> None:
            embedding_text = self.embedding_client.format_document(title, body)
            content_hash = _text_hash(embedding_text)

            needs_update = (
                row["embedding_missing"]
                or row["embedding_model"] != model
                or row["embedding_model_version"] != model_version
                or int(row["embedding_dimensions"] or 0) != dimensions
                or row["embedding_format_version"] != format_version
                or row["embedding_text_hash"] != content_hash
            )

            if needs_update:
                pending.append(
                    {
                        "table": table,
                        "id": int(row["id"]),
                        "title": title,
                        "body": body,
                        "content_hash": content_hash,
                    }
                )

        for row in guides:
            title = str(row["title"])
            body = (
                f"Category: {row['category']}\n"
                f"Keywords: {row['keywords']}\n"
                f"Guidance: {row['content']}"
            )
            add_if_needed("behavior_guides", row, title, body)

        for row in lessons:
            category = str(row["category"])
            title = f"Reviewed behavior lesson - {category}"
            body = (
                f"Category: {category}\n"
                f"Reviewed behavioral lesson: {row['lesson_text']}"
            )
            add_if_needed("feedback_lessons", row, title, body)

        for row in production_lessons:
            title = f"Zara learned rule: {row['rule_key']}"

            body = (
                f"Rule ID: {row['rule_key']}\n"
                f"Category: {row['category']}\n"
                f"Scope: {row['scope']}\n"
                f"Priority: {row['priority']}\n"
                f"Guidance: {row['lesson_text']}"
            )

            add_if_needed(
                "zara_production_lessons",
                row,
                title,
                body,
            )

        counts = {
            "behavior_guides": 0,
            "feedback_lessons": 0,
            "zara_production_lessons": 0,
        }

        # Batch requests limit HTTP overhead and avoid one request per row.
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]

            vectors = self.embedding_client.embed_documents(
                [(item["title"], item["body"]) for item in batch]
            )

            # Save each successful batch in one PostgreSQL transaction.
            with self._connect() as connection:
                for item, vector in zip(batch, vectors):
                    literal = _vector_literal(vector)

                    if item["table"] == "behavior_guides":
                        connection.execute(
                            """
                            UPDATE behavior_guides
                            SET embedding = %s::vector,
                                embedding_model = %s,
                                embedding_model_version = %s,
                                embedding_dimensions = %s,
                                embedding_format_version = %s,
                                embedding_text_hash = %s,
                                updated_at = NOW()
                            WHERE id = %s AND approved = TRUE
                            """,
                            (
                                literal,
                                model,
                                model_version,
                                dimensions,
                                format_version,
                                item["content_hash"],
                                item["id"],
                            ),
                        )
                    elif item["table"] == "feedback_lessons":
                        connection.execute(
                            """
                            UPDATE feedback_lessons
                            SET embedding = %s::vector,
                                embedding_model = %s,
                                embedding_model_version = %s,
                                embedding_dimensions = %s,
                                embedding_format_version = %s,
                                embedding_text_hash = %s,
                                updated_at = NOW()
                            WHERE id = %s AND approved = TRUE
                            """,
                            (
                                literal,
                                model,
                                model_version,
                                dimensions,
                                format_version,
                                item["content_hash"],
                                item["id"],
                            ),
                        )
                    elif item["table"] == "zara_production_lessons":
                        connection.execute("""
                            UPDATE zara_production_lessons
                            SET
                                embedding = %s::vector,
                                embedding_model = %s,
                                embedding_model_version = %s,
                                embedding_dimensions = %s,
                                embedding_format_version = %s,
                                embedding_text_hash = %s,
                                updated_at = NOW()
                            WHERE id = %s
                        """, (
                            literal,
                            model,
                            model_version,
                            dimensions,
                            format_version,
                            item["content_hash"],
                            item["id"],
                        ))
                    else:
                        raise ValueError("Unexpected embedding table.")

                    counts[item["table"]] += 1

        return {
            "embedding_model": model,
            "embedding_model_version": model_version,
            "embedding_dimensions": dimensions,
            "embedding_format_version": format_version,
            "pending_before_indexing": len(pending),
            "behavior_guides_indexed": counts["behavior_guides"],
            "feedback_lessons_indexed": counts["feedback_lessons"],
            "system_a_lessons_indexed": counts[
                "zara_production_lessons"
            ],
        }

    def get_user_profile_entries(self) -> list[dict[str, Any]]:
        """Return explicitly approved profile fields."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_key, content
                FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                  AND approved = TRUE
                  AND active = TRUE
                ORDER BY memory_key
                LIMIT 25
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _tool_memory_scope(scope: str) -> str:
        scope = str(scope or "all").strip().lower()

        if scope not in _TOOL_MANAGED_SCOPE_SQL:
            raise ValueError("Scope must be all, profile, or notes.")

        # Only fixed, application-owned SQL fragments are returned.
        return _TOOL_MANAGED_SCOPE_SQL[scope]

    def list_tool_managed_memory(
        self,
        scope: str = "all",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List only approved profile fields and Zara-created personal notes."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("Limit must be an integer.")

        limit = max(1, min(limit, 20))
        scope_sql = self._tool_memory_scope(scope)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE approved = TRUE
                  AND active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND {scope_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def search_tool_managed_memory(
        self,
        query: str,
        scope: str = "all",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search manageable personal records using parameterized SQL."""
        query = str(query or "").strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("Limit must be an integer.")

        limit = max(1, min(limit, 10))
        scope_sql = self._tool_memory_scope(scope)

        terms = list(dict.fromkeys(
            re.findall(r"[\w'-]{2,}", query, flags=re.UNICODE)
        ))[:8]

        if not terms:
            raise ValueError("Search query contains no searchable terms.")

        predicates = []
        params: list[Any] = []

        for term in terms:
            # Values are bound parameters, never SQL fragments.
            predicates.append(
                "(COALESCE(memory_key, '') ILIKE %s "
                "OR content ILIKE %s)"
            )
            wildcard_term = f"%{term}%"
            params.extend([wildcard_term, wildcard_term])

        search_sql = " OR ".join(predicates)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE approved = TRUE
                  AND active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND {scope_sql}
                  AND ({search_sql})
                ORDER BY updated_at DESC, id DESC
                LIMIT %s
                """,
                (*params, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_tool_managed_memory_by_id(
        self,
        memory_id: int,
    ) -> dict[str, Any] | None:
        """Fetch a single record only from the tool-managed scope."""
        if (
            isinstance(memory_id, bool)
            or not isinstance(memory_id, int)
            or memory_id <= 0
        ):
            raise ValueError("Record ID must be a positive integer.")

        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE id = %s
                  AND approved = TRUE
                  AND active = TRUE
                  AND (expires_at IS NULL OR expires_at > NOW())
                  AND {_TOOL_MANAGED_SCOPE_SQL["all"]}
                LIMIT 1
                """,
                (memory_id,),
            ).fetchone()

        return dict(row) if row else None

    def get_tool_managed_memory_by_key(
        self,
        memory_key: str,
    ) -> list[dict[str, Any]]:
        """Return up to two profile matches so duplicate keys are detectable."""
        key = str(memory_key or "").strip().casefold()

        if not key:
            raise ValueError("Profile key cannot be empty.")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                  AND memory_key = %s
                  AND approved = TRUE
                  AND active = TRUE
                ORDER BY updated_at DESC, id DESC
                LIMIT 2
                """,
                (key,),
            ).fetchall()

        return [dict(row) for row in rows]

    def create_tool_profile_entry_if_missing(
        self,
        memory_key: str,
        content: str,
    ) -> dict[str, Any]:
        """Insert a new profile field without silently overwriting an old one."""
        key = str(memory_key or "").strip().casefold()
        value = " ".join(str(content or "").split())

        if not re.fullmatch(r"[a-z0-9_-]{1,64}", key):
            raise ValueError("Invalid profile key.")
        if not value or len(value) > 600:
            raise ValueError("Profile content must contain 1–600 characters.")

        with self._connect() as connection:
            # Serializes concurrent creation attempts for the same profile key.
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                (f"zara-profile:{key}",),
            )

            existing = connection.execute(
                """
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                  AND memory_key = %s
                  AND approved = TRUE
                  AND active = TRUE
                ORDER BY updated_at DESC, id DESC
                LIMIT 2
                """,
                (key,),
            ).fetchall()

            if existing:
                return {
                    "status": "exists",
                    "records": [dict(row) for row in existing],
                }

            row = connection.execute(
                """
                INSERT INTO user_memory (
                    memory_key, memory_type, content, source,
                    approved, active, created_at, updated_at, expires_at
                )
                VALUES (
                    %s, 'profile', %s, 'user_profile_cli',
                    TRUE, TRUE, NOW(), NOW(), NULL
                )
                RETURNING
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                """,
                (key, value),
            ).fetchone()

        return {"status": "created", "record": dict(row)}

    def insert_tool_memory_note(
        self,
        content: str,
        label: str = "",
    ) -> dict[str, Any]:
        """Insert a new note; never upsert or overwrite an existing record."""
        value = str(content or "").strip()

        if not value or len(value) > 1500:
            raise ValueError("Note content must contain 1–1500 characters.")

        key = re.sub(
            r"[^a-z0-9_-]+",
            "_",
            str(label or "").casefold(),
        ).strip("_-") or None

        if key and len(key) > 64:
            raise ValueError("Note label must be at most 64 characters.")

        with self._connect() as connection:
            connection.execute(
                """
                SELECT pg_advisory_xact_lock(
                    hashtext(%s)::bigint
                )
                """,
                (f"zara-tool-memory-note:{value}",),
            )
                        
            existing = connection.execute(
                """
                SELECT
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                FROM user_memory
                WHERE memory_type = 'note'
                  AND source = 'zara_memory_tool'
                  AND approved = TRUE
                  AND active = TRUE
                  AND content = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (value,),
            ).fetchone()

            if existing:
                return {
                    "status": "already_exists",
                    "record": dict(existing),
                }

            row = connection.execute(
                """
                INSERT INTO user_memory (
                    memory_key, memory_type, content, source,
                    approved, active, created_at, updated_at, expires_at
                )
                VALUES (
                    %s, 'note', %s, 'zara_memory_tool',
                    TRUE, TRUE, NOW(), NOW(), NULL
                )
                RETURNING
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                """,
                (key, value),
            ).fetchone()

        return {"status": "created", "record": dict(row)}

    def update_tool_managed_memory_by_id(
        self,
        memory_id: int,
        expected_content: str,
        new_content: str,
    ) -> dict[str, Any] | None:
        """Update only if the previewed value still matches the stored value."""
        if (
            isinstance(memory_id, bool)
            or not isinstance(memory_id, int)
            or memory_id <= 0
        ):
            raise ValueError("Record ID must be a positive integer.")

        old_value = str(expected_content or "")
        new_value = str(new_content or "").strip()

        if not new_value or len(new_value) > 1500:
            raise ValueError("New content must contain 1–1500 characters.")

        current = self.get_tool_managed_memory_by_id(memory_id)

        if current is None or str(current.get("content") or "") != old_value:
            return None

        max_length = (
            600 if current.get("memory_type") == "profile" else 1500
        )

        if len(new_value) > max_length:
            raise ValueError(
                f"Content exceeds the {max_length}-character limit "
                "for this record type."
            )       

        with self._connect() as connection:
            row = connection.execute(
                f"""
                UPDATE user_memory
                SET content = %s,
                    updated_at = NOW(),
                    expires_at = CASE
                        WHEN memory_type = 'profile' THEN NULL
                        ELSE expires_at
                    END
                WHERE id = %s
                  AND content = %s
                  AND approved = TRUE
                  AND active = TRUE
                  AND {_TOOL_MANAGED_SCOPE_SQL["all"]}
                RETURNING
                    id, memory_key, memory_type, content, source,
                    created_at, updated_at
                """,
                (new_value, memory_id, old_value),
            ).fetchone()

        return dict(row) if row else None

    def delete_tool_managed_memory_by_id(
        self,
        memory_id: int,
        expected_content: str,
    ) -> bool:
        """Delete only the exact record version shown in the confirmation."""
        if (
            isinstance(memory_id, bool)
            or not isinstance(memory_id, int)
            or memory_id <= 0
        ):
            raise ValueError("Record ID must be a positive integer.")

        with self._connect() as connection:
            result = connection.execute(
                f"""
                DELETE FROM user_memory
                WHERE id = %s
                  AND content = %s
                  AND approved = TRUE
                  AND active = TRUE
                  AND {_TOOL_MANAGED_SCOPE_SQL["all"]}
                """,
                (memory_id, str(expected_content or "")),
            )

        return result.rowcount == 1

    def save_user_profile_entry(
        self, memory_key: str, content: str
    ) -> None:
        """Save or update a profile field explicitly supplied by the user."""
        key = str(memory_key or "").strip().casefold()
        value = " ".join(str(content or "").split())

        if (
            not key
            or len(key) > 64
            or not all(c.isalnum() or c in "_-" for c in key)
        ):
            raise ValueError("Key must use letters, numbers, _ or - (max 64).")
        if not value:
            raise ValueError("Profile value cannot be empty.")
        if len(value) > 600:
            raise ValueError("Profile value is limited to 600 characters.")

        with self._connect() as connection:
            result = connection.execute(
                """
                UPDATE user_memory
                SET content = %s, approved = TRUE, active = TRUE,
                    updated_at = NOW(), expires_at = NULL
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                  AND memory_key = %s
                """,
                (value, key),
            )
            if result.rowcount == 0:
                connection.execute(
                    """
                    INSERT INTO user_memory (
                        memory_key, memory_type, content, source,
                        approved, active, created_at, updated_at
                    )
                    VALUES (%s, 'profile', %s, 'user_profile_cli',
                            TRUE, TRUE, NOW(), NOW())
                    """,
                    (key, value),
                )

    def delete_user_profile_entry(self, memory_key: str) -> bool:
        key = str(memory_key or "").strip().casefold()
        if not key:
            return False

        with self._connect() as connection:
            result = connection.execute(
                """
                DELETE FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                  AND memory_key = %s
                """,
                (key,),
            )
        return result.rowcount > 0

    def get_user_profile_clear_preview(
        self,
    ) -> list[dict[str, Any]]:
        """Return a content-free fingerprint of the exact clear scope."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, memory_key, content
                FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                ORDER BY id
                """
            ).fetchall()

        return [
            {
                "id": int(row["id"]),
                "memory_key": str(row.get("memory_key") or ""),
                "content_sha256": hashlib.sha256(
                    str(row.get("content") or "").encode("utf-8")
                ).hexdigest(),
            }
            for row in rows
        ]

    def clear_user_profile(
        self,
        expected_preview: list[dict[str, Any]] | None = None,
    ) -> int:
        """Clear the defined profile scope, optionally verifying its preview."""
        with self._connect() as connection:
            if expected_preview is not None:
                # Prevent concurrent writes while comparing and clearing.
                connection.execute(
                    "LOCK TABLE user_memory IN SHARE ROW EXCLUSIVE MODE"
                )

                rows = connection.execute(
                    """
                    SELECT id, memory_key, content
                    FROM user_memory
                    WHERE memory_type = 'profile'
                      AND source = 'user_profile_cli'
                    ORDER BY id
                    FOR UPDATE
                    """
                ).fetchall()

                current = [
                    {
                        "id": int(row["id"]),
                        "memory_key": str(row.get("memory_key") or ""),
                        "content_sha256": hashlib.sha256(
                            str(row.get("content") or "").encode("utf-8")
                        ).hexdigest(),
                    }
                    for row in rows
                ]

                expected: list[dict[str, Any]] = []

                for item in expected_preview:
                    if not isinstance(item, dict):
                        raise ValueError(
                            "Invalid profile-clear preview. Run preview again."
                        )

                    try:
                        record_id = int(item["id"])
                    except (KeyError, TypeError, ValueError):
                        raise ValueError(
                            "Invalid profile-clear preview. Run preview again."
                        ) from None

                    if isinstance(item.get("id"), bool) or record_id <= 0:
                        raise ValueError(
                            "Invalid profile-clear preview. Run preview again."
                        )

                    expected.append({
                        "id": record_id,
                        "memory_key": str(
                            item.get("memory_key") or ""
                        ),
                        "content_sha256": str(
                            item.get("content_sha256") or ""
                        ),
                    })

                if sorted(current, key=lambda item: item["id"]) != sorted(
                    expected,
                    key=lambda item: item["id"],
                ):
                    raise ValueError(
                        "Profile records changed after preview. "
                        "Nothing was cleared; run preview again."
                    )

            result = connection.execute(
                """
                DELETE FROM user_memory
                WHERE memory_type = 'profile'
                  AND source = 'user_profile_cli'
                """
            )

        return result.rowcount

    def keyword_search(
        self,
        query: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Search approved knowledge using PostgreSQL full-text search.

        Results are bounded per source so one large document collection
        cannot completely crowd out behavioral guidance.

        This is a lexical retrieval channel, independent of embeddings.
        """
        query = str(query or "").strip()[:500]

        if not query:
            return []

        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Keyword retrieval limit must be an integer."
            ) from exc

        limit = max(1, min(limit, 30))

        branches: list[str] = []

        branches.append("""
            SELECT
                'behavior_guide'::text AS source,
                bg.id::bigint AS source_id,
                NULL::bigint AS source_parent_id,
                bg.guide_key::text AS source_key,
                NULL::text AS rule_id,
                NULL::text AS scope,
                0::integer AS priority,
                bg.version::integer AS version,
                bg.category::text AS category,
                bg.title::text AS title,
                bg.content::text AS content,
                NULL::text AS source_file,
                NULL::text AS source_path,
                0::integer AS source_version,
                NULL::integer AS page_number,
                NULL::text AS section,
                NULL::integer AS chunk_index,
                COALESCE(bg.embedding_text_hash, '')::text
                    AS content_sha256,

                EXTRACT(
                    EPOCH FROM COALESCE(bg.updated_at, bg.created_at)
                )::double precision AS updated_at_epoch,

                ts_rank_cd(
                    to_tsvector(
                        'simple',
                        COALESCE(bg.title, '') || ' ' ||
                        COALESCE(bg.keywords, '') || ' ' ||
                        COALESCE(bg.category, '') || ' ' ||
                        COALESCE(bg.content, '')
                    ),
                    q.tsquery
                )::double precision AS keyword_score,

                NULL::jsonb AS metadata,
                NULL::text AS extraction_method,
                NULL::text AS extraction_status,
                NULL::jsonb AS extraction_warnings,
                NULL::jsonb AS extraction_metadata,
                NULL::text AS source_encoding,
                NULL::integer AS page_count,
                0.0::double precision AS similarity

            FROM behavior_guides AS bg
            CROSS JOIN search_query AS q

            WHERE bg.approved = TRUE
                AND numnode(q.tsquery) > 0
                AND to_tsvector(
                    'simple',
                    COALESCE(bg.title, '') || ' ' ||
                    COALESCE(bg.keywords, '') || ' ' ||
                    COALESCE(bg.category, '') || ' ' ||
                    COALESCE(bg.content, '')
                ) @@ q.tsquery

            ORDER BY
                keyword_score DESC,
                bg.updated_at DESC,
                bg.id DESC

            LIMIT %s
        """)

        branches.append("""
            SELECT
                'feedback_lesson'::text AS source,
                fl.id::bigint AS source_id,
                fl.feedback_id::bigint AS source_parent_id,
                fl.category::text AS source_key,
                NULL::text AS rule_id,
                NULL::text AS scope,
                0::integer AS priority,
                0::integer AS version,
                fl.category::text AS category,
                'Reviewed behavior lesson'::text AS title,
                fl.lesson_text::text AS content,
                NULL::text AS source_file,
                NULL::text AS source_path,
                0::integer AS source_version,
                NULL::integer AS page_number,
                NULL::text AS section,
                NULL::integer AS chunk_index,
                COALESCE(fl.embedding_text_hash, '')::text
                    AS content_sha256,

                EXTRACT(
                    EPOCH FROM COALESCE(fl.updated_at, fl.created_at)
                )::double precision AS updated_at_epoch,

                ts_rank_cd(
                    to_tsvector(
                        'simple',
                        COALESCE(fl.category, '') || ' ' ||
                        COALESCE(fl.lesson_text, '')
                    ),
                    q.tsquery
                )::double precision AS keyword_score,

                NULL::jsonb AS metadata,
                NULL::text AS extraction_method,
                NULL::text AS extraction_status,
                NULL::jsonb AS extraction_warnings,
                NULL::jsonb AS extraction_metadata,
                NULL::text AS source_encoding,
                NULL::integer AS page_count,
                0.0::double precision AS similarity

            FROM feedback_lessons AS fl
            CROSS JOIN search_query AS q

            WHERE fl.approved = TRUE
                AND numnode(q.tsquery) > 0
                AND to_tsvector(
                    'simple',
                    COALESCE(fl.category, '') || ' ' ||
                    COALESCE(fl.lesson_text, '')
                ) @@ q.tsquery

            ORDER BY
                keyword_score DESC,
                fl.updated_at DESC,
                fl.id DESC

            LIMIT %s
        """)

        if getattr(
            self,
            "_system_a_promotion_schema_ready",
            False,
        ):
            branches.append("""
                SELECT
                    'system_a_lesson'::text AS source,
                    pl.id::bigint AS source_id,
                    pl.candidate_id::bigint AS source_parent_id,
                    pl.rule_key::text AS source_key,
                    pl.rule_key::text AS rule_id,
                    pl.scope::text AS scope,
                    pl.priority::integer AS priority,
                    pl.version::integer AS version,
                    pl.category::text AS category,
                    'Approved System A lesson'::text AS title,
                    pl.lesson_text::text AS content,
                    NULL::text AS source_file,
                    NULL::text AS source_path,
                    0::integer AS source_version,
                    NULL::integer AS page_number,
                    NULL::text AS section,
                    NULL::integer AS chunk_index,
                    COALESCE(pl.embedding_text_hash, '')::text
                        AS content_sha256,

                    EXTRACT(
                        EPOCH FROM COALESCE(pl.updated_at, pl.created_at)
                    )::double precision AS updated_at_epoch,

                    ts_rank_cd(
                        to_tsvector(
                            'simple',
                            COALESCE(pl.rule_key, '') || ' ' ||
                            COALESCE(pl.category, '') || ' ' ||
                            COALESCE(pl.scope, '') || ' ' ||
                            COALESCE(pl.lesson_text, '')
                        ),
                        q.tsquery
                    )::double precision AS keyword_score,

                    NULL::jsonb AS metadata,
                    NULL::text AS extraction_method,
                    NULL::text AS extraction_status,
                    NULL::jsonb AS extraction_warnings,
                    NULL::jsonb AS extraction_metadata,
                    NULL::text AS source_encoding,
                    NULL::integer AS page_count,
                    0.0::double precision AS similarity

                FROM zara_production_lessons AS pl
                CROSS JOIN search_query AS q

                WHERE pl.active = TRUE
                    AND numnode(q.tsquery) > 0
                    AND to_tsvector(
                        'simple',
                        COALESCE(pl.rule_key, '') || ' ' ||
                        COALESCE(pl.category, '') || ' ' ||
                        COALESCE(pl.scope, '') || ' ' ||
                        COALESCE(pl.lesson_text, '')
                    ) @@ q.tsquery

                ORDER BY
                    keyword_score DESC,
                    pl.updated_at DESC,
                    pl.id DESC

                LIMIT %s
            """)

        branches.append("""
            SELECT
                'document_chunk'::text AS source,
                dc.id::bigint AS source_id,
                dc.document_id::bigint AS source_parent_id,
                d.source_key::text AS source_key,
                NULL::text AS rule_id,
                NULL::text AS scope,
                0::integer AS priority,
                0::integer AS version,
                d.category::text AS category,
                d.file_name::text AS title,
                dc.content::text AS content,
                d.file_name::text AS source_file,
                d.source_path::text AS source_path,
                d.source_version::integer AS source_version,
                dc.page_number::integer AS page_number,
                dc.section::text AS section,
                dc.chunk_index::integer AS chunk_index,
                COALESCE(dc.embedding_text_hash, '')::text
                    AS content_sha256,

                EXTRACT(
                    EPOCH FROM d.updated_at
                )::double precision AS updated_at_epoch,

                (
                    ts_rank_cd(
                        to_tsvector(
                            'simple',
                            COALESCE(dc.content, '') || ' ' ||
                            COALESCE(dc.section, '')
                        ),
                        q.tsquery
                    )
                    +
                    0.25 * ts_rank_cd(
                        to_tsvector(
                            'simple',
                            COALESCE(d.file_name, '') || ' ' ||
                            COALESCE(d.category, '')
                        ),
                        q.tsquery
                    )
                )::double precision AS keyword_score,

                dc.metadata,
                d.extraction_method,
                d.extraction_status,
                d.extraction_warnings,
                d.extraction_metadata,
                d.source_encoding,
                d.page_count,
                0.0::double precision AS similarity

            FROM document_chunks AS dc

            JOIN documents AS d
                ON d.id = dc.document_id

            CROSS JOIN search_query AS q

            WHERE d.ingestion_status = 'ready'
                AND numnode(q.tsquery) > 0
                AND (
                    to_tsvector(
                        'simple',
                        COALESCE(dc.content, '') || ' ' ||
                        COALESCE(dc.section, '')
                    ) @@ q.tsquery

                    OR

                    to_tsvector(
                        'simple',
                        COALESCE(d.file_name, '') || ' ' ||
                        COALESCE(d.category, '')
                    ) @@ q.tsquery
                )

            ORDER BY
                keyword_score DESC,
                d.updated_at DESC,
                dc.id DESC

            LIMIT %s
        """)

        sql = (
            """
            WITH search_query AS (
                SELECT websearch_to_tsquery(
                    'simple',
                    %s
                ) AS tsquery
            ),
            candidates AS (
            """
            + "\nUNION ALL\n".join(branches)
            + """
            )
            SELECT *
            FROM candidates
            ORDER BY
                keyword_score DESC,
                updated_at_epoch DESC,
                source ASC,
                source_id DESC
            LIMIT %s
            """
        )

        params = (
            [query]
            + [limit] * len(branches)
            + [limit * len(branches)]
        )

        with self._connect() as connection:
            rows = connection.execute(
                sql,
                tuple(params),
            ).fetchall()

        return [dict(row) for row in rows]
    
    def semantic_search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search approved guidance, reviewed feedback, and documents."""
        query = (query or "").strip()
        if not query:
            return []

        limit = max(1, min(int(limit), 30))
        embedding_client = self.embedding_client
        model = embedding_client.model
        model_version = embedding_client.model_version
        dimensions = embedding_client.dimensions
        format_version = embedding_client.format_version

        vector = _vector_literal(
            embedding_client.embed_query(query)
        )

        with self._connect() as connection:
            guide_rows = connection.execute(
                """
                SELECT
                    'behavior_guide' AS source,
                    bg.id AS source_id,
                    bg.guide_key AS source_key,
                    bg.source_type,
                    bg.category,
                    bg.title,
                    bg.content,
                    EXTRACT(
                        EPOCH FROM bg.updated_at
                    )::double precision AS updated_at_epoch,                    
                    1 - (bg.embedding <=> %s::vector) AS similarity
                FROM behavior_guides AS bg
                WHERE bg.approved = TRUE
                    AND bg.embedding IS NOT NULL
                    AND bg.embedding_model = %s
                    AND bg.embedding_model_version = %s
                    AND bg.embedding_dimensions = %s
                    AND bg.embedding_format_version = %s
                ORDER BY bg.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vector,
                    model,
                    model_version,
                    dimensions,
                    format_version,
                    vector,
                    limit,
                ),  
            ).fetchall()

            lesson_rows = connection.execute(
                """
                SELECT
                    'feedback_lesson' AS source,
                    fl.id AS source_id,
                    fl.feedback_id AS source_parent_id,
                    fl.category,
                    'Reviewed behavior lesson' AS title,
                    fl.lesson_text AS content,
                    EXTRACT(
                        EPOCH FROM fl.updated_at
                    )::double precision AS updated_at_epoch,
                    1 - (fl.embedding <=> %s::vector) AS similarity
                FROM feedback_lessons AS fl
                WHERE fl.approved = TRUE
                    AND fl.embedding IS NOT NULL
                    AND fl.embedding_model = %s
                    AND fl.embedding_model_version = %s
                    AND fl.embedding_dimensions = %s
                    AND fl.embedding_format_version = %s
                ORDER BY fl.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vector,
                    model,
                    model_version,
                    dimensions,
                    format_version,
                    vector,
                    limit,
                ),
            ).fetchall()

            system_a_rows = []

            if getattr(
                self,
                "_system_a_promotion_schema_ready",
                False,
            ):
                system_a_rows = connection.execute("""
                    SELECT
                        'system_a_lesson' AS source,

                        pl.id AS source_id,
                        pl.candidate_id AS source_parent_id,

                        pl.rule_key AS source_key,
                        pl.rule_key AS rule_id,

                        pl.scope,
                        pl.priority,
                        pl.version,
                        pl.category,

                        'Approved System A lesson' AS title,
                        pl.lesson_text AS content,

                        EXTRACT(
                            EPOCH FROM pl.updated_at
                        )::double precision AS updated_at_epoch,

                        1 - (
                            pl.embedding <=> %s::vector
                        ) AS similarity

                    FROM zara_production_lessons AS pl

                    WHERE pl.active = TRUE
                        AND pl.embedding IS NOT NULL
                        AND pl.embedding_model = %s
                        AND pl.embedding_model_version = %s
                        AND pl.embedding_dimensions = %s
                        AND pl.embedding_format_version = %s
                    ORDER BY pl.embedding <=> %s::vector
                    LIMIT %s
                """, 
                    (
                        vector,
                        model,
                        model_version,
                        dimensions,
                        format_version,
                        vector,
                        limit,
                    ),
                ).fetchall()

            document_rows = connection.execute(
                """
                SELECT
                    'document_chunk' AS source,
                    dc.id AS source_id,
                    dc.document_id AS source_parent_id,
                    dc.chunk_index,

                    d.category,
                    d.file_name AS title,
                    dc.content,
                    d.file_name AS source_file,
                    d.source_path,
                    d.source_key,
                    d.source_version,
                    d.supersedes_document_id,
                    EXTRACT(
                        EPOCH FROM d.updated_at
                    )::double precision AS updated_at_epoch,
                    d.content_sha256,
                    d.extracted_text_sha256,
                    d.extraction_method,
                    d.extraction_status,
                    d.extraction_warnings,
                    d.extraction_metadata,
                    d.source_encoding,
                    d.page_count,
                    dc.page_number,
                    dc.section,
                    dc.metadata,

                    1 - (
                        dc.embedding <=> %s::vector
                    ) AS similarity

                FROM document_chunks AS dc

                JOIN documents AS d
                    ON d.id = dc.document_id

                WHERE d.ingestion_status = 'ready'
                    AND dc.embedding IS NOT NULL
                    AND dc.embedding_model = %s
                    AND dc.embedding_model_version = %s
                    AND dc.embedding_dimensions = %s
                    AND dc.embedding_format_version = %s

                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vector,
                    model,
                    model_version,
                    dimensions,
                    format_version,
                    vector,
                    limit,
                ),
            ).fetchall()

        rows = [
            dict(row)
            for row in (
                *guide_rows,
                *lesson_rows,
                *system_a_rows,
                *document_rows,
            )
        ]
        valid_rows: list[dict[str, Any]] = []

        for row in rows:
            try:
                score = float(row["similarity"])
            except (KeyError, TypeError, ValueError):
                continue

            if not math.isfinite(score) or not -1.0001 <= score <= 1.0001:
                continue

            row["similarity"] = max(-1.0, min(1.0, score))
            valid_rows.append(row)

        valid_rows.sort(
            key=lambda row: (
                float(row["similarity"]),
                int(row.get("priority") or 0),
                str(row.get("source", "")),
                str(row.get("source_id", "")),
            ),
            reverse=True,
        )

        unique_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()

        for row in valid_rows:
            content = " ".join(
                str(row.get("content", "")).split()
            ).casefold()

            if not content:
                continue

            source = str(row.get("source", ""))

            if source == "document_chunk":
                key = (
                    source,
                    str(
                        row.get("source_key")
                        or row.get("source_path")
                        or row.get("source_file")
                        or ""
                    ).casefold(),
                    str(row.get("page_number") or ""),
                    str(row.get("section") or "").casefold(),
                    content,
                )
            else:
                key = (source, content)

            if key in seen:
                continue

            seen.add(key)
            unique_rows.append(row)

            if len(unique_rows) >= limit:
                break

        return unique_rows

    def get_document_chunks_by_source_pages(
        self,
        file_name: str,
        page_numbers: list[int],
    ) -> list[dict[str, Any]]:
        """Retrieve exact source pages without guessing ambiguous filenames."""
        reference = (
            str(file_name or "")
            .replace("\\", "/")
            .strip()
        )
        has_path = "/" in reference
        basename = reference.rsplit("/", 1)[-1].strip()

        pages: list[int] = []

        for value in page_numbers:
            if isinstance(value, bool):
                continue

            try:
                page = int(value)
            except (TypeError, ValueError):
                continue

            if page > 0 and page not in pages:
                pages.append(page)

        if not basename or not pages:
            return []

        with self._connect() as connection:
            if has_path:
                source_predicate = (
                    "LOWER(REPLACE(d.source_path, chr(92), '/')) "
                    "= LOWER(%s)"
                )
                source_value = reference.rstrip("/")
            else:
                ambiguity = connection.execute(
                    """
                    SELECT COUNT(
                        DISTINCT CASE
                            WHEN COALESCE(source_path, '') <> ''
                            THEN LOWER(
                                REPLACE(source_path, chr(92), '/')
                            )
                            ELSE '__empty_path__:' || id::text
                        END
                    ) AS source_count
                    FROM documents
                    WHERE ingestion_status = 'ready'
                      AND LOWER(file_name) = LOWER(%s)
                    """,
                    (basename,),
                ).fetchone()

                if int(ambiguity["source_count"] or 0) > 1:
                    # Same filename in different locations is ambiguous.
                    return []

                source_predicate = "LOWER(d.file_name) = LOWER(%s)"
                source_value = basename

            rows = connection.execute(
                f"""
                WITH eligible_documents AS (
                    SELECT
                        d.id,
                        d.file_name,
                        d.category,
                        d.source_path,
                        d.source_key,
                        d.source_version,
                        d.content_sha256,
                        d.extracted_text_sha256,
                        d.extraction_method,
                        d.extraction_status,
                        d.extraction_warnings,
                        d.extraction_metadata,
                        d.source_encoding,
                        d.page_count,
                        d.updated_at
                    FROM documents AS d
                    WHERE d.ingestion_status = 'ready'
                      AND {source_predicate}
                )
                SELECT
                    'document_chunk' AS source,
                    dc.id AS source_id,
                    dc.document_id AS source_parent_id,
                    dc.chunk_index,
                    d.category,
                    d.file_name AS title,
                    dc.content,
                    d.file_name AS source_file,
                    d.source_path,
                    d.source_key,
                    d.source_version,
                    d.content_sha256,
                    d.extracted_text_sha256,
                    d.extraction_method,
                    d.extraction_status,
                    d.extraction_warnings,
                    d.extraction_metadata,
                    d.source_encoding,
                    d.page_count,
                    dc.page_number,
                    dc.section,
                    dc.metadata,
                    0.0::double precision AS similarity
                FROM document_chunks AS dc
                JOIN eligible_documents AS d
                  ON d.id = dc.document_id
                WHERE dc.page_number = ANY(%s)
                ORDER BY
                    array_position(%s::integer[], dc.page_number),
                    dc.chunk_index
                """,
                (source_value, pages, pages),
            ).fetchall()

        return [dict(row) for row in rows]