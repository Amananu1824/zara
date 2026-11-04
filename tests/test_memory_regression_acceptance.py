from __future__ import annotations

import time
import unittest

from datetime import datetime, timezone
from threading import Event, RLock
from unittest.mock import MagicMock

from app.ai.chat_engine import ChatEngine
from app.ai.tools.database import (
    DatabaseToolService,
    _PendingMutation,
)
from app.learning_ai.learning_store import (
    DatabaseOperationOutcomeUnknown,
)
from app.api.chat import ChatResponse
from app.memory_management.manager import (
    MemoryManager,
    _sha256,
)


# ============================================================
# TEST FIXTURES — NO REAL DATABASE CONNECTION
# ============================================================

class FakeCursor:
    def __init__(self, *, row=None, rows=None):
        self.row = row
        self.rows = list(rows or [])

    def fetchone(self):
        return self.row

    def fetchall(self):
        return list(self.rows)


class FakeConnection:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.calls.append((query, params))

        if self.responses:
            return self.responses.pop(0)

        return FakeCursor()


class FakeStore:
    def __init__(self, connection):
        self.connection = connection
        self.embedding_client = MagicMock()
        self.embedding_client.model = "test-embedding"
        self.embedding_client.embed_documents.return_value = [
            [0.1, 0.2, 0.3]
        ]

    def _connect(self):
        return self.connection


def valid_candidate(
    *,
    summary="The user is learning Python for a long-term project.",
    quote="I am learning Python for my long-term project.",
    confidence=0.9,
    importance=3,
    sensitivity="normal",
):
    return {
        "topic_key": "learning.python",
        "topic": "Python learning",
        "memory_type": "learning",
        "summary": summary,
        "evidence_quote": quote,
        "importance": importance,
        "confidence": confidence,
        "sensitivity": sensitivity,
    }


def memory_row(
    *,
    memory_id=1,
    status="approved",
    active=True,
    content_hash=None,
    summary="The user is developing the Zara V2 assistant.",
):
    topic_key = "project.zara"
    topic = "Zara project"
    memory_type = "project"

    return {
        "id": memory_id,
        "topic_key": topic_key,
        "topic": topic,
        "memory_type": memory_type,
        "summary": summary,
        "status": status,
        "source_kind": "user_requested",
        "sensitivity": "normal",
        "content_hash": content_hash or _sha256(
            f"{topic_key}|{memory_type}|{summary}"
        ),
        "version": memory_id,
        "supersedes_id": None,
        "evidence_count": 1,
        "active": active,
        "importance": 4,
        "confidence": 0.95,
        "provenance": [],
        "has_embedding": True,
        "embedding_model": "test-embedding",
        "embedding_text_hash": _sha256(
            f"title: {topic} | text: {summary}"
        ),
        "parent_id": None,
        "parent_topic_key": None,
        "parent_version": None,
    }


# ============================================================
# B2 — EXTRACTION, VALIDATION AND IMPORTANCE
# ============================================================

class MemoryCandidateValidationAcceptanceTests(unittest.TestCase):

    def test_candidate_requires_user_authored_evidence(self):
        item = valid_candidate(
            quote="A sentence that was never supplied by the user."
        )

        result = MemoryManager._validate_candidate(
            item,
            ["I am learning Python for my long-term project."],
            explicit=True,
        )

        self.assertIsNone(result)

    def test_explicit_save_has_importance_floor(self):
        item = valid_candidate(
            importance=1,
            confidence=0.7,
        )

        result = MemoryManager._validate_candidate(
            item,
            [item["evidence_quote"]],
            explicit=True,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["importance"], 4)

    def test_automatic_candidate_requires_higher_confidence(self):
        item = valid_candidate(confidence=0.79)

        result = MemoryManager._validate_candidate(
            item,
            [item["evidence_quote"]],
            explicit=False,
        )

        self.assertIsNone(result)

    def test_duplicate_candidates_merge_conservatively(self):
        first = valid_candidate(
            summary="The user is learning Python for a long-term project.",
            quote="I am learning Python for my long-term project.",
            confidence=0.96,
            importance=3,
            sensitivity="normal",
        )

        second = valid_candidate(
            summary="The user is building a Python application.",
            quote="I am building a Python application.",
            confidence=0.82,
            importance=5,
            sensitivity="sensitive",
        )

        result = MemoryManager._validated_unique_candidates(
            [first, second],
            [
                first["evidence_quote"],
                second["evidence_quote"],
            ],
            explicit=False,
            limit=5,
        )

        self.assertEqual(len(result), 1)
        self.assertIn("learning Python", result[0]["summary"])
        self.assertIn(
            "building a Python application",
            result[0]["summary"],
        )
        self.assertEqual(result[0]["importance"], 5)
        self.assertEqual(result[0]["confidence"], 0.82)
        self.assertEqual(result[0]["sensitivity"], "sensitive")

    def test_temporary_signal_reduces_automatic_importance(self):
        durable = valid_candidate(
            quote="I am learning Python for my long-term project."
        )

        temporary = valid_candidate(
            quote="Hypothetically, this is only for today."
        )

        durable_candidate = MemoryManager._validate_candidate(
            durable,
            [durable["evidence_quote"]],
            explicit=False,
        )

        temporary_candidate = MemoryManager._validate_candidate(
            temporary,
            [temporary["evidence_quote"]],
            explicit=False,
        )

        self.assertIsNotNone(durable_candidate)
        self.assertIsNotNone(temporary_candidate)

        durable_score = MemoryManager._score_automatic_candidate(
            durable_candidate,
            durable,
            [durable["evidence_quote"]],
        )

        temporary_score = MemoryManager._score_automatic_candidate(
            temporary_candidate,
            temporary,
            [temporary["evidence_quote"]],
        )

        self.assertLess(temporary_score[0], durable_score[0])

        self.assertIn(
            "temporary_or_hypothetical_signal",
            temporary_score[2],
        )


class AutomaticMemoryPersistenceAcceptanceTests(unittest.TestCase):

    def test_automatic_extraction_creates_candidate_only(self):
        manager = MemoryManager(FakeStore(FakeConnection()))
        manager._persist_memory = MagicMock(
            return_value={"id": 11}
        )

        item = valid_candidate()

        processed = manager.save_automatic_candidates(
            [item],
            [item["evidence_quote"]],
            trace_id="automatic-trace",
        )

        self.assertEqual(processed, 1)

        call = manager._persist_memory.call_args

        self.assertEqual(
            call.kwargs["source_kind"],
            "automatic_extraction",
        )

        self.assertEqual(
            call.kwargs["status"],
            "candidate",
        )

    def test_explicit_save_is_not_processed_by_automatic_pipeline(self):
        manager = MemoryManager(FakeStore(FakeConnection()))
        manager._persist_memory = MagicMock()

        item = valid_candidate()

        result = manager.save_automatic_candidates(
            [item],
            [
                "Please remember this: "
                "I am learning Python for my project."
            ],
            trace_id="explicit-save-trace",
        )

        self.assertEqual(result, 0)
        manager._persist_memory.assert_not_called()


# ============================================================
# B1 — VERSION HISTORY, AUDIT AND LIFECYCLE
# ============================================================

class MemoryLifecycleAcceptanceTests(unittest.TestCase):

    def test_history_returns_all_versions_in_correct_order(self):
        versions = [
            {"id": 3, "version": 3, "status": "approved"},
            {"id": 2, "version": 2, "status": "superseded"},
            {"id": 1, "version": 1, "status": "superseded"},
        ]

        connection = FakeConnection([
            FakeCursor(row={"topic_key": "project.zara"}),
            FakeCursor(rows=versions),
        ])

        result = MemoryManager(
            FakeStore(connection)
        ).get_memory_history(3)

        self.assertEqual(
            [row["version"] for row in result],
            [3, 2, 1],
        )

    def test_audit_read_excludes_raw_conversation_text(self):
        connection = FakeConnection([
            FakeCursor(row={"topic_key": "project.zara"}),
            FakeCursor(rows=[{"action": "corrected", "details": {}}]),
        ])

        result = MemoryManager(
            FakeStore(connection)
        ).get_memory_audit(1)

        self.assertEqual(len(result), 1)

        audit_sql = connection.calls[1][0].casefold()

        self.assertNotIn("user_message", audit_sql)
        self.assertNotIn("assistant_response", audit_sql)

    def test_new_version_supersedes_old_version_transactionally(self):
        current = {"id": 5}
        next_version = {"version": 3}
        inserted = {
            "id": 6,
            "status": "approved",
            "version": 3,
        }

        connection = FakeConnection([
            FakeCursor(),
            FakeCursor(row=current),
            FakeCursor(row=None),
            FakeCursor(row=current),
            FakeCursor(row=next_version),
            FakeCursor(),
            FakeCursor(row=inserted),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        candidate = {
            "topic_key": "project.zara",
            "topic": "Zara project",
            "memory_type": "project",
            "summary": (
                "The corrected Zara project memory "
                "is sufficiently detailed."
            ),
            "importance": 4,
            "confidence": 0.95,
            "sensitivity": "normal",
        }

        result = manager._persist_memory(
            candidate,
            source_kind="user_corrected",
            status="approved",
            provenance_event={
                "origin": "user_memory_correction",
                "source_trace_id": "version-acceptance",
                "evidence_hash": "evidence-hash",
            },
            expected_current_id=5,
        )

        self.assertEqual(result["id"], 6)
        self.assertEqual(result["version"], 3)

        supersede_sql = next(
            query
            for query, _ in connection.calls
            if "SET status = 'superseded'" in query
        )

        self.assertIn("active = FALSE", supersede_sql)

        insert_call = next(
            params
            for query, params in connection.calls
            if "INSERT INTO user_topic_memories" in query
        )

        self.assertEqual(insert_call[11], 3)
        self.assertEqual(insert_call[12], 5)

        manager._write_audit_event.assert_called_once()

    def test_correction_refuses_non_current_candidate(self):
        row = memory_row(
            status="candidate",
            active=True,
        )

        connection = FakeConnection([
            FakeCursor(row=row)
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._persist_memory = MagicMock()

        result = manager.correct_memory(
            1,
            "The corrected Zara project memory is sufficiently long.",
        )

        self.assertEqual(result["status"], "stale_target")
        manager._persist_memory.assert_not_called()

    def test_delete_is_reversible_soft_delete(self):
        target = {
            "topic_key": "project.zara",
            "status": "approved",
            "active": True,
        }

        current = {
            "id": 1,
            "status": "approved",
            "active": True,
        }

        connection = FakeConnection([
            FakeCursor(row=target),
            FakeCursor(),
            FakeCursor(row=current),
            FakeCursor(),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.delete_memory(1)

        self.assertEqual(result["status"], "deleted")
        self.assertTrue(result["recoverable"])

        update_sql = next(
            query
            for query, _ in connection.calls
            if "SET status = 'deleted'" in query
        )

        self.assertIn("active = FALSE", update_sql)

        self.assertFalse(any(
            query.lstrip().upper().startswith(
                "DELETE FROM USER_TOPIC_MEMORIES"
            )
            for query, _ in connection.calls
        ))

        manager._write_audit_event.assert_called_once()

    def test_delete_refuses_non_current_candidate(self):
        target = {
            "topic_key": "project.zara",
            "status": "candidate",
            "active": True,
        }

        current = {
            "id": 1,
            "status": "candidate",
            "active": True,
        }

        connection = FakeConnection([
            FakeCursor(row=target),
            FakeCursor(),
            FakeCursor(row=current),
        ])

        result = MemoryManager(
            FakeStore(connection)
        ).delete_memory(1)

        self.assertEqual(result["status"], "stale_target")

        self.assertFalse(any(
            "SET status = 'deleted'" in query
            for query, _ in connection.calls
        ))


# ============================================================
# B3 — APPROVAL, REJECTION AND CONSOLIDATION
# ============================================================

class CandidateLifecycleAcceptanceTests(unittest.TestCase):

    def test_approval_does_not_overwrite_different_current_memory(self):
        candidate = {
            "id": 7,
            "topic_key": "project.zara",
            "summary": "Candidate summary",
            "content_hash": "candidate-hash",
            "status": "candidate",
            "source_kind": "automatic_extraction",
            "provenance": [],
            "importance": 3,
            "confidence": 0.9,
            "sensitivity": "normal",
            "evidence_count": 1,
            "version": 2,
            "active": True,
            "expired": False,
        }

        current = {
            "id": 5,
            "content_hash": "different-hash",
        }

        connection = FakeConnection([
            FakeCursor(row={"topic_key": "project.zara"}),
            FakeCursor(),
            FakeCursor(row=candidate),
            FakeCursor(row=current),
        ])

        result = MemoryManager(
            FakeStore(connection)
        ).approve_memory_candidate(7)

        self.assertEqual(result["status"], "conflict")

        self.assertFalse(any(
            query.lstrip().upper().startswith(
                "UPDATE USER_TOPIC_MEMORIES"
            )
            for query, _ in connection.calls
        ))

    def test_rejection_rejects_unknown_reason(self):
        connection = FakeConnection()
        manager = MemoryManager(FakeStore(connection))

        with self.assertRaises(ValueError):
            manager.reject_memory_candidate(
                7,
                reason_code="invented_reason",
            )

        self.assertEqual(connection.calls, [])

    def test_rejection_changes_state_without_physical_deletion(self):
        candidate = {
            "id": 7,
            "status": "candidate",
            "active": True,
        }

        updated = {"id": 7}

        connection = FakeConnection([
            FakeCursor(row={"topic_key": "project.zara"}),
            FakeCursor(),
            FakeCursor(row=candidate),
            FakeCursor(row=updated),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.reject_memory_candidate(
            7,
            reason_code="temporary",
            trace_id="rejection-acceptance",
        )

        self.assertEqual(result["status"], "rejected")

        update_sql = next(
            query
            for query, _ in connection.calls
            if "SET status = 'rejected'" in query
        )

        self.assertIn("active = FALSE", update_sql)
        manager._write_audit_event.assert_called_once()

    def test_consolidation_preview_preserves_conflicts_and_is_read_only(self):
        duplicate_groups = [{
            "topic_key": "project.zara",
            "duplicate_count": 2,
            "memory_ids": [7, 8],
        }]

        conflicts = [{
            "topic_key": "project.zara",
            "distinct_versions": 2,
            "memory_ids": [7, 9],
        }]

        connection = FakeConnection([
            FakeCursor(rows=duplicate_groups),
            FakeCursor(rows=conflicts),
        ])

        result = MemoryManager(
            FakeStore(connection)
        ).preview_candidate_consolidation()

        self.assertEqual(len(result["exact_duplicate_groups"]), 1)
        self.assertEqual(len(result["conflicting_topics"]), 1)
        self.assertFalse(result["automatic_content_rewrite"])

        self.assertTrue(all(
            query.lstrip().upper().startswith("SELECT")
            for query, _ in connection.calls
        ))

    def test_exact_duplicate_consolidation_keeps_best_record(self):
        rows = [
            {
                "id": 7,
                "topic_key": "project.zara",
                "content_hash": "same-hash",
                "provenance": [],
                "importance": 3,
                "confidence": 0.8,
                "evidence_count": 1,
                "version": 1,
            },
            {
                "id": 8,
                "topic_key": "project.zara",
                "content_hash": "same-hash",
                "provenance": [],
                "importance": 5,
                "confidence": 0.9,
                "evidence_count": 2,
                "version": 2,
            },
        ]

        connection = FakeConnection([
            FakeCursor(rows=[{"topic_key": "project.zara"}]),
            FakeCursor(),
            FakeCursor(rows=rows),
            FakeCursor(),
            FakeCursor(),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.consolidate_memory_candidates(
            topic_key="project.zara",
            trace_id="consolidation-acceptance",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["groups_merged"], 1)
        self.assertEqual(result["records_retired"], 1)
        self.assertEqual(result["kept_memory_ids"], [8])

        retired_sql = next(
            query
            for query, _ in connection.calls
            if "SET status = 'superseded'" in query
        )

        self.assertIn("active = FALSE", retired_sql)
        manager._write_audit_event.assert_called_once()


# ============================================================
# B5 — RETRIEVAL AND CONFLICT RESOLUTION
# ============================================================

class MemoryRetrievalAcceptanceTests(unittest.TestCase):

    def test_pending_conflict_blocks_approved_memory_from_recall(self):
        now = datetime.now(timezone.utc)

        rows = [
            {
                "id": 1,
                "topic_key": "project.zara",
                "topic": "Zara",
                "memory_type": "project",
                "summary": "Current approved Zara fact.",
                "importance": 5,
                "confidence": 0.95,
                "source_kind": "user_requested",
                "updated_at": now,
                "age_seconds": 0,
                "pending_conflict_count": 1,
            },
            {
                "id": 2,
                "topic_key": "learning.python",
                "topic": "Python",
                "memory_type": "learning",
                "summary": "Current approved Python fact.",
                "importance": 4,
                "confidence": 0.9,
                "source_kind": "user_requested",
                "updated_at": now,
                "age_seconds": 0,
                "pending_conflict_count": 0,
            },
        ]

        manager = MemoryManager(
            FakeStore(
                FakeConnection([
                    FakeCursor(rows=rows)
                ])
            )
        )

        result = manager.retrieve_recent_approved(limit=3)

        self.assertEqual([row["id"] for row in result], [2])
        self.assertEqual(
            manager.last_retrieval_diagnosis["conflict_count"],
            1,
        )


# ============================================================
# B6 — CONFIRMATION SECURITY AND RESULT CONTRACTS
# ============================================================

class ConfirmationAndResultAcceptanceTests(unittest.TestCase):

    @staticmethod
    def make_service_with_pending_repair(token):
        service = object.__new__(DatabaseToolService)

        service._lock = RLock()
        service._mutation_denial = lambda: None
        service.memory_manager = MagicMock()

        service._pending = {
            token: _PendingMutation(
                token=token,
                action="repair",
                memory_id=0,
                expected_content="",
                new_content=None,
                label="Memory repair",
                expires_at=time.monotonic() + 60,
                scope="topic_memory",
                payload={
                    "plan_fingerprint": "f" * 64,
                    "limit": 250,
                },
            )
        }

        service.memory_manager.preview_memory_repair.return_value = {
            "plan_fingerprint": "f" * 64,
        }

        service.memory_manager.apply_memory_repairs.return_value = {
            "status": "completed",
            "repair_id": "a" * 32,
            "repaired_records": 3,
            "repaired_actions": 4,
            "remaining_records": 0,
            "scan_may_be_incomplete": False,
        }

        return service

    def test_confirmation_token_is_consumed_only_once(self):
        token = "one-time-confirmation-token"

        service = self.make_service_with_pending_repair(token)

        first = service.confirm_action(token)
        second = service.confirm_action(token)

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "failed")

        service.memory_manager.apply_memory_repairs.assert_called_once()

    def test_expired_confirmation_never_executes_mutation(self):
        token = "expired-confirmation-token"

        service = self.make_service_with_pending_repair(token)
        pending = service._pending[token]

        service._pending[token] = _PendingMutation(
            token=pending.token,
            action=pending.action,
            memory_id=pending.memory_id,
            expected_content=pending.expected_content,
            new_content=pending.new_content,
            label=pending.label,
            expires_at=time.monotonic() - 1,
            scope=pending.scope,
            payload=pending.payload,
        )

        result = service.confirm_action(token)

        self.assertEqual(result["status"], "failed")

        service.memory_manager.apply_memory_repairs.assert_not_called()

    def test_pre_cancelled_confirmation_never_executes(self):
        token = "cancelled-confirmation-token"

        service = self.make_service_with_pending_repair(token)

        cancel_event = Event()
        cancel_event.set()

        result = service.confirm_action(
            token,
            cancel_event=cancel_event,
        )

        self.assertEqual(result["status"], "cancelled")

        service.memory_manager.apply_memory_repairs.assert_not_called()

    def test_control_result_redacts_token_and_filters_private_fields(self):
        engine = object.__new__(ChatEngine)

        engine.last_database_tool_trace = []

        engine._record_task_result = MagicMock(
            return_value={
                "status": "success",
                "assistant_message": "Operation recorded.",
            }
        )

        token = "sensitive-confirmation-token"

        result = {
            "status": "success",
            "system_message": f"Completed {token}",
            "data": {
                "action": "repair",
                "repair_id": "a" * 32,
                "repaired_records": 3,
                "remaining_records": 2,
                "scan_may_be_incomplete": True,
                "private_payload": "must-not-pass",
            },
        }

        ChatEngine._record_database_control_result(
            engine,
            result,
            operation="database_confirm",
            sensitive_value=token,
        )

        call = engine._record_task_result.call_args

        self.assertNotIn(token, call.args[2])

        safe_data = call.kwargs["data"]

        self.assertEqual(safe_data["repaired_records"], 3)
        self.assertEqual(safe_data["remaining_records"], 2)
        self.assertTrue(safe_data["scan_may_be_incomplete"])
        self.assertNotIn("private_payload", safe_data)


# ============================================================
# B4 — SHARED CHATENGINE AND API CONTRACT
# ============================================================

class CrossInterfaceAcceptanceTests(unittest.TestCase):

    def test_natural_confirmation_forwards_cancellation(self):
        engine = object.__new__(ChatEngine)

        engine.database_tools = MagicMock()
        engine.database_tools.confirmation_intent.return_value = "confirm"

        engine.database_tools.confirm_latest_action.return_value = {
            "status": "success",
            "system_message": "Confirmed.",
            "data": {},
        }

        engine._record_database_control_result = MagicMock(
            return_value="Confirmed."
        )

        event = Event()

        result = ChatEngine._reply_locked(
            engine,
            "confirm",
            cancel_event=event,
        )

        self.assertEqual(result, "Confirmed.")

        engine.database_tools.confirm_latest_action.assert_called_once_with(
            cancel_event=event
        )

    def test_api_response_accepts_shared_engine_result_contract(self):
        payload = {
            "assistant_message": "The actual Zara answer.",
            "system_results": [],
            "display_events": [
                {
                    "speaker": "Zara",
                    "message": "The actual Zara answer.",
                }
            ],
            "trace_id": "trace-acceptance-1",
        }

        response = ChatResponse(
            reply=payload["assistant_message"],
            session_id="acceptance-session",
            language_mode="hinglish",
            system_results=payload["system_results"],
            trace_id=payload["trace_id"],
            display_events=payload["display_events"],
        )

        self.assertEqual(
            response.reply,
            "The actual Zara answer.",
        )

        self.assertEqual(
            response.trace_id,
            "trace-acceptance-1",
        )

        self.assertEqual(len(response.display_events), 1)

class FinalMemoryAcceptanceTests(unittest.TestCase):

    def test_successful_candidate_approval_creates_approved_state(self):
        candidate = {
            "id": 7,
            "topic_key": "project.zara",
            "summary": "A verified Zara project candidate.",
            "content_hash": "candidate-hash",
            "status": "candidate",
            "source_kind": "automatic_extraction",
            "provenance": [],
            "importance": 4,
            "confidence": 0.95,
            "sensitivity": "normal",
            "evidence_count": 1,
            "version": 1,
            "active": True,
            "expired": False,
        }

        connection = FakeConnection([
            FakeCursor(row={"topic_key": "project.zara"}),
            FakeCursor(),
            FakeCursor(row=candidate),
            FakeCursor(row=None),
            FakeCursor(row={"id": 7, "version": 2}),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.approve_memory_candidate(7)

        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["memory_id"], 7)
        self.assertEqual(result["version"], 2)
        self.assertIsNone(result["previous_memory_id"])
        manager._write_audit_event.assert_called_once()

    def test_successful_restore_creates_new_approved_version(self):
        old_memory = {
            "id": 1,
            "topic_key": "project.zara",
            "topic": "Zara project",
            "memory_type": "project",
            "summary": (
                "The user is developing an earlier version "
                "of the Zara assistant."
            ),
            "importance": 4,
            "confidence": 0.95,
            "sensitivity": "normal",
            "status": "deleted",
            "active": False,
        }

        inserted = {
            "id": 2,
            "status": "approved",
            "version": 2,
        }

        connection = FakeConnection([
            FakeCursor(row=old_memory),
            FakeCursor(row=None),
            FakeCursor(),
            FakeCursor(row=None),
            FakeCursor(row=None),
            FakeCursor(row=None),
            FakeCursor(row={"version": 2}),
            FakeCursor(row=inserted),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.restore_memory(1)

        self.assertEqual(result["status"], "restored")
        self.assertEqual(result["memory_id"], 2)
        self.assertEqual(result["restored_from_id"], 1)
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["persist_status"], "approved")
        manager._write_audit_event.assert_called_once()

    def test_repair_apply_creates_snapshot_and_audit(self):
        row = memory_row(memory_id=12)

        plan = {
            "plan_fingerprint": "f" * 64,
            "repair_actions": [{
                "memory_id": 12,
                "topic_key": row["topic_key"],
                "version": row["version"],
                "content_hash": row["content_hash"],
                "summary_hash": _sha256(row["summary"]),
                "status": "approved",
                "active": True,
                "embedding_model": row["embedding_model"],
                "embedding_text_hash": row["embedding_text_hash"],
                "actions": ["quarantine"],
            }],
            "scan_may_be_incomplete": False,
        }

        after = {
            "status": "quarantined",
            "active": False,
            "version": 1,
            "content_hash": row["content_hash"],
            "embedding_model": row["embedding_model"],
            "embedding_text_hash": row["embedding_text_hash"],
        }

        connection = FakeConnection([
            FakeCursor(),
            FakeCursor(row=row),
            FakeCursor(),
            FakeCursor(),
            FakeCursor(row=after),
            FakeCursor(),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager.preview_memory_repair = MagicMock(
            return_value=plan
        )
        manager._write_audit_event = MagicMock()

        result = manager.apply_memory_repairs(
            "f" * 64,
            limit=500,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["repaired_records"], 1)
        self.assertEqual(result["repaired_actions"], 1)
        self.assertTrue(result["rollback_available"])

        sql = "\n".join(
            query for query, _ in connection.calls
        )

        self.assertIn(
            "INSERT INTO user_topic_memory_repair_snapshots",
            sql,
        )
        self.assertIn(
            "UPDATE user_topic_memories",
            sql,
        )
        self.assertIn(
            "UPDATE user_topic_memory_repair_snapshots",
            sql,
        )

        self.assertEqual(
            manager._write_audit_event.call_args.kwargs["action"],
            "repair_applied",
        )

    def test_repair_rollback_restores_unchanged_snapshot(self):
        snapshot = {
            "id": 91,
            "memory_id": 12,
            "topic_key": "project.zara",
            "status": "applied",
            "before_status": "approved",
            "before_active": True,
            "before_embedding": "[0.1,0.2,0.3]",
            "before_embedding_model": "old-model",
            "before_embedding_text_hash": "old-hash",
            "after_status": "quarantined",
            "after_active": False,
            "after_version": 1,
            "after_content_hash": "content-hash",
            "after_embedding_model": "test-embedding",
            "after_embedding_text_hash": "text-hash",
        }

        current = {
            "status": "quarantined",
            "active": False,
            "version": 1,
            "content_hash": "content-hash",
            "embedding_model": "test-embedding",
            "embedding_text_hash": "text-hash",
        }

        connection = FakeConnection([
            FakeCursor(rows=[snapshot]),
            FakeCursor(),
            FakeCursor(rows=[snapshot]),
            FakeCursor(row=current),
            FakeCursor(row=None),
            FakeCursor(),
            FakeCursor(),
        ])

        manager = MemoryManager(FakeStore(connection))
        manager._write_audit_event = MagicMock()

        result = manager.rollback_memory_repair("a" * 32)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["rolled_back"], 1)
        self.assertEqual(result["conflicts"], 0)

        self.assertIn(
            "FOR UPDATE OF s",
            connection.calls[2][0],
        )

        memory_update = [
            query
            for query, _ in connection.calls
            if "UPDATE user_topic_memories" in query
        ]

        self.assertEqual(len(memory_update), 1)

        self.assertEqual(
            manager._write_audit_event.call_args.kwargs["action"],
            "repair_rolled_back",
        )

    def test_stale_repair_preview_never_applies_mutation(self):
        service = object.__new__(DatabaseToolService)
        manager = MagicMock()
        service.memory_manager = manager

        manager.preview_memory_repair.return_value = {
            "plan_fingerprint": "b" * 64,
        }

        pending = _PendingMutation(
            token="stale-preview-token",
            action="repair",
            memory_id=0,
            expected_content="",
            new_content=None,
            label="Memory repair",
            expires_at=time.monotonic() + 60,
            scope="topic_memory",
            payload={
                "plan_fingerprint": "f" * 64,
                "limit": 250,
            },
        )

        result = service._confirm_topic_memory_mutation(
            pending,
            started_at=time.monotonic(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            result["error_code"],
            "stale_confirmation_target",
        )

        manager.apply_memory_repairs.assert_not_called()

    def test_unknown_rollback_outcome_never_reports_success(self):
        service = object.__new__(DatabaseToolService)
        manager = MagicMock()
        service.memory_manager = manager

        repair_id = "c" * 32

        manager.preview_memory_repair_rollback.return_value = {
            "rollback_available": 1,
        }

        manager.rollback_memory_repair.side_effect = (
            DatabaseOperationOutcomeUnknown(
                "The final transaction outcome is unknown."
            )
        )

        pending = _PendingMutation(
            token="uncertain-rollback-token",
            action="repair_rollback",
            memory_id=0,
            expected_content="",
            new_content=None,
            label="Memory rollback",
            expires_at=time.monotonic() + 60,
            scope="topic_memory",
            payload={"repair_id": repair_id},
        )

        result = service._confirm_topic_memory_mutation(
            pending,
            started_at=time.monotonic(),
        )

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["data"]["verified"])
        self.assertEqual(result["data"]["outcome"], "unknown")
        self.assertEqual(
            result["error_code"],
            "database_outcome_unknown",
        )
        
if __name__ == "__main__":
    unittest.main()