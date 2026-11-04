import time
import unittest

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.ai.chat_engine import ChatEngine
from app.ai.tools.database import (
    DatabaseToolService,
    _PendingMutation,
)
from app.memory_management.manager import (
    MemoryManager,
    _parse_json_object,
    _sha256,
)


class FakeCursor:
    def __init__(self, rows=None, row=None):
        self.rows = list(rows or [])
        self.row = row

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, *responses):
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
        self.embedding_client = SimpleNamespace(
            model="test-embedding",
            embed_documents=MagicMock(
                return_value=[[0.1, 0.2, 0.3]]
            ),
        )

    def _connect(self):
        return self.connection


def make_memory_row(
    *,
    memory_id=1,
    status="approved",
    active=True,
    source_kind="user_requested",
    sensitivity="normal",
):
    topic_key = "project.zara"
    topic = "Zara Project"
    memory_type = "project"
    summary = "User is developing the Zara V2 assistant."

    return {
        "id": memory_id,
        "topic_key": topic_key,
        "topic": topic,
        "memory_type": memory_type,
        "summary": summary,
        "status": status,
        "source_kind": source_kind,
        "sensitivity": sensitivity,
        "content_hash": _sha256(
            f"{topic_key}|{memory_type}|{summary}"
        ),
        "version": 1,
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


class MemoryExtractionAcceptanceTests(unittest.TestCase):

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            _parse_json_object(
                '{"facts": [], "facts": ["incorrect"]}'
            )

    def test_restricted_content_is_not_accepted_as_inline_memory(self):
        evidence = MemoryManager._inline_save_evidence(
            "My API key is ABC12345, please save this."
        )

        self.assertEqual(evidence, "")


class MemoryRetrievalAcceptanceTests(unittest.TestCase):

    def test_unrelated_memory_is_not_eligible(self):
        result = MemoryManager._score_retrieval_row(
            {
                "topic": "Music",
                "summary": "Jazz guitar and classical music.",
                "similarity": 0.05,
                "importance": 2,
                "confidence": 0.5,
                "age_seconds": 100,
            },
            ["python", "backend"],
            semantic_match=False,
        )

        self.assertFalse(result["eligible"])

    def test_relevant_memory_is_eligible(self):
        result = MemoryManager._score_retrieval_row(
            {
                "topic": "Python backend",
                "summary": "FastAPI and PostgreSQL project.",
                "similarity": 0.85,
                "importance": 5,
                "confidence": 0.95,
                "age_seconds": 100,
            },
            ["python", "backend"],
            semantic_match=True,
        )

        self.assertTrue(result["eligible"])


class MemoryRepairAcceptanceTests(unittest.TestCase):

    def test_inactive_privacy_risk_is_quarantined(self):
        row = make_memory_row(
            status="approved",
            active=False,
            source_kind="automatic_extraction",
            sensitivity="personal",
        )

        store = FakeStore(
            FakeConnection(FakeCursor(rows=[row]))
        )

        manager = MemoryManager(store)
        report = manager.diagnose_memory_integrity(limit=10)

        actions = report["repair_actions"]

        self.assertEqual(len(actions), 1)
        self.assertIn("quarantine", actions[0]["actions"])
        self.assertNotIn(
            "normalize_active",
            actions[0]["actions"],
        )

    def test_scan_detects_records_beyond_requested_limit(self):
        rows = [
            make_memory_row(memory_id=1),
            make_memory_row(memory_id=2),
        ]

        store = FakeStore(
            FakeConnection(FakeCursor(rows=rows))
        )

        manager = MemoryManager(store)
        report = manager.diagnose_memory_integrity(limit=1)

        self.assertEqual(report["records_scanned"], 1)
        self.assertTrue(report["scan_may_be_incomplete"])

    def test_exact_scan_size_is_not_false_positive(self):
        store = FakeStore(
            FakeConnection(
                FakeCursor(rows=[make_memory_row()])
            )
        )

        manager = MemoryManager(store)
        report = manager.diagnose_memory_integrity(limit=1)

        self.assertFalse(report["scan_may_be_incomplete"])

    def test_no_action_repair_returns_complete_result_contract(self):
        manager = object.__new__(MemoryManager)

        manager.preview_memory_repair = MagicMock(
            return_value={
                "plan_fingerprint": "f" * 64,
                "repair_actions": [],
                "scan_may_be_incomplete": True,
            }
        )

        result = manager.apply_memory_repairs("f" * 64)

        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["repaired_records"], 0)
        self.assertEqual(result["repaired_actions"], 0)
        self.assertEqual(result["remaining_records"], 0)
        self.assertTrue(result["scan_may_be_incomplete"])

    def test_quarantined_memory_cannot_be_restored_normally(self):
        row = make_memory_row(
            status="quarantined",
            active=False,
            sensitivity="personal",
        )

        manager = MemoryManager(
            FakeStore(
                FakeConnection(FakeCursor(row=row))
            )
        )

        manager._persist_memory = MagicMock()

        result = manager.restore_memory(1)

        self.assertEqual(result["status"], "denied")
        self.assertEqual(
            result["reason"],
            "quarantined_requires_manual_privacy_review",
        )

        manager._persist_memory.assert_not_called()

    def test_restore_rejects_a_newer_concurrent_memory(self):
        connection = FakeConnection(
            FakeCursor(),
            FakeCursor(row={"id": 99}),
        )

        store = FakeStore(connection)
        manager = MemoryManager(store)

        candidate = {
            "topic_key": "project.zara",
            "topic": "Zara",
            "memory_type": "project",
            "summary": "User is developing Zara V2.",
            "importance": 4,
            "confidence": 0.95,
            "sensitivity": "normal",
        }

        with self.assertRaisesRegex(
            ValueError,
            "memory_stale_target",
        ):
            manager._persist_memory(
                candidate,
                source_kind="user_corrected",
                status="approved",
                provenance_event={
                    "origin": "user_memory_restore",
                    "evidence_hash": "restore-test",
                },
                expected_no_current=True,
            )

        self.assertEqual(len(connection.calls), 2)

    def test_concurrent_rollback_rechecks_snapshot_state(self):
        connection = FakeConnection(
            FakeCursor(
                rows=[{"topic_key": "project.zara"}]
            ),
            FakeCursor(),
            FakeCursor(rows=[]),
        )

        manager = MemoryManager(FakeStore(connection))

        result = manager.rollback_memory_repair("a" * 32)

        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(result["rolled_back"], 0)
        self.assertEqual(result["conflicts"], 0)

        self.assertIn(
            "FOR UPDATE OF s",
            connection.calls[2][0],
        )


class DatabaseRepairIntegrationAcceptanceTests(unittest.TestCase):

    def test_repair_confirmation_passes_limit_and_cancellation(self):
        service = object.__new__(DatabaseToolService)

        manager = MagicMock()
        service.memory_manager = manager

        fingerprint = "f" * 64
        event = Event()

        manager.preview_memory_repair.return_value = {
            "plan_fingerprint": fingerprint,
            "scan_may_be_incomplete": True,
        }

        manager.apply_memory_repairs.return_value = {
            "status": "completed",
            "repair_id": "a" * 32,
            "repaired_records": 5,
            "repaired_actions": 7,
            "remaining_records": 2,
            "scan_may_be_incomplete": True,
        }

        pending = _PendingMutation(
            token="confirmation-token",
            action="repair",
            memory_id=0,
            expected_content="",
            new_content=None,
            label="Memory repair",
            expires_at=time.monotonic() + 60,
            scope="topic_memory",
            payload={
                "plan_fingerprint": fingerprint,
                "limit": 321,
            },
        )

        result = service._confirm_topic_memory_mutation(
            pending,
            started_at=time.monotonic(),
            cancel_event=event,
        )

        manager.apply_memory_repairs.assert_called_once_with(
            fingerprint,
            limit=321,
            trace_id="confirmation-token",
            cancel_event=event,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["remaining_records"], 2)
        self.assertTrue(
            result["data"]["scan_may_be_incomplete"]
        )

    def test_rollback_confirmation_passes_cancellation(self):
        service = object.__new__(DatabaseToolService)

        manager = MagicMock()
        service.memory_manager = manager

        repair_id = "b" * 32
        event = Event()

        manager.preview_memory_repair_rollback.return_value = {
            "rollback_available": 2,
        }

        manager.rollback_memory_repair.return_value = {
            "status": "completed",
            "repair_id": repair_id,
            "rolled_back": 2,
            "conflicts": 0,
        }

        pending = _PendingMutation(
            token="rollback-token",
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
            cancel_event=event,
        )

        manager.rollback_memory_repair.assert_called_once_with(
            repair_id,
            trace_id="rollback-token",
            cancel_event=event,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["rolled_back"], 2)


class ChatEngineAcceptanceTests(unittest.TestCase):

    def test_api_result_contains_actual_answer_and_contract_fields(self):
        engine = object.__new__(ChatEngine)

        engine.reply = MagicMock(
            return_value="Actual Zara response."
        )

        engine.last_task_results = [
            {
                "operation": "database_confirm",
                "status": "success",
                "system_message": "Operation verified.",
                "data": {},
            }
        ]

        engine.last_improvement_trace_id = "trace-123"

        engine.build_display_events = MagicMock(
            return_value=[
                {"speaker": "Zara", "message": "Actual Zara response."}
            ]
        )

        result = engine.reply_with_result("Hello")

        self.assertEqual(
            result["assistant_message"],
            "Actual Zara response.",
        )

        self.assertEqual(len(result["system_results"]), 1)
        self.assertIn("display_events", result)
        self.assertEqual(result["trace_id"], "trace-123")

    def test_confirmation_forwards_cancellation_event(self):
        engine = object.__new__(ChatEngine)

        event = Event()

        engine.database_tools = MagicMock()
        engine.database_tools.confirm_action.return_value = {}

        engine._record_database_control_result = MagicMock(
            return_value="Confirmation processed."
        )

        ChatEngine.confirm_database_action(
            engine,
            "one-time-token",
            cancel_event=event,
        )

        engine.database_tools.confirm_action.assert_called_once_with(
            "one-time-token",
            cancel_event=event,
        )

    def test_repair_response_reports_remaining_records(self):
        result = {
            "operation": "database_confirm",
            "status": "success",
            "data": {
                "action": "repair",
                "repair_id": "a" * 32,
                "repaired_records": 5,
                "repaired_actions": 7,
                "remaining_records": 2,
                "scan_may_be_incomplete": True,
            },
        }

        answer = ChatEngine._format_task_assistant_message(
            None,
            result,
            "hinglish",
        )

        self.assertIn("5 record(s)", answer)
        self.assertIn("2 repairable", answer)
        self.assertIn("Repair ID", answer)


if __name__ == "__main__":
    unittest.main()