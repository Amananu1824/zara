"""Interactive terminal UI for Zara V2."""
from __future__ import annotations

import time
from typing import Any
import json

from app.improvement_repair.code_repair import CodeRepairService
from app.ai.chat_engine import ChatEngine
from app.ai.ollama_client import OllamaClient, OllamaError
from app.core.prompt_manager import PromptManager
from app.terminal.language import normalize_language_mode
from app.terminal.ollama_stream import stream_chat_response
from app.learning_ai.document_ingest import DocumentIngestor
from app.learning_ai.profile_defaults import (
    DEFAULT_PROFILE_FIELDS,
    get_profile_setup_status,
    save_default_profile_fields,
)


HELP_TEXT = """Commands:
  /help                         Show these commands
  /reset                        Clear this conversation's history
  /mode hinglish                Use natural Roman Hinglish
  /mode hindi                   Use Hindi in Devanagari script
  /mode english                 Use English
  /feedback good                Rate the last answer as good
  /feedback bad <reason>        Rate the last answer as bad
  /feedback-list                Show recent feedback records
  /feedback-stats               Show feedback counts
  /feedback-review <id>         Approve a reviewed correction
  /learning-index               Index approved guidance and lesson embeddings
  /lesson-candidates            List System A validated lesson candidates
  /lesson-promote <id>          Human-approve and publish one validated lesson
  /lesson-production [all]      Show active (or all) production lessons
  /lesson-history <id>          Show System A promotion/rollback history
  /lesson-rollback <id>         Roll back one active production lesson
  /profile                      Show saved profile fields
  /profile-setup preview         Preview default profile fields (no changes)
  /profile-setup confirm         Save missing defaults; preserve existing fields
  /profile-set <key> <value>    Save or update a profile field
  /profile-delete <key>         Delete one profile field
  /profile-clear preview         Preview profile fields; no changes
  /profile-clear confirm         Clear only the unchanged previewed fields
  /exit                         Quit Zara
  /ingest "<file path>"         Index supported PDF, DOCX, TXT, MD, CSV, JSON or HTML
  /documents                    List indexed documents
  /document-archive <id>        Archive an active document
  /document-restore <id>        Restore a previous document version
  /document-history <id>        Show document lifecycle history
  /doc-search <query>           Search indexed document chunks
  /memory-latest [all|profile|notes]   Show recent saved records
  /memory-search <query>               Search saved profile fields and notes
  /db-confirm <token>                  Confirm one pending database change
  /db-cancel <token>                   Cancel one pending database change
    
  /lesson-auto-promotion status       Show System A authorization
  /lesson-auto-promotion enable       Grant standing promotion permission
  /lesson-auto-promotion disable      Revoke standing promotion permission

  /code-repair list                   List pending code proposals
  /code-repair list all               List all code proposals
  /code-repair show <id>              Inspect complete proposed diff
  /code-repair approve <id>           Approve a staged patch
  /code-repair reject <id>            Reject a staged patch
  /code-repair apply <id>             Apply an approved patch
  /code-repair rollback <id>          Restore its original backup
  /code-repair history <id>           Inspect patch audit history
"""


def _format_ollama_duration(
    metrics: dict[str, object],
    key: str,
) -> str:
    value = metrics.get(key)

    if not isinstance(value, (int, float)):
        return "n/a"

    return f"{value / 1_000_000_000:.2f}s"

def _format_ollama_seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"

    return f"{value:.2f}s"

def run_chat_loop() -> None:
    try:
        engine = ChatEngine(
            client=OllamaClient(),
            prompt_manager=PromptManager(),
            language_mode="hinglish",
            preferred_address="Boss",
            allow_database_tools=True,
            allow_database_mutations=True,
        )
        document_ingestor = DocumentIngestor(engine.learning_store)
        code_repair_service = CodeRepairService(
            engine.learning_store
        )

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Zara setup error: {exc}")
        return
    
    profile_clear_preview = None

    print("Zara V2 is ready. Type /help to see commands; /exit to quit Zara.\n")
    print(
        "Note: Voice identity is not verified. 'Boss' is your configured "
        "form of address, not identity verification.\n"
    )

    while True:
        try:
            user_message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nZara: Bye! Phir baat karenge.")
            break

        if not user_message:
            continue

        command = user_message.lower().strip()

        if command in {"/exit", "/quit", "/bye"}:
            print("Zara: Bye! Phir baat karenge.")
            break

        if command == "/help":
            print(HELP_TEXT)
            continue

        if command == "/lesson-candidates":
            try:
                candidates = (
                    engine.improvement_observer.list_lesson_candidates(
                        status="validated",
                        limit=50,
                    )
                )
            except Exception as exc:
                print(
                    f"[System A candidates error] "
                    f"{type(exc).__name__}\n"
                )
                continue

            if not candidates:
                print(
                    "No validated System A candidates are available.\n"
                )
                continue

            print("Validated System A lesson candidates:")

            for item in candidates:
                print(
                    f"  #{item['id']} | "
                    f"{item['rule_key']} v{item['version']} | "
                    f"{item['risk_level']} | "
                    f"confidence={float(item['confidence']):.3f}"
                )

                print(
                    f"    {str(item['lesson_text'])[:280]}"
                )

            print()
            continue

        if (
            command == "/lesson-production"
            or command.startswith("/lesson-production ")
        ):
            pieces = user_message.split(maxsplit=1)

            show_all = (
                len(pieces) == 2
                and pieces[1].strip().lower() == "all"
            )

            if len(pieces) == 2 and not show_all:
                print("Use: /lesson-production [all]\n")
                continue

            try:
                lessons = (
                    engine.improvement_observer.list_production_lessons(
                        active_only=not show_all,
                        limit=100,
                    )
                )
            except Exception as exc:
                print(
                    f"[System A production error] "
                    f"{type(exc).__name__}\n"
                )
                continue

            if not lessons:
                print("No production lessons found.\n")
                continue

            print("System A production lessons:")

            for item in lessons:
                state = (
                    "ACTIVE" if item["active"] else "INACTIVE"
                )

                print(
                    f"  #{item['id']} "
                    f"candidate=#{item['candidate_id']} | "
                    f"{state} | "
                    f"{item['rule_key']} v{item['version']} | "
                    f"{item['scope']} | "
                    f"priority={item['priority']}"
                )

                print(
                    f"    {str(item['lesson_text'])[:280]}"
                )

            print()
            continue

        if command.startswith("/lesson-history "):
            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print(
                    "Use: /lesson-history <candidate_id>\n"
                )
                continue

            try:
                history = (
                    engine.improvement_observer.lesson_promotion_history(
                        int(pieces[1]),
                        limit=30,
                    )
                )
            except Exception as exc:
                print(
                    f"[System A history error] "
                    f"{type(exc).__name__}\n"
                )
                continue

            if not history:
                print(
                    "No promotion history found for this candidate.\n"
                )
                continue

            for event in history:
                print(
                    f"  Event #{event['id']} | "
                    f"{event['action']} | "
                    f"{event['outcome']} | "
                    f"reviewer={event['reviewer']}"
                )

                print(f"    {event['note']}")

            print()
            continue

        if command.startswith("/lesson-promote "):
            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print(
                    "Use: /lesson-promote <candidate_id>\n"
                )
                continue

            candidate_id = int(pieces[1])

            candidate = (
                engine.improvement_observer.get_lesson_candidate(
                    candidate_id
                )
            )

            if (
                candidate is None
                or candidate.get("status") != "validated"
            ):
                print(
                    "Only an existing validated candidate "
                    "can be promoted.\n"
                )
                continue

            print("Candidate selected for production approval:")

            print(
                f"  #{candidate['id']} | "
                f"{candidate['rule_key']} "
                f"v{candidate['version']} | "
                f"{candidate['risk_level']}"
            )

            print(
                f"  {candidate['lesson_text']}\n"
            )

            try:
                reviewer = input(
                    "Reviewer label "
                    "(audit label, not identity verification): "
                ).strip()

                note = input(
                    "Approval reason (minimum 20 characters): "
                ).strip()

                raw_priority = input(
                    "Priority 1-100 [50]: "
                ).strip()

                try:
                    priority = (
                        int(raw_priority)
                        if raw_priority
                        else 50
                    )
                except ValueError:
                    print(
                        "Priority must be an integer from 1 to 100.\n"
                    )
                    continue

                resolve_input = input(
                    f"For an intentional same-rule conflict "
                    f"resolution, type RESOLVE {candidate_id}; "
                    "otherwise press Enter: "
                ).strip()

                confirmation = input(
                    f"Type PROMOTE {candidate_id} "
                    "to publish this rule: "
                ).strip()

            except (KeyboardInterrupt, EOFError):
                print(
                    "\nPromotion cancelled; no changes requested.\n"
                )
                continue

            if confirmation != f"PROMOTE {candidate_id}":
                print(
                    "Promotion cancelled; "
                    "no production change made.\n"
                )
                continue

            try:
                result = (
                    engine.improvement_observer.promote_lesson_candidate(
                        candidate_id,
                        reviewer=reviewer,
                        note=note,
                        approval_confirmed=True,
                        priority=priority,
                        resolve_same_rule_conflict=(
                            resolve_input == f"RESOLVE {candidate_id}"
                        ),
                    )
                )

            except Exception as exc:
                print(
                    f"[System A promotion error] "
                    f"{type(exc).__name__}\n"
                )

            else:
                print(f"Promotion result: {result}\n")

            continue

        if command.startswith("/lesson-rollback "):
            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print(
                    "Use: /lesson-rollback <candidate_id>\n"
                )
                continue

            candidate_id = int(pieces[1])

            try:
                reviewer = input(
                    "Reviewer label "
                    "(audit label, not identity verification): "
                ).strip()

                note = input(
                    "Rollback reason (minimum 20 characters): "
                ).strip()

                confirmation = input(
                    f"Type ROLLBACK {candidate_id} "
                    "to deactivate and restore the previous version: "
                ).strip()

            except (KeyboardInterrupt, EOFError):
                print(
                    "\nRollback cancelled; no changes requested.\n"
                )
                continue

            if confirmation != f"ROLLBACK {candidate_id}":
                print(
                    "Rollback cancelled; "
                    "no production change made.\n"
                )
                continue

            try:
                result = (
                    engine.improvement_observer.rollback_lesson_promotion(
                        candidate_id,
                        reviewer=reviewer,
                        note=note,
                        confirmed=True,
                    )
                )

            except Exception as exc:
                print(
                    f"[System A rollback error] "
                    f"{type(exc).__name__}\n"
                )

            else:
                print(f"Rollback result: {result}\n")

            continue
        # ----------------------------------------------
        # A4: Standing automatic lesson promotion
        # ----------------------------------------------

        if command.startswith("/lesson-auto-promotion"):

            pieces = command.split()

            if len(pieces) != 2 or pieces[1] not in {
                "status",
                "enable",
                "disable",
            }:
                print(
                    "Use /lesson-auto-promotion status, "
                    "enable or disable.\n"
                )
                continue

            action = pieces[1]

            if action == "status":
                try:
                    result = (
                        engine.improvement_observer
                        .automatic_promotion_status()
                    )
                except Exception as exc:
                    print(
                        f"[Authorization error] "
                        f"{type(exc).__name__}\n"
                    )
                else:
                    print(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        "\n",
                    )

                continue

            expected = (
                "AUTHORIZE ZARA SYSTEM A PROMOTION"
                if action == "enable"
                else "REVOKE ZARA SYSTEM A PROMOTION"
            )

            try:
                note = input(
                    "Authorization reason (minimum 20 characters): "
                ).strip()

                confirmation = input(
                    f"Type exactly: {expected}\nConfirmation: "
                ).strip()

            except (KeyboardInterrupt, EOFError):
                print("\nAuthorization operation cancelled.\n")
                continue

            if confirmation != expected:
                print(
                    "Authorization not changed. "
                    "No permission was granted.\n"
                )
                continue

            try:
                result = (
                    engine.improvement_observer
                    .set_automatic_promotion_authorization(
                        enabled=(action == "enable"),
                        reviewer="local_terminal_operator",
                        note=note,
                        confirmed=True,
                    )
                )

            except Exception as exc:
                print(
                    f"[Authorization error] "
                    f"{type(exc).__name__}\n"
                )
            else:
                print(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "\n",
                )

            continue

        # ----------------------------------------------
        # A6: Staged code and rule repair
        # ----------------------------------------------

        if command == "/code-repair" or command.startswith(
            "/code-repair "
        ):

            pieces = user_message.split(maxsplit=2)

            if len(pieces) == 1:
                action = "list"
                argument = "pending_review"

            else:
                action = pieces[1].lower()
                argument = (
                    pieces[2].strip()
                    if len(pieces) == 3
                    else ""
                )

            if action == "list":

                selected_status = argument or "pending_review"

                try:
                    rows = code_repair_service.list_proposals(
                        status=selected_status,
                        limit=50,
                    )

                except Exception as exc:
                    print(
                        f"[Code repair list error] "
                        f"{type(exc).__name__}\n"
                    )
                    continue

                if not rows:
                    print("No matching code repair proposals.\n")
                    continue

                for row in rows:
                    print(
                        f"#{row['id']} | "
                        f"{row['rule_key']} | "
                        f"{row['file_path']} | "
                        f"{row['status']}"
                    )

                    print(
                        f"  Diagnosis: {row['diagnosis'][:250]}"
                    )

                print()
                continue

            if action == "show":

                if not argument.isdigit():
                    print(
                        "Use /code-repair show <id>\n"
                    )
                    continue

                try:
                    proposal = code_repair_service.get_proposal(
                        int(argument)
                    )

                except Exception as exc:
                    print(
                        f"[Proposal error] "
                        f"{type(exc).__name__}\n"
                    )
                    continue

                if proposal is None:
                    print("Proposal not found.\n")
                    continue

                print(
                    f"Proposal #{proposal['id']}"
                )
                print(
                    f"Status: {proposal['status']}"
                )
                print(
                    f"Target: {proposal['file_path']}"
                )
                print(
                    f"Rule: {proposal['rule_key']}"
                )
                print(
                    f"Diagnosis: {proposal['diagnosis']}"
                )

                print("\n--- PROPOSED DIFF ---\n")
                print(proposal["patch_diff"])
                print("\n--- END DIFF ---\n")

                print(
                    "Static validation:",
                    json.dumps(
                        proposal["validation"],
                        ensure_ascii=False,
                        default=str,
                    ),
                    "\n",
                )

                continue

            if action == "history":

                if not argument.isdigit():
                    print(
                        "Use /code-repair history <id>\n"
                    )
                    continue

                try:
                    events = code_repair_service.proposal_history(
                        int(argument)
                    )

                except Exception as exc:
                    print(
                        f"[History error] "
                        f"{type(exc).__name__}\n"
                    )
                    continue

                for event in events:
                    print(
                        f"#{event['id']} | "
                        f"{event['event_type']} | "
                        f"{event['actor']}"
                    )
                    print(f"  {event['note']}")

                print()
                continue

            if action in {
                "approve",
                "reject",
                "apply",
                "rollback",
            }:

                if not argument.isdigit():
                    print(
                        f"Use /code-repair {action} <id>\n"
                    )
                    continue

                proposal_id = int(argument)

                try:
                    reviewer = input(
                        "Reviewer label "
                        "(audit label, not identity verification): "
                    ).strip()

                    note = input(
                        "Reason (minimum 20 characters): "
                    ).strip()

                    if action == "approve":
                        expected = (
                            f"APPROVE CODE PATCH {proposal_id}"
                        )

                    elif action == "reject":
                        expected = (
                            f"REJECT CODE PATCH {proposal_id}"
                        )

                    elif action == "apply":
                        expected = (
                            f"APPLY CODE PATCH {proposal_id}"
                        )

                    else:
                        expected = (
                            f"ROLLBACK CODE PATCH {proposal_id}"
                        )

                    confirmation = input(
                        f"Type exactly: {expected}\nConfirmation: "
                    ).strip()

                except (KeyboardInterrupt, EOFError):
                    print("\nOperation cancelled.\n")
                    continue

                if confirmation != expected:
                    print(
                        "Operation cancelled. "
                        "No requested change was made.\n"
                    )
                    continue

                try:

                    if action in {"approve", "reject"}:
                        result = code_repair_service.review_proposal(
                            proposal_id,
                            reviewer=reviewer,
                            note=note,
                            decision=action,
                            confirmed=True,
                        )

                    elif action == "apply":
                        result = (
                            code_repair_service.apply_approved_proposal(
                                proposal_id,
                                reviewer=reviewer,
                                note=note,
                                confirmation=confirmation,
                            )
                        )

                    else:
                        result = (
                            code_repair_service.rollback_applied_proposal(
                                proposal_id,
                                reviewer=reviewer,
                                note=note,
                                confirmation=confirmation,
                            )
                        )

                except Exception as exc:
                    print(
                        f"[Code repair error] "
                        f"{type(exc).__name__}\n"
                    )

                else:
                    print(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                            indent=2,
                            default=str,
                        ),
                        "\n",
                    )

                continue

            print(
                "Unknown code repair command. "
                "Use /help for available commands.\n"
            )
            continue
        
        if command == "/profile":
            try:
                entries = engine.learning_store.get_user_profile_entries()
            except Exception as exc:
                print(f"[Profile error] {exc}\n")
            else:
                if not entries:
                    print("No profile fields saved yet.\n")
                else:
                    print("Saved user profile:")
                    for item in entries:
                        print(f"  {item['memory_key']}: {item['content']}")
                    print()
            continue

        if command == "/memory-latest" or command.startswith("/memory-latest "):
            pieces = user_message.split(maxsplit=1)
            scope = pieces[1].strip().lower() if len(pieces) == 2 else "all"

            result = engine.database_tools.execute_tool(
                "memory_latest",
                {"scope": scope, "limit": 10},
                user_message,
            )
            print(engine.database_tools.render_result(result) + "\n")
            continue

        if command == "/memory-search" or command.startswith("/memory-search "):
            pieces = user_message.split(maxsplit=1)

            if len(pieces) != 2 or not pieces[1].strip():
                print("Use: /memory-search <query>\n")
                continue

            result = engine.database_tools.execute_tool(
                "memory_search",
                {
                    "query": pieces[1].strip(),
                    "scope": "all",
                    "limit": 10,
                },
                user_message,
            )
            print(engine.database_tools.render_result(result) + "\n")
            continue

        if command == "/profile-setup preview":
            try:
                missing, present = get_profile_setup_status(
                    engine.learning_store
                )
            except Exception as exc:
                print(f"[Profile setup error] {exc}\n")
            else:
                print("Profile setup preview (no changes made):")
                print(
                    f"  Default fields available: {len(DEFAULT_PROFILE_FIELDS)}\n"
                    f"  Fields that would be added: {len(missing)}\n"
                    f"  Existing default fields preserved: {len(present)}"
                )
                if missing:
                    print("  Missing keys: " + ", ".join(missing))
                print("Type /profile-setup confirm to save only missing fields.\n")
            continue

        if command == "/profile-setup confirm":
            try:
                saved, preserved = save_default_profile_fields(
                    engine.learning_store
                )
            except Exception as exc:
                print(f"[Profile setup error] {exc}\n")
            else:
                print(
                    f"Profile setup complete: saved {len(saved)} field(s); "
                    f"preserved {len(preserved)} existing default field(s).\n"
                )
                if not saved:
                    print("No missing default fields; nothing was changed.\n")
            continue

        if command == "/profile-setup" or command.startswith("/profile-setup "):
            print(
                "Use /profile-setup preview to inspect changes, then "
                "/profile-setup confirm to save missing fields.\n"
            )
            continue

        if command == "/profile-set" or command.startswith("/profile-set "):
            pieces = user_message.split(maxsplit=2)
            if len(pieces) != 3:
                print("Use: /profile-set <key> <value>\n")
                continue
            try:
                result = engine.request_profile_save(pieces[1], pieces[2])
            except Exception as exc:
                print(f"[Profile error] {exc}\n")
            else:
                print(result + "\n")

            continue

        if command == "/profile-delete" or command.startswith("/profile-delete "):
            pieces = user_message.split(maxsplit=1)
            if len(pieces) != 2:
                print("Use: /profile-delete <key>\n")
                continue
            try:
                result = engine.request_profile_delete_by_key(pieces[1])
            except Exception as exc:
                print(f"[Profile error] {exc}\n")
            else:
                print(result + "\n")

            continue

        if command == "/profile-clear confirm":
            pending = profile_clear_preview
            profile_clear_preview = None

            if pending is None:
                print(
                    "Pehle /profile-clear preview chalayein. "
                    "Abhi koi profile data delete nahi hua.\n"
                )
                continue

            expires_at, expected_preview = pending

            if time.monotonic() >= expires_at:
                print(
                    "Profile-clear preview expire ho gaya. "
                    "Dobara preview chalayein; koi data delete nahi hua.\n"
                )
                continue

            try:
                count = engine.learning_store.clear_user_profile(
                    expected_preview=expected_preview
                )
            except ValueError as exc:
                print(f"[Profile clear cancelled] {exc}\n")
            except Exception as exc:
                print(
                    f"[Profile clear error] {type(exc).__name__}. "
                    "Deletion result ko verify karein.\n"
                )
            else:
                print(f"Deleted {count} profile field(s).\n")

            continue

        if command in {
            "/profile-clear",
            "/profile-clear preview",
        }:
            try:
                preview_rows = (
                    engine.learning_store.get_user_profile_clear_preview()
                )
            except Exception as exc:
                profile_clear_preview = None
                print(
                    f"[Profile preview error] {type(exc).__name__}\n"
                )
            else:
                if not preview_rows:
                    profile_clear_preview = None
                    print(
                        "Profile-clear preview: koi profile field nahi mila. "
                        "Koi change nahi hua.\n"
                    )
                    continue

                profile_clear_preview = (
                    time.monotonic() + 120,
                    preview_rows,
                )

                labels = [
                    f"{row['memory_key'] or '(unnamed)'} "
                    f"(record #{row['id']})"
                    for row in preview_rows
                ]

                print("PROFILE CLEAR PREVIEW — no changes made")
                print(f"Fields in scope: {len(preview_rows)}")

                for label in labels:
                    print(f"  - {label}")

                print(
                    "Confirm within 120 seconds using "
                    "/profile-clear confirm. "
                    "Only this unchanged previewed scope will be cleared.\n"
                )

            continue

        if command.startswith("/profile-clear"):
            print(
                "Use /profile-clear preview, then "
                "/profile-clear confirm if the preview is correct.\n"
            )
            continue

        if command == "/ingest" or command.startswith("/ingest "):
            pieces = user_message.split(maxsplit=1)

            if len(pieces) != 2:
                print('Use: /ingest "C:\\path\\to\\document.pdf"\n')
                continue

            raw_path = pieces[1].strip().strip('"').strip("'")

            try:
                result = document_ingestor.ingest_file(raw_path)
            except Exception as exc:
                print(f"[Document ingestion error] {exc}\n")
            else:
                print("Document ingestion result:")
                print(f"  Status: {result['status']}")
                print(f"  ID: {result['document_id']}")
                print(f"  File: {result['file_name']}")
                print(f"  Chunks: {result['chunks']}")
                chunking = result.get("chunking")

                if isinstance(chunking, dict):
                    print(  
                        f"  Chunking: {chunking.get('strategy', 'unknown')} | "
                        f"Average: {chunking.get('average_chars', 0)} chars | "
                        f"Maximum: {chunking.get('max_chars', 0)} chars"
                    )
                    print(
                        f"  Chunk types: {chunking.get('content_kinds', {})}"
                    )
                print(f"  Pages: {result['pages']}")
                extraction = result.get("extraction", {})

                if extraction:
                    print(
                        "  Extraction method: "
                        f"{extraction.get('method', 'unknown')}"
                    )

                    print(
                        "  Extraction status: "
                        f"{extraction.get('status', 'unknown')}"
                    )

                    print(
                        "  Source encoding: "
                        f"{extraction.get('source_encoding') or 'not applicable'}"
                    )

                    for warning in extraction.get("warnings", []):
                        print(
                            f"  Extraction warning: {warning}"
                        )

                print()

            continue

        if command == "/documents":
            try:
                documents = document_ingestor.list_documents()
            except Exception as exc:
                print(f"[Documents error] {exc}\n")
                continue

            if not documents:
                print("No documents indexed yet.\n")
                continue

            for document in documents:
                print(
                    f"#{document['id']} "
                    f"[{document['ingestion_status']}] "
                    f"{document['file_name']} "
                    f"v{document['source_version']}"
                )
                print(
                    f"  Category: {document['category']} | "
                    f"Language: {document['language']} | "
                    f"Chunks: {document['chunk_count']}"
                )

                if document.get("source_path"):
                    print(f"  Source path: {document['source_path']}")

                if document.get("extraction_status"):
                    print(
                        "  Extraction: "
                        f"{document['extraction_status']} | "
                        f"{document.get('extraction_method', 'legacy')}"
                    )

                warnings = document.get("extraction_warnings") or []

                for warning in warnings:
                    print(f"  Warning: {warning}")

                if document.get("superseded_version"):
                    print(
                        "  Warning: superseded indexed version; "
                        "excluded from retrieval"
                    )

            print()
            continue

        if command.startswith("/document-archive "):

            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print("Use: /document-archive <id>\n")
                continue

            try:
                result = document_ingestor.archive_document(
                    int(pieces[1])
                )

                print(
                    f"Document archive result: {result}\n"
                )

            except Exception as exc:
                print(
                    f"[Document archive error] "
                    f"{type(exc).__name__}\n"
                )

            continue

        if command.startswith("/document-restore "):

            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print("Use: /document-restore <id>\n")
                continue

            try:
                result = document_ingestor.restore_document(
                    int(pieces[1])
                )

                print(
                    f"Document restore result: {result}\n"
                )

            except Exception as exc:
                print(
                    f"[Document restore error] "
                    f"{type(exc).__name__}\n"
                )

            continue

        if command.startswith("/document-history "):

            pieces = user_message.split()

            if len(pieces) != 2 or not pieces[1].isdigit():
                print("Use: /document-history <id>\n")
                continue

            try:
                history = document_ingestor.document_history(
                    int(pieces[1])
                )

                if not history:
                    print("No document history found.\n")
                    continue

                for event in history:
                    print(
                        f"Event #{event['id']} | "
                        f"Document #{event['document_id']} | "
                        f"{event['event_type']}"
                    )

                    print(
                        f"  Details: {event['details']}"
                    )

            except Exception as exc:
                print(
                    f"[Document history error] "
                    f"{type(exc).__name__}\n"
                )

            print()
            continue
        
        if command == "/doc-search" or command.startswith("/doc-search "):
            pieces = user_message.split(maxsplit=1)

            if len(pieces) != 2 or not pieces[1].strip():
                print("Use: /doc-search <your search query>\n")
                continue

            query = pieces[1].strip()

            try:
                results = engine.learning_store.semantic_search(query, limit=30)
            except Exception as exc:
                print(f"[Document search error] {exc}\n")
                continue

            documents = [
                result for result in results
                if result.get("source") == "document_chunk"
            ]

            if not documents:
                print("No relevant document chunks found.\n")
                continue

            for index, result in enumerate(documents[:5], start=1):
                source = result.get("source_file") or result.get("title")
                page = result.get("page_number")
                page_text = f", page {page}" if page is not None else ""

                print(
                    f"{index}. {source}{page_text} | "
                    f"similarity {float(result['similarity']):.4f}"
                )
                print(f"   {result['content'][:350]}\n")

            continue

        if command == "/reset":
            engine.reset()
            print("Zara: Conversation history clear kar di hai.\n")
            continue

        if command == "/learning-index":
            started = time.perf_counter()

            try:
                stats = engine.learning_store.index_pending_embeddings()
            except Exception as exc:
                elapsed = time.perf_counter() - started
                print(f"[Learning index error] {exc}")
                print(f"[Index time: {elapsed:.2f} s]\n")
            else:
                elapsed = time.perf_counter() - started

                print("Learning index result:")
                print(f"  Model: {stats['embedding_model']}")
                print(
                    "  Behavior guides indexed: "
                    f"{stats['behavior_guides_indexed']}"
                )
                print(
                    "  Feedback lessons indexed: "
                    f"{stats['feedback_lessons_indexed']}"
                )
                print(
                    "  System A lessons indexed: "
                    f"{stats.get('system_a_lessons_indexed', 0)}"
                )
                print(
                    "  Pending records before indexing: "
                    f"{stats['pending_before_indexing']}"
                )
                print(f"  Index time: {elapsed:.2f} s\n")

            continue

        if command == "/feedback-stats":
            stats = engine.feedback_stats()

            print(
                f"Feedback records: {stats['total']} | "
                f"Good: {stats['good']} | "
                f"Bad: {stats['bad']} | "
                f"Reviewed: {stats['reviewed']}\n"
            )
            continue

        if command == "/feedback-list":
            rows = engine.list_feedback(10)

            if not rows:
                print("No feedback records saved yet.\n")
                continue

            for row in rows:
                rating = "GOOD" if row["rating"] == 1 else "BAD"
                status = "reviewed" if row["reviewed"] else "pending"

                print(
                    f"#{row['id']} [{rating}] [{status}] "
                    f"{row['created_at']}"
                )
                print(f"  You: {row['user_message'][:180]}")
                print(f"  Zara: {row['assistant_response'][:180]}")
                print(f"  Note: {row['note'] or '(no reason supplied)'}")
                print()

            continue

        if command.startswith("/feedback-review"):
            pieces = user_message.split(maxsplit=1)

            if len(pieces) != 2:
                print("Use: /feedback-review <id>\n")
                continue

            try:
                feedback_id = int(pieces[1])

                result = engine.database_tools.execute_tool(
                    "feedback_review",
                    {"feedback_id": feedback_id},
                    user_message,
                )

                print(engine.database_tools.render_result(result) + "\n")

            except (ValueError, TypeError) as exc:
                print(f"[Feedback error] {exc}\n")
            except Exception as exc:
                print(f"[Feedback error] {type(exc).__name__}\n")

            continue

        if command == "/feedback" or command.startswith("/feedback "):
            pieces = user_message.split(maxsplit=2)

            if len(pieces) < 2 or pieces[1].lower() not in {"good", "bad"}:
                print(
                    "Use /feedback good or "
                    "/feedback bad <reason>\n"
                )
                continue

            rating = 1 if pieces[1].lower() == "good" else -1
            note = pieces[2].strip() if len(pieces) > 2 else ""

            try:
                feedback_id = engine.record_feedback(rating, note)
            except ValueError as exc:
                print(f"[Feedback error] {exc}\n")
                continue

            print(f"Feedback saved as #{feedback_id}.")

            if rating == -1 and note:
                print(
                    "To approve this correction for future relevant "
                    f"responses, run /feedback-review {feedback_id}."
                )
            elif rating == -1:
                print(
                    "No reason was supplied, so Zara cannot infer the "
                    "specific mistake from this rating alone."
                )

            print()
            continue

        if command.startswith("/mode"):
            pieces = command.split(maxsplit=1)
            mode = normalize_language_mode(
                pieces[1] if len(pieces) > 1 else ""
            )

            if not mode:
                print(
                    "Use: /mode hinglish, /mode hindi, or /mode english\n"
                )
            else:
                engine.set_language_mode(mode)
                print(f"Zara: Language mode set to {mode}.\n")

            continue

        previous_callback = engine.system_result_callback
        displayed_result_ids: set[int] = set()
        zara_prefix_printed = False

        def display_system_result(result: dict[str, Any]) -> None:
            displayed_result_ids.add(id(result))

            status = str(
                result.get("status") or "failed"
            ).upper()

            message = str(
                result.get("system_message") or ""
            )

            print(
                f"\nSystem: [{status}] {message}"
            )

            data = result.get("data") or {}
            preview = data.get("confirmation_preview")

            if isinstance(preview, str) and preview:
                print(f"System: {preview}")

            if callable(previous_callback):
                previous_callback(result)

        engine.system_result_callback = display_system_result

        engine.client.last_stream_metrics = None
        start_time = time.perf_counter()
        first_output_time: float | None = None

        stream = stream_chat_response(engine, user_message)

        try:
            for chunk in stream:
                if chunk:
                    if first_output_time is None:
                        first_output_time = time.perf_counter()

                    if not zara_prefix_printed:
                        print("\nZara: ", end="", flush=True)
                        zara_prefix_printed = True

                    print(chunk, end="", flush=True)

            if zara_prefix_printed:
                print()

        except KeyboardInterrupt:
            engine.cancel_active_stream()

            try:
                stream.close()
            except Exception:
                pass

            print(
                "\n[Stream cancelled. "
                "Checking completed operation results...]\n"
            )

            # for task_result in engine.last_task_results:
            #     print(
            #         f"[System result: {task_result['status'].upper()}] "
            #         f"{task_result['system_message']}"
            #     )

        except (OllamaError, ValueError) as exc:
            try:
                stream.close()
            except Exception:
                pass

            print(f"\n[Zara error] {exc}")

        except Exception as exc:
            try:
                stream.close()
            except Exception:
                pass

            print(f"\n[Unexpected error] {type(exc).__name__}")

        finally:
            for task_result in engine.last_task_results:
                if id(task_result) not in displayed_result_ids:
                    display_system_result(task_result)

            if not zara_prefix_printed and engine.last_task_results:
                assistant_message = str(
                    engine.last_task_results[-1].get(
                        "assistant_message"
                    ) or ""
                )

                if assistant_message:
                    print(f"\nZara: {assistant_message}\n")

            engine.system_result_callback = previous_callback

            total_seconds = time.perf_counter() - start_time

            if first_output_time is None:
                first_output = "not received"
            else:
                first_seconds = first_output_time - start_time
                first_output = f"{first_seconds:.2f} s"

            rag_seconds = engine.last_retrieval_seconds
            rag_mode = engine.last_retrieval_mode
            rag_count = engine.last_retrieved_guidance_count

            print(
                f"[Timing: RAG {rag_seconds:.2f} s "
                f"({rag_mode}, {rag_count} item(s)) | "
                f"first output {first_output} | "
                f"total response {total_seconds:.2f} s]\n"
            )

            manifest = getattr(
                engine,
                "last_retrieval_manifest",
                {},
            )
            selected = manifest.get("selected", [])

            if selected:
                evidence_labels = []

                for item in selected[:4]:
                    label = (
                        f"{item.get('source', 'unknown')}"
                        f"#{item.get('source_id', '?')}"
                        f" score={item.get('similarity', 0):.3f}"
                    )

                    source_file = item.get("source_file")
                    page = item.get("page_number")

                    if source_file:
                        label += f" file={source_file}"
                    if page is not None:
                        label += f" page={page}"

                    evidence_labels.append(label)

                print(
                    "[RAG Evidence: "
                    + " | ".join(evidence_labels)
                    + "]\n"
                )
            else:
                print(
                    "[RAG Evidence: "
                    f"{manifest.get('status', 'not_available')}]"
                )

            turn_metrics = getattr(engine, "last_turn_metrics", {})

            if turn_metrics:
                print(
                    "[Pipeline: "
                    f"profile_db={_format_ollama_seconds(turn_metrics.get('profile_load_seconds'))} | "
                    f"context={_format_ollama_seconds(turn_metrics.get('context_build_seconds'))} | "
                    f"generation={_format_ollama_seconds(turn_metrics.get('generation_seconds'))} | "
                    f"prompt_chars={turn_metrics.get('prompt_chars', 'n/a')} | "
                    f"prompt_messages={turn_metrics.get('prompt_messages', 'n/a')} | "
                    f"history_messages={turn_metrics.get('history_messages', 'n/a')} | "
                    f"status={turn_metrics.get('status', 'n/a')}]\n"
                )

            context = turn_metrics.get("context_manifest", {})

            if context:
                source_chars = context.get("source_chars", {})
                source_items = context.get("source_items", {})

                print(
                    "[Context: "
                    f"base={source_chars.get('base_prompt', 'n/a')} chars | "
                    f"history={source_chars.get('history', 'n/a')} chars/"
                    f"{source_items.get('history_messages', 'n/a')} messages | "
                    f"profile={source_chars.get('profile', 'n/a')} chars/"
                    f"{source_items.get('profile_fields', 'n/a')} fields | "
                    f"guidance={source_chars.get('behavior_guidance', 'n/a')} chars | "
                    f"feedback={source_chars.get('feedback_lessons', 'n/a')} chars | "
                    f"documents={source_chars.get('documents', 'n/a')} chars/"
                    f"{source_items.get('document_chunks', 'n/a')} chunks | "
                    f"total_prompt={source_chars.get('all_messages', 'n/a')} chars]"
                )
            
            metrics = (
                engine.client.last_stream_metrics
                or engine.client.last_chat_metrics
            )

            if metrics:
                eval_duration_ns = metrics.get("eval_duration")
                eval_count = metrics.get("eval_count")

                if (
                    isinstance(eval_duration_ns, (int, float))
                    and eval_duration_ns > 0
                    and isinstance(eval_count, (int, float))
                ):
                    speed = eval_count * 1_000_000_000 / eval_duration_ns
                    speed_text = f"{speed:.2f} tok/s"
                else:
                    speed_text = "n/a"

                print(
                    "[Ollama metrics: "
                    f"model={metrics.get('model', 'n/a')} | "
                    f"load={_format_ollama_duration(metrics, 'load_duration')} | "
                    f"prompt_eval={_format_ollama_duration(metrics, 'prompt_eval_duration')} | "
                    f"generation={_format_ollama_duration(metrics, 'eval_duration')} | "
                    f"server_total={_format_ollama_duration(metrics, 'total_duration')} | "
                    f"client_request={_format_ollama_seconds(metrics.get('request_wall_seconds'))} | "
                    f"model_first_content={_format_ollama_seconds(metrics.get('first_content_seconds'))} | "
                    f"input_tokens={metrics.get('prompt_eval_count', 'n/a')} | "
                    f"cached_input_tokens={metrics.get('prompt_eval_cached_count', 'n/a')} | "
                    f"output_tokens={metrics.get('eval_count', 'n/a')} | "
                    f"speed={speed_text} | "
                    f"done_reason={metrics.get('done_reason', 'n/a')}]\n"
                )
