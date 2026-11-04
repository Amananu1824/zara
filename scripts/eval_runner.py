"""Run Zara V2 behavior evaluation cases without modifying runtime code.

- Uses the real ChatEngine and Ollama streaming path.
- Creates a fresh conversation engine for every independent case.
- Reads existing learning data through PostgreSQL read-only transactions.
- Does not execute arbitrary test_setup instructions.
- Saves results locally as JSON for later comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import hashlib
import math
from collections import Counter, defaultdict
from statistics import median
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "app"
    / "knowledge"
    / "hinglish_playbook_eval_cases.jsonl"
)

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "evaluations"

PAGE_PATTERN = re.compile(
    r"\b(?:(?:physical|pdf)\s+)?page(?:\s+number)?"
    r"\s*[:#-]?\s*(\d{1,4})\b"
    r"|\bp\.\s*(\d{1,4})\b"
    r"|पेज(?:\s*नंबर)?\s*[:#-]?\s*(\d{1,4})"
    r"|पृष्ठ(?:\s*संख्या)?\s*[:#-]?\s*(\d{1,4})",
    re.IGNORECASE,
)

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")

STOCK_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:"
    r"aur\s+kuch\s+(?:chahiye|puchna|poochna|madad|help)|"
    r"anything\s+else|"
    r"need\s+anything\s+else|"
    r"if\s+you\s+(?:want|would\s+like)|"
    r"let\s+me\s+know\s+if\s+you\s+need"
    r")\b",
    re.IGNORECASE,
)

_PROFILE_FACT_KEYS = frozenset({
    "education",
    "current_learning",
    "current_goal",
    "long_term_goals",
    "interests",
    "hobbies",
    "business_focus",
})

_PROFILE_STOP_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this",
    "your", "you", "are", "was", "have", "has", "about",
    "mein", "mere", "meri", "mera", "hai", "hain", "kar",
    "raha", "rahe", "ki", "ka", "ke", "aur", "par",
    "under", "into", "through", "using",
})

_GENERIC_PRAISE_PATTERN = re.compile(
    r"\b(?:you are|you're|aap|tum)\b.{0,35}"
    r"\b(?:amazing|awesome|brilliant|talented|hardworking|"
    r"mehnati|zabardast|shandaar)\b"
    r"|"
    r"\b(?:proud of you|aapki mehnat.{0,35}"
    r"(?:great|amazing|achhi|acchi)|"
    r"(?:aapka|aapki|aapke)\b.{0,35}"
    r"(?:bahut achha|bahut achhi|bohot achha|"
    r"amazing|awesome|zabardast))\b",
    re.IGNORECASE,
)


def _match_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


def _measure_profile_grounding(
    required_context: dict[str, Any],
    profile_entries: list[dict[str, Any]] | None,
    answer: str,
) -> tuple[int, int, list[str]]:
    requested_fields = required_context.get("profile_fields", [])

    if not isinstance(requested_fields, list):
        return 0, 0, []

    saved_values = {
        str(item.get("memory_key") or "").strip().casefold():
            str(item.get("content") or "").strip()
        for item in (profile_entries or [])
        if isinstance(item, dict)
    }

    answer_tokens = set(_match_tokens(answer))
    available = 0
    covered: list[str] = []

    for raw_key in requested_fields:
        key = str(raw_key).strip().casefold()
        value = saved_values.get(key, "")

        if not key or not value:
            continue

        available += 1
        fact_tokens = {
            token
            for token in _match_tokens(value)
            if len(token) >= 3 and token not in _PROFILE_STOP_WORDS
        }

        if not fact_tokens:
            normalized_value = " ".join(_match_tokens(value))
            if normalized_value and normalized_value in " ".join(
                _match_tokens(answer)
            ):
                covered.append(key)
            continue

        required_overlap = (
            1 if len(fact_tokens) == 1
            else max(2, (len(fact_tokens) + 1) // 2)
        )

        if len(fact_tokens & answer_tokens) >= required_overlap:
            covered.append(key)

    return available, len(covered), covered


def _find_irrelevant_profile_leaks(
    answer: str,
    profile_entries: list[dict[str, Any]] | None,
) -> list[str]:
    normalized_answer = " ".join(_match_tokens(answer))
    leaks: list[str] = []

    for item in profile_entries or []:
        key = str(item.get("memory_key") or "").strip().casefold()
        value = str(item.get("content") or "").strip()

        if key not in _PROFILE_FACT_KEYS or not value:
            continue

        value_tokens = _match_tokens(value)
        normalized_value = " ".join(value_tokens)

        # Only flag a clearly repeated stored fact, not a coincidental
        # mention of a generic word such as "project" or "learning".
        if (
            len(set(value_tokens)) >= 3
            and len(normalized_value) >= 18
            and normalized_value in normalized_answer
        ):
            leaks.append(key)

    return sorted(set(leaks))

def find_repeated_sentences(answer: str) -> list[str]:
    """Find exact repeated prose sentences for mechanical evaluation."""
    pieces = re.split(
        r"(?<=[.!?])\s+|\n+",
        (answer or "").strip(),
    )

    seen: set[str] = set()
    duplicates: list[str] = []

    for piece in pieces:
        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            piece.casefold(),
        ).strip()

        if len(normalized.split()) < 4:
            continue

        if normalized in seen and normalized not in duplicates:
            duplicates.append(normalized)
        else:
            seen.add(normalized)

    return duplicates


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate the JSONL dataset without modifying it."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            case_id = str(item.get("id", "")).strip()
            user_message = item.get("user_message")
            normalized_id = case_id.casefold()

            if not case_id:
                raise ValueError(
                    f"Line {line_number} has no non-empty 'id'."
                )

            if normalized_id in seen_ids:
                raise ValueError(f"Duplicate test ID: {case_id}")

            if not isinstance(user_message, str) or not user_message.strip():
                raise ValueError(
                    f"Case {case_id} has an empty or invalid user_message."
                )

            history = item.get("history", [])

            if not isinstance(history, list):
                raise ValueError(
                    f"Case {case_id}: 'history' must be a JSON list."
                )

            for field in ("pass_criteria", "fail_conditions", "rubric_tags"):
                if field in item and not isinstance(item[field], list):
                    raise ValueError(
                        f"Case {case_id}: '{field}' must be a JSON list."
                    )

            for field in ("required_tools", "forbidden_tools",
                          "expected_guidance_tags"):
                if field in item:
                    value = item[field]
                    if not isinstance(value, list) or not all(
                        isinstance(entry, str) and entry.strip()
                        for entry in value
                    ):
                        raise ValueError(
                            f"Case {case_id}: '{field}' must be a "
                            "list of non-empty strings."
                        )

            if "enable_database_tools" in item and not isinstance(
                item["enable_database_tools"], bool
            ):
                raise ValueError(
                    f"Case {case_id}: 'enable_database_tools' "
                    "must be true or false."
                )

            required_context = item.get("required_context")
            if (
                required_context is not None
                and not isinstance(required_context, dict)
            ):
                raise ValueError(
                    f"Case {case_id}: 'required_context' must "
                    "be an object or null."
                )

            if "max_words" in item:
                max_words = item["max_words"]
                if (
                    not isinstance(max_words, int)
                    or isinstance(max_words, bool)
                    or max_words < 1
                ):
                    raise ValueError(
                        f"Case {case_id}: 'max_words' must be "
                        "a positive integer."
                    )

            seen_ids.add(case_id)
            seen_ids.add(normalized_id)
            cases.append(item)

    if not cases:
        raise ValueError("The evaluation dataset has no cases.")

    return cases


def normalize_history(
    raw_history: list[Any],
) -> list[dict[str, str]]:
    """Convert supported fixture formats to ChatEngine history messages."""
    history: list[dict[str, str]] = []

    for index, item in enumerate(raw_history):
        if not isinstance(item, dict):
            raise ValueError(
                f"History item {index + 1} must be a JSON object."
            )

        if "role" in item and "content" in item:
            role = str(item["role"]).strip().lower()
            content = item["content"]

            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"History item {index + 1} has unsupported role {role!r}. "
                    "Only user and assistant history messages are accepted."
                )

            if not isinstance(content, str):
                raise ValueError(
                    f"History item {index + 1} content must be text."
                )

            history.append({"role": role, "content": content})
            continue

        # Also allow simple turn fixtures: {"user": "...", "assistant": "..."}
        if "user" in item and "assistant" in item:
            user_text = item["user"]
            assistant_text = item["assistant"]

            if not isinstance(user_text, str) or not isinstance(
                assistant_text, str
            ):
                raise ValueError(
                    f"History turn {index + 1} must contain text values."
                )

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
            continue

        raise ValueError(
            f"History item {index + 1} has an unsupported format. "
            "Expected role/content or user/assistant fields."
        )

    return history


def has_unapplied_setup(case: dict[str, Any]) -> bool:
    """Identify fixture instructions that this runner deliberately won't run."""
    setup = case.get("test_setup")

    if setup is None or setup is False:
        return False

    if isinstance(setup, str):
        return setup.strip().casefold() not in {
            "",
            "none",
            "null",
            "n/a",
            "na",
            "no setup",
            "not required",
        }

    if isinstance(setup, (list, dict)):
        return bool(setup)

    return bool(setup)


def make_read_only_learning_store():
    """Build the real LearningStore interface without running its initializer."""
    from app.learning_ai.embedding_client import EmbeddingClient
    from app.learning_ai.learning_store import LearningStore

    class ReadOnlyLearningStore(LearningStore):
        def __init__(self) -> None:
            # Deliberately bypass LearningStore.__init__:
            # it creates database tables and seeds built-in guidance.
            self.embedding_client = EmbeddingClient()

        def _connect(self):
            connection = super()._connect()

            try:
                # This must be the first SQL statement in this transaction.
                connection.execute("SET TRANSACTION READ ONLY")
            except Exception:
                connection.close()
                raise

            return connection

    return ReadOnlyLearningStore()


def find_page_mentions(answer: str) -> list[int]:
    """Return page numbers explicitly labelled in English or Hindi."""
    values: list[int] = []

    for match in PAGE_PATTERN.finditer(answer):
        value = next(
            (group for group in match.groups() if group is not None),
            None,
        )

        if value is not None:
            values.append(int(value))

    return values


def is_simple_page_answer(answer: str, expected_page: int) -> bool:
    """Recognize only a simple, objectively checkable page-number reply."""
    text = re.sub(r"\s+", " ", answer.strip().casefold())
    page = re.escape(str(expected_page))

    patterns = (
        rf"{page}[.!]?",
        rf"page\s*[:#-]?\s*{page}[.!]?",
        rf"pdf\s+page\s*[:#-]?\s*{page}[.!]?",
        rf"physical\s+page\s*[:#-]?\s*{page}[.!]?",
        rf"source\s*:\s*page\s*[:#-]?\s*{page}[.!]?",
        rf"पेज\s*(?:नंबर\s*)?[:#-]?\s*{page}[.!]?",
        rf"पृष्ठ\s*(?:संख्या\s*)?[:#-]?\s*{page}[.!]?",
    )

    return any(re.fullmatch(pattern, text) for pattern in patterns)


def assess_answer(
    case: dict[str, Any],
    answer: str,
    *,
    profile_entries: list[dict[str, Any]] | None = None,
    stream_metrics: dict[str, Any] | None = None,
    database_tool_trace: list[dict[str, Any]] | None = None,
    retrieval_manifest: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Apply conservative mechanical checks, not a pretend AI quality judge."""
    checks: list[dict[str, str]] = []
    failures: list[str] = []
    review_reasons: list[str] = []

    mode = str(case.get("language_mode") or "hinglish").strip().lower()
    user_message = str(case["user_message"])
    category = str(case.get("category", ""))
    required_context = case.get("required_context")

    if not answer.strip():
        failures.append("The model returned an empty answer.")
        checks.append(
            {"check": "non_empty_output", "result": "fail"}
        )
    else:
        checks.append(
            {"check": "non_empty_output", "result": "pass"}
        )

    # Roman Hinglish cases should not silently switch to Devanagari.
    if mode == "hinglish" and DEVANAGARI_PATTERN.search(answer):
        failures.append(
            "Devanagari characters appeared in a Roman Hinglish response."
        )
        checks.append(
            {"check": "hinglish_script", "result": "fail"}
        )
    elif mode == "hinglish":
        checks.append(
            {"check": "hinglish_script", "result": "pass"}
        )

    rubric_tags = {
        str(tag)
        for tag in case.get("rubric_tags", [])
    }

    if "no_unnecessary_question" in rubric_tags:
        has_unwanted_question = bool(
            re.search(r"\?\s*[\W_]*$", answer)  
            or STOCK_FOLLOW_UP_PATTERN.search(answer)
        )

        checks.append({
            "check": "no_unnecessary_question",
            "result": "fail" if has_unwanted_question else "pass",
        })

        if has_unwanted_question:
            failures.append(
                "This case should not end with an unnecessary "
                "follow-up question or a stock help offer."
            )

    if "no_repeated_sentence" in rubric_tags:
        repeated = find_repeated_sentences(answer)

        checks.append({
            "check": "no_repeated_sentence",
            "result": "fail" if repeated else "pass",
        })

        if repeated:
            failures.append(
                "The answer repeats a sentence: "
                + "; ".join(repeated)
            )

    required_tools = {
        str(name).strip().casefold()
        for name in case.get("required_tools", [])
        if str(name).strip()
    }
    forbidden_tools = {
        str(name).strip().casefold()
        for name in case.get("forbidden_tools", [])
        if str(name).strip()
    }

    if required_tools or forbidden_tools:
        observed_tools = _database_tool_names(database_tool_trace)
        missing_tools = sorted(required_tools - observed_tools)
        unexpected_tools = sorted(forbidden_tools & observed_tools)

        passed = not missing_tools and not unexpected_tools

        checks.append({
            "check": "database_tool_routing",
            "result": "pass" if passed else "fail",
        })

        if missing_tools:
            failures.append(
                "Required database tool(s) were not called: "
                + ", ".join(missing_tools)
            )

        if unexpected_tools:
            failures.append(
                "Forbidden database tool(s) were called: "
                + ", ".join(unexpected_tools)
            )

    # Detect answers cut off by Ollama's output-token limit.
    done_reason = str(
        (stream_metrics or {}).get("done_reason") or ""
    ).casefold()

    if done_reason in {"length", "max_tokens", "limit"}:
        checks.append({
            "check": "generation_not_truncated",
            "result": "fail",
        })
        failures.append(
            "Ollama stopped because the output limit was reached."
        )
    elif done_reason in {"stop", "eos", "end_turn"}:
        checks.append({
            "check": "generation_not_truncated",
            "result": "pass",
        })

    # Profile recall must include saved facts, not merely mention a profile.
    if category == "profile.recall" and isinstance(required_context, dict):
        available, covered, covered_keys = _measure_profile_grounding(
            required_context,
            profile_entries,
            answer,
        )

        if available == 0:
            checks.append({
                "check": "profile_fact_grounding",
                "result": "review_needed",
            })
            review_reasons.append(
                "Requested saved profile values were unavailable "
                "for the grounding check."
            )
        else:
            configured_minimum = required_context.get(
                "minimum_profile_fields_covered"
            )
            minimum = (
                configured_minimum
                if isinstance(configured_minimum, int)
                and not isinstance(configured_minimum, bool)
                and 1 <= configured_minimum <= available
                else max(1, (available + 1) // 2)
            )

            if covered == 0:
                checks.append({
                    "check": "profile_fact_grounding",
                    "result": "fail",
                })
                failures.append(
                    "The answer did not measurably reflect any requested "
                    "saved profile values."
                )   
            elif covered >= minimum:
                checks.append({
                    "check": "profile_fact_grounding",
                    "result": "pass",
                })
            else:
                checks.append({
                    "check": "profile_fact_grounding",
                    "result": "review_needed",
                })
                review_reasons.append(
                    f"Profile grounding needs review: {covered}/{available} "
                    f"saved fields matched; target is {minimum}."
                )

            # Record only field names, not the user's saved personal values.
            case["_profile_grounding_fields"] = covered_keys

        if _GENERIC_PRAISE_PATTERN.search(answer):
            checks.append({
                "check": "no_generic_praise",
                "result": "fail",
            })
            failures.append(
                "The profile answer contains generic praise instead "
                "of focusing on saved facts."
            )
        else:
            checks.append({
                "check": "no_generic_praise",
                "result": "pass",
            })

    if "no_irrelevant_profile" in rubric_tags:
        leaks = _find_irrelevant_profile_leaks(
            answer,
            profile_entries,
        )
        checks.append({
            "check": "no_irrelevant_profile",
            "result": "fail" if leaks else "pass",
        })

        if leaks:
            failures.append(
                "Unrequested saved profile details appeared in the answer: "
                + ", ".join(leaks)
            )

    # Optional, explicit word budget per evaluation case.
    max_words = case.get("max_words")
    if isinstance(max_words, int) and not isinstance(max_words, bool):
        word_count = len(answer.split())
        exceeded = max_words >= 1 and word_count > max_words

        checks.append({
            "check": "response_length",
            "result": "fail" if exceeded else "pass",
        })

        if exceeded:
            failures.append(
                f"Response contains {word_count} words; "
                f"the case limit is {max_words}."
            )
    
    expected_page: int | None = None
    context = (
        required_context
        if isinstance(required_context, dict)
        else {}
    )

    raw_expected_page = context.get("physical_pdf_page")
    if (
        isinstance(raw_expected_page, int)
        and not isinstance(raw_expected_page, bool)
        and raw_expected_page > 0
    ):
        expected_page = raw_expected_page

    page_mentions = find_page_mentions(answer)

    if (
        expected_page is not None
        and is_simple_page_answer(answer, expected_page)
    ):  
        page_mentions = [expected_page]

    if category == "rag.source_grounding":
        require_source_page = context.get("require_source_page") is True
        expect_page_unavailable = (
            context.get("expect_page_unavailable") is True
        )

        requested_pages: set[int] = set()
        raw_requested_pages = context.get("requested_pages", [])

        if isinstance(raw_requested_pages, list):
            for value in raw_requested_pages:
                if (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ):
                    requested_pages.add(value)

        retrieved_pages = _retrieved_document_pages(retrieval_manifest)

        requested_file = str(context.get("source_file") or "").strip()

        if requested_file:
            wanted_filename = (
                requested_file.replace("\\", "/")
                .rsplit("/", 1)[-1]
                .casefold()
            )

            retrieved_pages = [
                item for item in retrieved_pages
                if str(
                    item.get("source_file") or ""
                ).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                == wanted_filename
            ]

            # A basename alone cannot prove which of two indexed documents
            # with the same filename supplied the quoted page.
            source_identities: set[tuple[str, str]] = set()

            for item in retrieved_pages:
                document_id = item.get("document_id")
                content_hash = str(item.get("content_sha256") or "")

                if isinstance(document_id, int) and not isinstance(
                    document_id, bool
                ):
                    source_identities.add(
                        ("document_id", str(document_id))
                    )
                elif content_hash:
                    source_identities.add(
                        ("content_sha256", content_hash)
                    )

            if len(source_identities) > 1:
                checks.append({
                    "check": "retrieved_source_unambiguous",
                    "result": "review_needed",
                })
                review_reasons.append(
                    "More than one indexed document matches the requested "
                    "filename. A page citation cannot be grounded to a "
                    "unique source from the filename alone."
                )
                retrieved_pages = []

        supported_pages = {
            item["page_number"]
            for item in retrieved_pages
            if isinstance(item.get("page_number"), int)
        }

        # A fixture may explicitly state that page metadata is unavailable.
        if expect_page_unavailable:
            if page_mentions:
                failures.append(
                    "The answer claims a physical page even though "
                    "this case marks page metadata as unavailable."
                )
                checks.append({
                    "check": "page_metadata_honesty",
                    "result": "fail",
                })
            else:
                checks.append({
                    "check": "page_metadata_honesty",
                    "result": "review_needed",
                })
                review_reasons.append(
                    "Verify that the answer explicitly explains "
                    "that the physical page metadata is unavailable."
                )

        # An exact source citation must be backed by the current turn's
        # retrieved evidence for the requested file.
        elif page_mentions:
            unsupported_pages = sorted(
                set(page_mentions) - supported_pages
            )

            if unsupported_pages:
                failures.append(
                    "The answer cites page(s) missing from the retrieved "
                    "source evidence: "
                    + ", ".join(map(str, unsupported_pages))
                )
                checks.append({
                    "check": "retrieved_source_citation",
                    "result": "fail",
                })
            else:
                checks.append({
                    "check": "retrieved_source_citation",
                    "result": "pass",
                })

        elif require_source_page:
            failures.append(
                "The request requires a source-page citation, "
                "but the answer does not state a page number."
            )
            checks.append({
                "check": "retrieved_source_citation",
                "result": "fail",
            })

        # Check the expected physical page independently from the topic ID.
        if expected_page is not None:
            if expected_page in page_mentions:
                checks.append({
                    "check": "source_page_number",
                    "result": "pass",
                })
            elif require_source_page:
                failures.append(
                    f"Expected physical PDF page {expected_page} "
                    "was not cited."
                )
                checks.append({
                    "check": "source_page_number",
                    "result": "fail",
                })
            else:
                checks.append({
                    "check": "source_page_number",
                    "result": "review_needed",
                })
                review_reasons.append(
                    f"Verify whether physical PDF page {expected_page} "
                    "should be cited in this answer."
                )

        # Requested page availability is a diagnostic, not proof that the
        # requested page exists or that its contents support the answer.
        if requested_pages:
            missing_requested_pages = sorted(
                requested_pages - supported_pages
            )

            if missing_requested_pages:
                checks.append({
                    "check": "requested_pages_retrieved",
                    "result": "review_needed",
                })
                review_reasons.append(
                    "Requested source page(s) were not verified in the "
                    "current retrieval: "
                    + ", ".join(map(str, missing_requested_pages))
                    + ". Check whether the answer clearly says which pages "
                    "could not be verified."
                )
            else:
                checks.append({
                    "check": "requested_pages_retrieved",
                    "result": "pass",
                })

    asks_page_only = bool(
        re.search(
            r"\bsirf\s+page\s+number\b",
            user_message,
            flags=re.IGNORECASE,
        )
    )

    if asks_page_only:
        word_count = len(answer.split())

        if word_count > 8:
            failures.append(
                "The test requests only the page number, "
                f"but the answer contains {word_count} words."
            )
            checks.append(
                {"check": "page_answer_conciseness", "result": "fail"}
            )
        else:
            checks.append(
                {"check": "page_answer_conciseness", "result": "pass"}
            )

    if failures:
        return "fail", checks, failures

    # PASS is intentionally limited to this fully mechanical test pattern.
    # Other qualitative criteria still require a human reviewer.
    if (
        not has_unapplied_setup(case)
        and category == "rag.source_grounding"
        and asks_page_only
        and expected_page is not None
        and is_simple_page_answer(answer, expected_page)
        and (
            mode != "hinglish"
            or not DEVANAGARI_PATTERN.search(answer)
        )
    ):
        return "pass", checks, []

    return (
        "review_needed",
        checks,
        review_reasons or [
            "Qualitative criteria still require human review."
        ],
    )

def _database_tool_names(
    trace: list[dict[str, Any]] | None,
) -> set[str]:
    names: set[str] = set()

    for item in trace or []:
        if not isinstance(item, dict):
            continue

        name = (
            item.get("tool_name")
            or item.get("name")
            or item.get("tool")
        )

        if not name and isinstance(item.get("tool_call"), dict):
            name = item["tool_call"].get("name")

        if isinstance(name, str) and name.strip():
            names.add(name.strip().casefold())

    return names


def _sanitize_database_tool_trace(
    trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Keep tool diagnostics without copying database record contents."""
    sanitized: list[dict[str, Any]] = []

    for item in trace or []:
        if not isinstance(item, dict):
            continue

        name = (
            item.get("tool_name")
            or item.get("name")
            or item.get("tool")
        )

        if not name and isinstance(item.get("tool_call"), dict):
            name = item["tool_call"].get("name")

        arguments = item.get("arguments", item.get("args", {}))
        argument_keys = (
            sorted(str(key) for key in arguments)
            if isinstance(arguments, dict)
            else []
        )

        status = item.get("status")
        if not isinstance(status, str):
            if item.get("error"):
                status = "error"
            elif "result" in item or "output" in item:
                status = "completed"
            else:
                status = "unknown"

        sanitized.append({
            "tool_name": str(name or "unknown"),
            "status": status,
            "argument_keys": argument_keys,
        })

    return sanitized


def _retrieved_document_pages(
    manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract traceable source metadata without storing document text."""
    context = manifest if isinstance(manifest, dict) else {}
    selected = context.get("selected", [])

    if not isinstance(selected, list):
        return []

    def optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def safe_strings(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, (set, frozenset)):
            value = sorted(value)

        if not isinstance(value, (list, tuple)):
            return []

        return list(dict.fromkeys(
            item.strip().casefold()
            for item in value
            if isinstance(item, str) and item.strip()
        ))[:30]

    evidence: list[dict[str, Any]] = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        if item.get("source") != "document_chunk":
            continue

        source_id = item.get("source_id")
        if isinstance(source_id, bool) or not isinstance(
            source_id, (int, str)
        ):
            source_id = None

        evidence.append({
            "source_id": source_id,
            "document_id": optional_int(item.get("document_id")),
            "chunk_index": optional_int(item.get("chunk_index")),
            "content_sha256": str(
                item.get("content_sha256") or ""
            )[:64],
            "source_file": str(
                item.get("source_file") or item.get("title") or ""
            )[:200],
            "page_number": optional_int(item.get("page_number")),
            "section": str(item.get("section") or "")[:200],
            "topic_numbers": safe_strings(
                item.get("topic_numbers", [])
            ),
            "rule_id": str(item.get("rule_id") or "")[:120],
            "rule_tags": safe_strings(item.get("rule_tags", [])),
            "match_reason": str(
                item.get("match_reason") or item.get("reason") or ""
            )[:120],
        })

    return evidence


def _sanitize_retrieval_manifest(
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep traceable retrieval metadata, never retrieved content."""
    if not isinstance(manifest, dict):
        return {}

    raw_selected = manifest.get("selected", [])
    if not isinstance(raw_selected, list):
        raw_selected = []

    safe_selected: list[dict[str, Any]] = []

    identifier_fields = (
        "source",
        "source_id",
        "id",
        "memory_key",
        "key",
        "title",
        "source_file",
        "category",
        "content_kind",
        "section",
        "rule_id",
        "guidance_id",
        "source_key",
        "document_id",
        "chunk_index",
        "content_sha256",
    )

    for item in raw_selected:
        if not isinstance(item, dict):
            continue

        safe_item: dict[str, Any] = {}

        for key in identifier_fields:
            value = item.get(key)

            if isinstance(value, str):
                safe_item[key] = value[:200]
            elif isinstance(value, int) and not isinstance(value, bool):
                safe_item[key] = value

        page = item.get("page_number")
        if isinstance(page, int) and not isinstance(page, bool):
            safe_item["page_number"] = page

        for key in ("score", "similarity"):
            value = item.get(key)
            if _is_finite_number(value):
                safe_item[key] = round(float(value), 5)

        safe_item["reason"] = str(
            item.get("reason") or item.get("match_reason") or ""
        )[:120]

        safe_item["match_reason"] = str(
            item.get("match_reason") or item.get("reason") or ""
        )[:120]

        raw_tags = item.get("rule_tags", item.get("tags", []))

        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        elif isinstance(raw_tags, (set, frozenset)):
            raw_tags = sorted(raw_tags)

        if isinstance(raw_tags, (list, tuple)):
            safe_item["rule_tags"] = list(dict.fromkeys(
                tag.strip().casefold()
                for tag in raw_tags
                if isinstance(tag, str) and tag.strip()
            ))[:30]

            for field in ("topic_numbers", "rule_tags"):
                raw_values = item.get(field, [])

                if isinstance(raw_values, str):
                    raw_values = [raw_values]
                elif isinstance(raw_values, (set, frozenset)):
                    raw_values = sorted(raw_values)

                if isinstance(raw_values, (list, tuple)):
                    safe_item[field] = list(dict.fromkeys(
                        value.strip().casefold()
                        for value in raw_values
                        if isinstance(value, str) and value.strip()
                    ))[:30]

        safe_selected.append(safe_item)

    def safe_count(value: Any) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool):
            return value if value >= 0 else None
        return None

    result: dict[str, Any] = {
        "status": str(manifest.get("status") or "unknown")[:80],
        "reason": str(manifest.get("reason") or "")[:160],
        "candidate_count": safe_count(
            manifest.get("candidate_count")
        ),
        "selected_count": safe_count(
            manifest.get("selected_count")
        ),
        "selected": safe_selected,
    }

    # Preserve useful diagnostics without persisting document content.
    thresholds = manifest.get("thresholds")
    if isinstance(thresholds, dict):
        result["thresholds"] = {
            str(key)[:80]: round(float(value), 5)
            for key, value in thresholds.items()
            if _is_finite_number(value)
        }

    for field in ("rejected", "selected_by_source"):
        value = manifest.get(field)

        if isinstance(value, dict):
            result[field] = {
                str(key)[:100]: count
                for key, raw_count in value.items()
                if (count := safe_count(raw_count)) is not None
            }

    prompt_context = manifest.get("prompt_context")

    if isinstance(prompt_context, dict):
        safe_prompt_context: dict[str, Any] = {}

        for field in (
            "behavior_guidance_items",
            "playbook_guidance_items",
            "behavior_guide_items",
            "feedback_lesson_items",
            "document_context_items",
        ):
            count = safe_count(prompt_context.get(field))
            if count is not None:
                safe_prompt_context[field] = count

        rule_ids = prompt_context.get("included_rule_ids", [])

        if isinstance(rule_ids, (list, tuple)):
            safe_prompt_context["included_rule_ids"] = list(dict.fromkeys(
                item.strip().casefold()
                for item in rule_ids
                if isinstance(item, str) and item.strip()
            ))[:100]

        result["prompt_context"] = safe_prompt_context

    explicit_reference = manifest.get(
        "explicit_source_page_reference"
    )

    if isinstance(explicit_reference, dict):
        safe_reference: dict[str, Any] = {
            "status": str(
                explicit_reference.get("status") or "unknown"
            )[:80],
            "file_reference": str(
                explicit_reference.get("file_reference") or ""
            )[:200],
        }

        for field in (
            "requested_pages",
            "matched_pages",
            "missing_pages",
        ):
            values = explicit_reference.get(field, [])

            if isinstance(values, (list, tuple)):
                safe_reference[field] = [
                    value
                    for value in values
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and value > 0
                ][:100]

        result["explicit_source_page_reference"] = safe_reference

    if result["selected_count"] is None:
        result["selected_count"] = len(safe_selected)

    return result

def diagnose_retrieval_stage(
    case: dict[str, Any],
    manifest: dict[str, Any] | None,
    verdict: str,
) -> dict[str, Any]:
    """Diagnose retrieval from source IDs, rule IDs and prompt delivery."""
    context = manifest if isinstance(manifest, dict) else {}

    selected = context.get("selected", [])
    if not isinstance(selected, list):
        selected = []

    status = str(context.get("status") or "unknown").casefold()
    reason = str(context.get("reason") or "").casefold()

    required = case.get("required_context")
    required = required if isinstance(required, dict) else {}

    raw_expected_tags = case.get("expected_guidance_tags", [])
    if not isinstance(raw_expected_tags, (list, tuple, set)):
        raw_expected_tags = []

    expected_tags = {
        tag.strip().casefold()
        for tag in raw_expected_tags
        if isinstance(tag, str) and tag.strip()
    }

    guidance_items: list[dict[str, Any]] = []
    document_items: list[dict[str, Any]] = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "").casefold()
        filename = str(
            item.get("source_file") or item.get("title") or ""
        ).replace("\\", "/").rsplit("/", 1)[-1].casefold()

        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        content_kind = str(
            item.get("content_kind")
            or metadata.get("content_kind")
            or ""
        ).casefold()

        is_playbook = (
            source == "document_chunk"
            and (
                filename == "hinglish_playbook.md"
                or content_kind == "behavioral_guidance"
            )
        )

        if source in {"behavior_guide", "feedback_lesson"} or is_playbook:
            guidance_items.append(item)

        if source == "document_chunk":
            document_items.append(item)

    def string_values(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        elif isinstance(value, (set, frozenset)):
            value = sorted(value)

        if not isinstance(value, (list, tuple)):
            return []

        return [
            part.strip().casefold()
            for part in value
            if isinstance(part, str) and part.strip()
        ]

    def extract_rule_ids(
        item: dict[str, Any],
    ) -> set[str]:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        ids: set[str] = set()

        for key in ("rule_id", "rule_tags", "tags"):
            ids.update(string_values(item.get(key)))
            ids.update(string_values(metadata.get(key)))

        for key in ("source_key", "guide_key", "guidance_id"):
            for value in (
                item.get(key),
                metadata.get(key),
            ):
                if isinstance(value, str):
                    ids.add(value.strip().casefold())

        # Some older ingested Playbook chunks only have a section heading.
        section = str(item.get("section") or "")
        match = re.search(
            r"\bquick behaviour card\s*[—–-]\s*"
            r"`([a-z][a-z0-9_.-]*)`",
            section,
            re.IGNORECASE,
        )

        if match:
            ids.add(match.group(1).casefold())

        return {
            value for value in ids
            if re.fullmatch(r"[a-z][a-z0-9_.-]{0,120}", value)
        }

    selected_rule_ids: set[str] = set()

    for item in guidance_items:
        selected_rule_ids.update(extract_rule_ids(item))

    prompt_context = context.get("prompt_context")
    prompt_context = (
        prompt_context if isinstance(prompt_context, dict) else None
    )

    raw_delivered_ids = (
        prompt_context.get("included_rule_ids", [])
        if prompt_context is not None
        else []
    )
    delivered_rule_ids = set(string_values(raw_delivered_ids))

    base = {
        "case_verdict": verdict,
        "retrieval_status": status,
        "retrieval_reason": reason,
        "expected_guidance_tags": sorted(expected_tags),
        "selected_guidance_count": len(guidance_items),
        "selected_document_count": len(document_items),
        "selected_rule_ids": sorted(selected_rule_ids),
        "prompt_delivery_traced": prompt_context is not None,
    }

    # if status == "skipped" or reason.startswith("skipped"):
    #     return {
    #         **base,
    #         "classification": "retrieval_skipped",
    #     }

    if status == "skipped" or reason.startswith("skipped"):
        if expected_tags:
            return {    
                **base,
                "classification": (
                    "expected_guidance_not_retrieved_due_to_policy_skip"
                ),
                "failure_stage": "retrieval_policy",
                "policy_skip_reason": reason or status,
                "identified_guidance_tags": [],
            }

        return {
            **base,
            "classification": "retrieval_skipped",
        }

    if status in {"unavailable", "error"}:
        return {
            **base,
            "classification": "retrieval_unavailable",
        }

    if status in {"not_run", "unknown"}:
        return {
            **base,
            "classification": "retrieval_not_run",
        }

    # First diagnose explicit Playbook-rule expectations by exact ID.
    if expected_tags:
        found_tags = expected_tags & selected_rule_ids

        if found_tags:
            missing_from_prompt = (
                found_tags - delivered_rule_ids
            )

            if prompt_context is None:
                classification = (
                    "expected_guidance_retrieved_delivery_unverified"
                )
            elif missing_from_prompt:
                classification = (
                    "expected_guidance_selected_but_not_in_prompt"
                )
            elif verdict in {"fail", "error"}:
                classification = (
                    "expected_guidance_in_prompt_answer_failed"
                )
            elif verdict == "review_needed":
                classification = (
                    "expected_guidance_in_prompt_answer_needs_review"
                )
            else:
                classification = "expected_guidance_in_prompt"

            return {
                **base,
                "classification": classification,
                "identified_guidance_tags": sorted(found_tags),
                "missing_from_prompt": sorted(missing_from_prompt),
                "prompt_included_rule_ids": sorted(delivered_rule_ids),
            }

        if status == "keyword_fallback":
            classification = "retrieval_fallback_inconclusive"
        elif not guidance_items:
            classification = "possible_retrieval_miss"
        elif not selected_rule_ids:
            classification = "guidance_rule_metadata_missing"
        else:
            classification = "expected_guidance_tag_not_selected"

        return {
            **base,
            "classification": classification,
            "identified_guidance_tags": [],
        }

    # Then diagnose expected document/source/page/topic evidence.
    explicit = context.get("explicit_source_page_reference")
    explicit = explicit if isinstance(explicit, dict) else {}

    explicit_status = str(
        explicit.get("status") or ""
    ).casefold()

    if explicit_status == "unresolved":
        return {
            **base,
            "classification": "explicit_source_page_not_retrieved",
            "explicit_source_page_reference": explicit,
        }

    if explicit_status == "partially_matched":
        return {
            **base,
            "classification": "explicit_source_pages_partially_retrieved",
            "explicit_source_page_reference": explicit,
        }

    expected_file = str(
        required.get("source_file")
        or explicit.get("file_reference")
        or ""
    ).replace("\\", "/").rsplit("/", 1)[-1].casefold()

    expected_section = str(
        required.get("section") or ""
    ).strip().casefold()

    expected_topics: set[str] = set()

    raw_topic_label = required.get("topic_label")
    if isinstance(raw_topic_label, str):
        if re.search(r"\btopic\b", raw_topic_label, re.IGNORECASE):
            expected_topics.update(
                re.findall(r"\d+", raw_topic_label)
            )

    raw_topic_number = required.get("topic_number")
    if (
        isinstance(raw_topic_number, int)
        and not isinstance(raw_topic_number, bool)
        and raw_topic_number > 0
    ):
        expected_topics.add(str(raw_topic_number))

    expected_pages: set[int] = set()

    physical_page = required.get("physical_pdf_page")
    if (
        isinstance(physical_page, int)
        and not isinstance(physical_page, bool)
        and physical_page > 0
    ):
        expected_pages.add(physical_page)

    raw_pages = required.get("requested_pages", [])
    if isinstance(raw_pages, (list, tuple)):
        expected_pages.update(
            value
            for value in raw_pages
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
        )

    if not expected_pages:
        raw_explicit_pages = explicit.get("requested_pages", [])
        if isinstance(raw_explicit_pages, (list, tuple)):
            expected_pages.update(
                value
                for value in raw_explicit_pages
                if isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            )

    has_document_target = bool(
        expected_file
        or expected_section
        or expected_topics
        or expected_pages
    )

    if has_document_target:
        source_items = [
            item for item in document_items
            if (
                not expected_file
                or str(
                    item.get("source_file") or item.get("title") or ""
                ).replace("\\", "/").rsplit("/", 1)[-1].casefold()
                == expected_file
            )
        ]

        if expected_file and not source_items:
            return {
                **base,
                "classification": "expected_document_source_not_retrieved",
                "expected_source_file": expected_file,
                "expected_pages": sorted(expected_pages),
            }

        if not source_items:
            return {
                **base,
                "classification": "expected_document_evidence_not_retrieved",
                "expected_pages": sorted(expected_pages),
            }

        identities: set[tuple[str, str]] = set()

        for item in source_items:
            document_id = item.get("document_id")
            content_hash = str(item.get("content_sha256") or "")

            if isinstance(document_id, int) and not isinstance(
                document_id, bool
            ):
                identities.add(("document_id", str(document_id)))
            elif content_hash:
                identities.add(("content_sha256", content_hash))

        if expected_file and len(identities) > 1:
            return {
                **base,
                "classification": "expected_document_source_ambiguous",
                "expected_source_file": expected_file,
            }

        observed_topics: set[str] = set()
        observed_sections: set[str] = set()
        observed_pages: set[int] = set()

        for item in source_items:
            observed_topics.update(
                string_values(item.get("topic_numbers", []))
            )

            section = str(item.get("section") or "").strip().casefold()
            if section:
                observed_sections.add(section)

            page = item.get("page_number")
            if isinstance(page, int) and not isinstance(page, bool):
                observed_pages.add(page)

        missing_topics = expected_topics - observed_topics
        if missing_topics:
            return {
                **base,
                "classification": "expected_document_topic_not_retrieved",
                "expected_topic_numbers": sorted(expected_topics),
                "identified_topic_numbers": sorted(observed_topics),
                "missing_topic_numbers": sorted(missing_topics),
            }

        if expected_section and not any(
            expected_section in section or section in expected_section
            for section in observed_sections
        ):
            return {
                **base,
                "classification": "expected_document_section_not_retrieved",
                "expected_section": expected_section,
                "identified_sections": sorted(observed_sections),
            }

        missing_pages = expected_pages - observed_pages
        if missing_pages:
            classification = (
                "expected_document_pages_partially_retrieved"
                if observed_pages & expected_pages
                else "expected_document_pages_not_retrieved"
            )

            return {
                **base,
                "classification": classification,
                "expected_pages": sorted(expected_pages),
                "identified_pages": sorted(observed_pages),
                "missing_pages": sorted(missing_pages),
            }

        if verdict in {"fail", "error"}:
            classification = "expected_document_evidence_retrieved_answer_failed"
        elif verdict == "review_needed":
            classification = (
                "expected_document_evidence_retrieved_answer_needs_review"
            )
        else:
            classification = "expected_document_evidence_retrieved"

        return {
            **base,
            "classification": classification,
            "expected_source_file": expected_file or None,
            "expected_pages": sorted(expected_pages),
            "identified_pages": sorted(observed_pages),
            "expected_topic_numbers": sorted(expected_topics),
            "identified_topic_numbers": sorted(observed_topics),
        }

    if status == "keyword_fallback":
        classification = "retrieval_fallback_inconclusive"
    elif status == "no_relevant_matches" or not selected:
        classification = "no_relevant_retrieval_matches"
    else:
        classification = "retrieval_completed_no_expected_target_configured"

    return {
        **base,
        "classification": classification,
    }

def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert Ollama nanosecond metrics into readable seconds."""
    if not metrics:
        return None

    def seconds(key: str) -> float | None:
        value = metrics.get(key)

        if isinstance(value, (int, float)):
            return round(value / 1_000_000_000, 4)

        return None

    generation_seconds = seconds("eval_duration")
    output_tokens = metrics.get("eval_count")
    token_rate = None

    if (
        generation_seconds is not None
        and generation_seconds > 0
        and isinstance(output_tokens, (int, float))
    ):
        token_rate = round(output_tokens / generation_seconds, 3)

    return {
        "model": metrics.get("model"),
        "done_reason": metrics.get("done_reason"),
        "load_seconds": seconds("load_duration"),
        "prompt_eval_seconds": seconds("prompt_eval_duration"),
        "generation_seconds": generation_seconds,
        "server_total_seconds": seconds("total_duration"),
        "input_tokens": metrics.get("prompt_eval_count"),
        "cached_input_tokens": metrics.get("prompt_eval_cached_count"),
        "output_tokens": output_tokens,
        "tokens_per_second": token_rate,
        "client_request_seconds": metrics.get("request_wall_seconds"),
        "model_first_content_seconds": metrics.get("first_content_seconds"),
    }


def run_one_case(
    case: dict[str, Any],
    *,
    settings: Any,
    ChatEngine: Any,
    OllamaClient: Any,
    PromptManager: Any,
    normalize_language_mode: Any,
    repeat_penalty: float | None = None,
) -> dict[str, Any]:
    """Run exactly one case through Zara's existing streamed chat path."""
    case_id = str(case["id"])
    client = OllamaClient()
    store = None
    engine = None
    started = perf_counter()
    first_output_seconds: float | None = None
    answer_parts: list[str] = []

    base_result: dict[str, Any] = {
        "id": case_id,
        "category": case.get("category"),
        "severity": case.get("severity"),
        "language_mode": case.get("language_mode", "hinglish"),
        "user_message": case["user_message"],
        "history": case.get("history", []),
        "required_context": case.get("required_context"),
        "pass_criteria": case.get("pass_criteria", []),
        "fail_conditions": case.get("fail_conditions", []),
        "rubric_tags": case.get("rubric_tags", []),
        "test_setup": case.get("test_setup"),
        "setup_applied": False,
        "answer": "",
        "verdict": "error",
        "checks": [],
        "reasons": [],
        "duration_seconds": None,
        "first_output_seconds": None,
        "retrieval": None,
        "ollama_metrics": None,
        "error": None,
        "response_policy": [],
        "database_tool_trace": [],
        "retrieved_source_evidence": [], 
        "retrieval_diagnosis": None,   
        "conversation_decision": {},
        "answer_repair": {},
    }

    try:
        mode = normalize_language_mode(
            str(case.get("language_mode") or "hinglish")
        )

        if mode is None:
            raise ValueError(
                f"Unsupported language_mode in case {case_id}: "
                f"{case.get('language_mode')!r}"
            )

        history = normalize_history(case.get("history", []))

        store = make_read_only_learning_store()

        engine = ChatEngine(
            client=client,
            prompt_manager=PromptManager(),
            language_mode=mode,
            preferred_address="Boss",
            learning_store=store,
            allow_database_tools=bool(
                case.get("enable_database_tools", False)
            ),
            allow_database_mutations=False,
        )

        if repeat_penalty is not None:
            original_stream_chat = engine.client.stream_chat

            def stream_chat_with_repeat_penalty(messages, options):
                experimental_options = {
                    **options,
                        "repeat_penalty": repeat_penalty,
                    }
                yield from original_stream_chat(
                    messages,
                    experimental_options,
                )       

            engine.client.stream_chat = stream_chat_with_repeat_penalty

        # A separate engine per case prevents case-to-case state leakage.
        engine.history = history[-settings.max_history_messages:]

        for chunk in engine.stream_reply(str(case["user_message"])):
            if chunk:
                if first_output_seconds is None:
                    first_output_seconds = perf_counter() - started

                answer_parts.append(chunk)

        answer = "".join(answer_parts).strip()

        repair_trace = getattr(
            engine,
            "last_answer_repair",
            {},
        )

        if isinstance(repair_trace, dict):
            base_result["answer_repair"] = {
                "status": repair_trace.get("status"),
                "reason_code": repair_trace.get("reason_code"),
                "issue_codes": repair_trace.get("issue_codes", []),
                "confidence": repair_trace.get("confidence"),
                "duration_ms": repair_trace.get("duration_ms"),
            }

        case_tags = set(case.get("rubric_tags", []))
        needs_profile_audit = (
            case.get("category") == "profile.recall"
            or "no_irrelevant_profile" in case_tags
        )

        profile_entries = None
        if needs_profile_audit:
            try:
                profile_entries = store.get_user_profile_entries()
            except Exception:
                profile_entries = None

        stream_metrics = getattr(
            engine.client,
            "last_stream_metrics",
            None,
        )

        database_tool_trace = getattr(
            engine, "last_database_tool_trace", []
        ) or []

        retrieval_manifest = getattr(
            engine, "last_retrieval_manifest", {}
        ) or {}

        verdict, checks, reasons = assess_answer(
            case,
            answer,
            profile_entries=profile_entries,
            stream_metrics=stream_metrics,
            database_tool_trace=database_tool_trace,
            retrieval_manifest=retrieval_manifest,
        )

        if has_unapplied_setup(case) and verdict != "fail":
            reasons.append(
                "This case has test_setup metadata. The runner deliberately "
                "did not execute fixture setup because it may mutate data."
            )
            verdict = "review_needed"

        retrieval_diagnosis = diagnose_retrieval_stage(
            case,
            retrieval_manifest,
            verdict,
        )

        base_result.update(
            {
                "answer": answer,
                "verdict": verdict,
                "checks": checks,
                "reasons": reasons,
                "retrieval_diagnosis": retrieval_diagnosis,
            }
        )

    except Exception as exc:
        base_result["verdict"] = "error"
        base_result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        base_result["duration_seconds"] = round(
            perf_counter() - started, 4
        )
        base_result["first_output_seconds"] = (
            round(first_output_seconds, 4)
            if first_output_seconds is not None
            else None
        )

        if engine is not None:
            base_result["retrieval"] = {
                "seconds": round(engine.last_retrieval_seconds, 4),
                "mode": engine.last_retrieval_mode,
                "item_count": engine.last_retrieved_guidance_count,
                "manifest": _sanitize_retrieval_manifest(
                    getattr(engine, "last_retrieval_manifest", {})
                ),
            }

            base_result["context_manifest"] = getattr(
                engine,
                "last_context_manifest",
                {},
            )

            base_result["turn_metrics"] = getattr(
                engine,
                "last_turn_metrics",
                {},
            )

            base_result["ollama_metrics"] = normalize_metrics(
                engine.client.last_stream_metrics
            )

            base_result["response_policy"] = list(
                getattr(engine, "last_response_policy", ())
            )

            base_result["conversation_decision"] = dict(
                getattr(engine, "last_conversation_decision", {}) or {}
            ) if engine is not None else {}

        if engine is not None:
            base_result["database_tool_trace"] = (
                _sanitize_database_tool_trace(
                    getattr(engine, "last_database_tool_trace", [])
                )
            )

            base_result["retrieved_source_evidence"] = (
                _retrieved_document_pages(
                    getattr(engine, "last_retrieval_manifest", {})
                )
            )

        try:
            client.session.close()
        except Exception:
            pass

        if store is not None:
            try:
                store.embedding_client.session.close()
            except Exception:
                pass

    return base_result

def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _distribution(values: list[Any]) -> dict[str, float | None]:
    """Summarize numeric measurements without hiding missing samples."""
    numbers = sorted(
        float(value)
        for value in values
        if _is_finite_number(value)
    )

    if not numbers:
        return {
            "count": 0,
            "average": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    p95_index = max(0, math.ceil(0.95 * len(numbers)) - 1)

    return {
        "count": len(numbers),
        "average": round(sum(numbers) / len(numbers), 4),
        "median": round(median(numbers), 4),
        "p95": round(numbers[p95_index], 4),
        "min": round(numbers[0], 4),
        "max": round(numbers[-1], 4),
    }


def _sha256_file(path: Path) -> str | None:
    """Fingerprint a source file without storing its contents."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _nested_value(item: dict[str, Any], *path: str) -> Any:
    current: Any = item

    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current

def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a measurable report; never pretend review_needed is a pass."""
    verdict_counts = Counter(
        str(item.get("verdict", "error"))
        for item in results
    )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checks_by_name: dict[str, Counter[str]] = defaultdict(Counter)
    retrieval_modes: Counter[str] = Counter()
    retrieval_diagnoses: Counter[str] = Counter()
    repair_statuses: Counter[str] = Counter()
    repair_issues: Counter[str] = Counter()
    repair_durations: list[float] = []

    for item in results:
        diagnosis = item.get("retrieval_diagnosis") or {}   
        classification = diagnosis.get("classification")

        if isinstance(classification, str):
            retrieval_diagnoses[classification] += 1

        category = str(item.get("category") or "uncategorized")
        by_category[category].append(item)

        for check in item.get("checks", []):
            if not isinstance(check, dict):
                continue

            name = str(check.get("check") or "unknown")
            result = str(check.get("result") or "unknown")
            checks_by_name[name][result] += 1

        retrieval = item.get("retrieval") or {}
        retrieval_modes[str(retrieval.get("mode") or "unknown")] += 1

        repair = item.get("answer_repair") or {}

        if isinstance(repair, dict):

            status = repair.get("status")

            if isinstance(status, str):
                repair_statuses[status] += 1

            issue_codes = repair.get("issue_codes", [])

            if isinstance(issue_codes, list):
                repair_issues.update(
                    code
                    for code in issue_codes[:5]
                    if isinstance(code, str)
                )

            duration = repair.get("duration_ms")

            if _is_finite_number(duration):
                repair_durations.append(float(duration))

    category_summary: dict[str, Any] = {}

    for category, cases in sorted(by_category.items()):
        counts = Counter(
            str(case.get("verdict", "error"))
            for case in cases
        )
        total = len(cases)
        hard_failures = counts.get("fail", 0) + counts.get("error", 0)

        category_summary[category] = {
            "total": total,
            "verdicts": dict(sorted(counts.items())),
            "hard_failures": hard_failures,
            "hard_failure_rate": round(hard_failures / total, 4),
        }

    durations = [item.get("duration_seconds") for item in results]
    first_output = [
        item.get("first_output_seconds") for item in results
    ]
    retrieval_seconds = [
        _nested_value(item, "retrieval", "seconds")
        for item in results
    ]
    prompt_eval_seconds = [
        _nested_value(item, "ollama_metrics", "prompt_eval_seconds")
        for item in results
    ]
    generation_seconds = [
        _nested_value(item, "ollama_metrics", "generation_seconds")
        for item in results
    ]
    model_load_seconds = [
        _nested_value(item, "ollama_metrics", "load_seconds")
        for item in results
    ]
    input_tokens = [
        _nested_value(item, "ollama_metrics", "input_tokens")
        for item in results
    ]
    output_tokens = [
        _nested_value(item, "ollama_metrics", "output_tokens")
        for item in results
    ]
    tokens_per_second = [
        _nested_value(item, "ollama_metrics", "tokens_per_second")
        for item in results
    ]
    prompt_chars = [
        _nested_value(
            item,
            "context_manifest",
            "source_chars",
            "all_messages",
        )
        for item in results
    ]

    diagnostic_signals = {
        "roman_script_violations": sum(
            str(item.get("language_mode", "hinglish")).lower()
            in {"hinglish", "hinglish_roman", "roman_hinglish"}
            and bool(DEVANAGARI_PATTERN.search(str(item.get("answer", ""))))
            for item in results
        ),
        "stock_follow_up_signals": sum(
            bool(STOCK_FOLLOW_UP_PATTERN.search(str(item.get("answer", ""))))
            for item in results
        ),
        "repeated_sentence_cases": sum(
            bool(find_repeated_sentences(str(item.get("answer", ""))))
            for item in results
        ),
        "answers_ending_with_question_mark": sum(
            str(item.get("answer", "")).rstrip().endswith("?")
            for item in results
        ),
        "empty_answers": sum(
            not str(item.get("answer", "")).strip()
            for item in results
        ),
    }

    retrieval_time_values = [
        float(value)
        for value in retrieval_seconds
        if _is_finite_number(value)
    ]

    return {
        "total": len(results),
        "verdict_counts": {
            "pass": verdict_counts.get("pass", 0),
            "fail": verdict_counts.get("fail", 0),
            "review_needed": verdict_counts.get("review_needed", 0),
            "error": verdict_counts.get("error", 0),
        },
        "hard_failure_rate": (
            round(
                (
                    verdict_counts.get("fail", 0)
                    + verdict_counts.get("error", 0)
                ) / len(results),
                4,
            )
            if results else None
        ),
        "by_category": category_summary,
        "checks": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(checks_by_name.items())
        },
        "diagnostic_signals": diagnostic_signals,
        "latency_seconds": {
            "total": _distribution(durations),
            "first_output": _distribution(first_output),
            "retrieval": _distribution(retrieval_time_values),
            "model_load": _distribution(model_load_seconds),
            "prompt_evaluation": _distribution(prompt_eval_seconds),
            "generation": _distribution(generation_seconds),
        },
        "model_usage": {
            "input_tokens": _distribution(input_tokens),
            "output_tokens": _distribution(output_tokens),
            "tokens_per_second": _distribution(tokens_per_second),
            "prompt_chars": _distribution(prompt_chars),
        },
        "retrieval_mode_counts": dict(sorted(retrieval_modes.items())),
        "retrieval_diagnosis_counts": dict(
            sorted(retrieval_diagnoses.items())
        ),
        "answer_repair": {
            "status_counts": dict(sorted(repair_statuses.items())),
            "issue_code_counts": dict(sorted(repair_issues.items())),
            "repair_duration_ms": _distribution(repair_durations),
            "repaired_count": repair_statuses.get("repaired", 0),
            "rejected_count": repair_statuses.get("rejected", 0),
            "unchanged_count": repair_statuses.get("unchanged", 0),
            "error_count": repair_statuses.get("error", 0),
        },
    }


def compare_with_baseline(
    current: list[dict[str, Any]],
    baseline_path: Path,
    baseline_data: dict[str, Any],
    *,
    current_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare per-case quality outcomes, latency, usage and source versions."""
    baseline_cases = baseline_data.get("cases", [])

    if not isinstance(baseline_cases, list):
        raise ValueError("Baseline JSON must contain a 'cases' list.")

    baseline_by_id = {
        str(item.get("id")): item
        for item in baseline_cases
        if isinstance(item, dict) and item.get("id")
    }

    verdict_rank = {
        "pass": 0,
        "review_needed": 1,
        "fail": 2,
        "error": 3,
    }

    metric_paths = {
        "total_seconds": ("duration_seconds",),
        "first_output_seconds": ("first_output_seconds",),
        "input_tokens": ("ollama_metrics", "input_tokens"),
        "output_tokens": ("ollama_metrics", "output_tokens"),
        "prompt_eval_seconds": (
            "ollama_metrics",
            "prompt_eval_seconds",
        ),
        "generation_seconds": (
            "ollama_metrics",
            "generation_seconds",
        ),
        "prompt_chars": (
            "context_manifest",
            "source_chars",
            "all_messages",
        ),
    }

    per_case: list[dict[str, Any]] = []
    changed_answers = 0
    changed_verdicts = 0
    regressions: list[str] = []
    hard_regressions: list[str] = []
    deltas_by_metric: dict[str, list[float]] = {
        metric: [] for metric in metric_paths
    }

    for item in current:
        case_id = str(item["id"])
        previous = baseline_by_id.get(case_id)

        if previous is None:
            per_case.append({
                "id": case_id,
                "comparison": "missing_from_baseline",
            })
            continue

        answer_changed = item.get("answer") != previous.get("answer")
        verdict_before = str(previous.get("verdict", "error"))
        verdict_after = str(item.get("verdict", "error"))
        verdict_changed = verdict_before != verdict_after

        if answer_changed:
            changed_answers += 1

        if verdict_changed:
            changed_verdicts += 1

        regressed = (
            verdict_rank.get(verdict_after, 3)
            > verdict_rank.get(verdict_before, 3)
        )
        hard_regression = (
            verdict_after in {"fail", "error"}
            and verdict_before not in {"fail", "error"}
        )

        if regressed:
            regressions.append(case_id)
        if hard_regression:
            hard_regressions.append(case_id)

        metrics: dict[str, Any] = {}

        for metric_name, path in metric_paths.items():
            previous_value = _nested_value(previous, *path)
            current_value = _nested_value(item, *path)

            delta = None
            if (
                _is_finite_number(previous_value)
                and _is_finite_number(current_value)
            ):
                delta = round(
                    float(current_value) - float(previous_value),
                    4,
                )
                deltas_by_metric[metric_name].append(delta)

            metrics[metric_name] = {
                "before": previous_value,
                "after": current_value,
                "delta": delta,
            }

        old_retrieval_mode = _nested_value(
            previous, "retrieval", "mode"
        )
        new_retrieval_mode = _nested_value(
            item, "retrieval", "mode"
        )

        per_case.append({
            "id": case_id,
            "comparison": "matched",
            "answer_changed": answer_changed,
            "verdict_before": verdict_before,
            "verdict_after": verdict_after,
            "verdict_changed": verdict_changed,
            "regressed": regressed,
            "hard_regression": hard_regression,
            "metrics": metrics,
            "retrieval_mode_before": old_retrieval_mode,
            "retrieval_mode_after": new_retrieval_mode,
            "retrieval_mode_changed": (
                old_retrieval_mode != new_retrieval_mode
            ),
        })

    matched = sum(
        item.get("comparison") == "matched"
        for item in per_case
    )

    baseline_provenance = baseline_data.get("provenance") or {}
    provenance_comparison: dict[str, Any] = {}

    if current_provenance is not None:
        keys = sorted(
            set(baseline_provenance) | set(current_provenance)
        )

        for key in keys:
            before = baseline_provenance.get(key)
            after = current_provenance.get(key)

            provenance_comparison[key] = {
                "before": before,
                "after": after,
                "changed": before != after,
                "comparable": before is not None and after is not None,
            }

    return {
        "baseline_file": str(baseline_path),
        "baseline_case_count": len(baseline_by_id),
        "current_case_count": len(current),
        "matched_case_count": matched,
        "missing_from_baseline": [
            row["id"]
            for row in per_case
            if row.get("comparison") == "missing_from_baseline"
        ],
        "changed_answer_count": changed_answers,
        "changed_verdict_count": changed_verdicts,
        "verdict_regression_count": len(regressions),
        "verdict_regression_ids": regressions,
        "hard_regression_count": len(hard_regressions),
        "hard_regression_ids": hard_regressions,
        "average_metric_deltas": {
            name: round(sum(values) / len(values), 4)
            if values else None
            for name, values in deltas_by_metric.items()
        },
        "provenance": provenance_comparison,
        "per_case": per_case,
    }


def print_dry_run(cases: list[dict[str, Any]], dataset: Path) -> None:
    """Validate and describe the dataset without contacting Ollama or DB."""
    print("Evaluation runner dry run")
    print(f"Dataset: {dataset}")
    print(f"Cases loaded: {len(cases)}")

    categories = Counter(
        str(case.get("category") or "uncategorized")
        for case in cases
    )
    print("\nCategories:")

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    setup_ids = [
        str(case["id"])
        for case in cases
        if has_unapplied_setup(case)
    ]

    print(f"\nCases with test_setup metadata: {len(setup_ids)}")

    if setup_ids:
        print("  " + ", ".join(setup_ids))

    print("\nPASS: JSONL parsing and basic schema validation completed.")
    print("No Ollama request or database connection was made.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run isolated Zara V2 evaluation cases."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to a timestamped file.",
    )
    parser.add_argument(
        "--compare",
        help="Optional earlier evaluation JSON file to compare against.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected cases.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Override repeat_penalty for evaluation requests only.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Run only these case IDs, e.g. --ids RAG-001 RAG-002.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset without using Ollama or PostgreSQL.",
    )
    parser.add_argument(
        "--force-output",
        action="store_true",
        help="Allow overwriting the chosen output file.",
    )

    args = parser.parse_args()
    
    if (
        args.repeat_penalty is not None
        and args.repeat_penalty <= 0    
    ):
        parser.error("--repeat-penalty must be greater than zero.")

    try:
        dataset_path = Path(args.dataset).expanduser().resolve()
        cases = load_cases(dataset_path)
    except (OSError, ValueError) as exc:
        print(f"[Dataset error] {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(cases, dataset_path)
        return 0

    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1.", file=sys.stderr)
        return 2

    if args.ids:
        requested_ids = {value.strip().upper() for value in args.ids}
        existing_ids = {str(case["id"]).upper() for case in cases}
        unknown_ids = requested_ids - existing_ids

        if unknown_ids:
            print(
                "Unknown test IDs: " + ", ".join(sorted(unknown_ids)),
                file=sys.stderr,
            )
            return 2

        cases = [
            case
            for case in cases
            if str(case["id"]).upper() in requested_ids
        ]

    if args.limit is not None:
        cases = cases[:args.limit]

    if not cases:
        print("No cases were selected.", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIRECTORY / f"eval_{timestamp}.json"

    if output_path.exists() and not args.force_output:
        print(
            f"Output already exists: {output_path}\n"
            "Choose a new --output path, or pass --force-output "
            "if overwriting is intentional.",
            file=sys.stderr,
        )
        return 2

    baseline_path: Path | None = None
    baseline_data: dict[str, Any] | None = None

    if args.compare:
        baseline_path = Path(args.compare).expanduser().resolve()

        if not baseline_path.is_file():
            print(
                f"Baseline file not found: {baseline_path}",
                file=sys.stderr,
            )
            return 2

        if baseline_path == output_path:
            print(
                "The baseline and output paths must be different.",
                file=sys.stderr,
            )
            return 2

        try:
            baseline_data = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Baseline error] {exc}", file=sys.stderr)
            return 2

    # Runtime imports happen only after dataset validation and dry-run handling.
    try:
        from app.ai.chat_engine import ChatEngine
        from app.ai.ollama_client import OllamaClient
        from app.core.config import settings
        from app.core.prompt_manager import PromptManager
        from app.terminal.language import normalize_language_mode
    except Exception as exc:
        print(f"[Import error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print("Zara V2 evaluation run")
    print(f"Dataset cases: {len(cases)}")
    print(f"Output: {output_path}")
    print("Database writes: blocked by read-only transactions")
    print("Conversation history: isolated per case")
    print()

    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        print(
            f"[{index}/{len(cases)}] {case_id} "
            f"({case.get('category', 'uncategorized')}) ... ",
            end="",
            flush=True,
        )

        result = run_one_case(
            case,
            settings=settings,
            ChatEngine=ChatEngine,
            OllamaClient=OllamaClient,
            PromptManager=PromptManager,
            normalize_language_mode=normalize_language_mode,
            repeat_penalty=args.repeat_penalty,
            
        )
        results.append(result)
        

        print(
            f"{result['verdict']} "
            f"| {result['duration_seconds']:.2f}s"
        )

        if result["verdict"] == "error":
            print(f"    Error: {result['error']}")

    base_prompt_path = PROJECT_ROOT / "app" / "prompts" / "00_zara_base.txt"

    source_paths = {
        "chat_engine": PROJECT_ROOT / "app" / "ai" / "chat_engine.py",
        "conversation_decision": (
            PROJECT_ROOT / "app" / "ai" / "conversation_decision.py"
        ),
        "response_policy": PROJECT_ROOT / "app" / "ai" / "response_policy.py",
        "response_rules": (
            PROJECT_ROOT / "app" / "terminal" / "response_rules.py"
        ),
        "prompt_manager": PROJECT_ROOT / "app" / "core" / "prompt_manager.py",
        "ollama_client": PROJECT_ROOT / "app" / "ai" / "ollama_client.py",
        "base_prompt": base_prompt_path,

        "semantic_retriever": (
            PROJECT_ROOT / "app" / "learning_ai" / "semantic_retriever.py"
        ),
        "learning_store": (
            PROJECT_ROOT / "app" / "learning_ai" / "learning_store.py"
        ),
        "document_ingest": (
            PROJECT_ROOT / "app" / "learning_ai" / "document_ingest.py"
        ),
        "database_tools": (
            PROJECT_ROOT / "app" / "ai" / "tools" / "database.py"
        ),
        "evaluation_runner": Path(__file__).resolve(),
        "hinglish_playbook": (
            PROJECT_ROOT / "app" / "knowledge" / "hinglish_playbook.md"
        ),
        "improvement_observer": (
            PROJECT_ROOT / "app" / "improvement" / "observer.py"
        ),
        "answer_repair": (
            PROJECT_ROOT / "app" / "improvement_repair" / "answer_repair.py"
        ),
        "code_repair": (
            PROJECT_ROOT / "app" / "improvement_repair" / "code_repair.py"
        ),
        "system_a_evaluation": (
            PROJECT_ROOT / "scripts" / "system_a_eval_runner.py"
        ),
        "system_a_eval_cases": (
            PROJECT_ROOT / "app" / "knowledge" / "system_a_eval_cases.jsonl"
        ),
    }

    run_provenance = {
        "evaluation_schema_version": 2,
        "evaluation_path": "real_chatengine_read_only_database",
        "dataset_sha256": _sha256_file(dataset_path),
        "source_sha256": {
            name: _sha256_file(path)
            for name, path in source_paths.items()
        },
    }   

    run_record: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_case_count": len(cases),
        "model": settings.ollama_model,
        "settings": {
            "temperature": settings.ollama_temperature,
            "top_p": settings.ollama_top_p,
            "max_history_messages": settings.max_history_messages,
        },
        "provenance": run_provenance,
        "summary": summarize(results),
        "cases": results,
        "repeat_penalty_override": args.repeat_penalty,
    }

    if baseline_path is not None and baseline_data is not None:
        try:
            run_record["comparison"] = compare_with_baseline(
                results,
                baseline_path,
                baseline_data,
                current_provenance=run_provenance
            )
        except (OSError, ValueError) as exc:
            print(f"[Comparison error] {exc}", file=sys.stderr)
            return 2

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(run_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[Output error] {exc}", file=sys.stderr)
        return 2

    print("\n=== RUN SUMMARY ===")
    print(json.dumps(run_record["summary"], indent=2))

    if "comparison" in run_record:
        comparison = run_record["comparison"]
        print("\n=== BASELINE COMPARISON ===")
        print(f"Matched cases: {comparison['matched_case_count']}")
        print(f"Changed answers: {comparison['changed_answer_count']}")
        print(f"Changed verdicts: {comparison['changed_verdict_count']}")
        average_deltas = comparison.get("average_metric_deltas", {})

        print(
            "Average total-time delta: "
            f"{average_deltas.get('total_seconds')} seconds"
        )
        print(
            "Average first-output delta: "
            f"{average_deltas.get('first_output_seconds')} seconds"
        )
        print(
            "Average input-token delta: "
            f"{average_deltas.get('input_tokens')} tokens"
        )

    print(f"\nResults saved: {output_path}")
    print("Qualitative review is still required for review_needed cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())