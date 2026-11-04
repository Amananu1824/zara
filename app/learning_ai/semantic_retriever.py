"""Select relevant guidance and document excerpts for Zara."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.learning_ai.learning_store import LearningStore


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _requested_topic_number(text: str) -> str | None:
    match = re.search(
        r"\btopic\s*#?\s*(\d+)\b",
        text or "",
        re.IGNORECASE,
    )
    return match.group(1) if match else None

def _explicit_source_page_reference(
    text: str,
) -> tuple[str, list[int]] | None:
    """Extract a quoted source filename and explicitly named pages."""
    filename_match = re.search(
        r"""["'“‘]([^"'”’\n]{1,180}\.(?:pdf|docx?|txt|md))["'”’]""",
        text or "",
        re.IGNORECASE,
    )

    if not filename_match:
        return None

    pages: list[int] = []

    for raw_page in re.findall(
        r"\bpages?\s*#?\s*(\d+)\b",
        text or "",
        re.IGNORECASE,
    ):
        page = int(raw_page)
        if page > 0 and page not in pages:
            pages.append(page)

    if not pages:
        return None

    # filename = Path(
    #     filename_match.group(1).replace("\\", "/")
    # ).name

    # return filename, pages

    source_reference = (
        filename_match.group(1)
        .replace("\\", "/")
        .strip()
    )

    if "/" not in source_reference:
        source_reference = Path(source_reference).name

    return source_reference, pages


def _document_page_key(
    item: dict[str, Any],
    *,
    include_path: bool = False,
) -> tuple[str, int] | None:
    if item.get("source") != "document_chunk":
        return None

    try:
        page = int(item.get("page_number"))
    except (TypeError, ValueError):
        return None

    if include_path:
        source_key = str(item.get("source_path") or "")
        source_key = (
            source_key.replace("\\", "/").rstrip("/").casefold()
        )
    else:
        source_key = Path(
            str(item.get("source_file") or item.get("title") or "")
            .replace("\\", "/")
        ).name.casefold()

    if not source_key:
        return None

    return source_key, page

_RULE_TAG_PATTERN = re.compile(
    r"\bquick behaviour card\s*[—–-]\s*`([a-z][a-z0-9_.-]*)`",
    re.IGNORECASE,
)

_RULE_ID_PATTERN = re.compile(
    r"[a-z][a-z0-9_.-]{0,120}",
    re.IGNORECASE,
)


def _extract_rule_tags(
    item: dict[str, Any],
) -> tuple[str, ...]:
    """Extract stable rule IDs from explicit metadata and section headings."""
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    candidates = (
        item.get("rule_tags"),
        item.get("tags"),
        item.get("rule_id"),
        metadata.get("rule_tags"),
        metadata.get("tags"),
        metadata.get("rule_id"),
    )

    tags: list[str] = []

    def add_tag(value: Any) -> None:
        if not isinstance(value, str):
            return

        tag = value.strip().casefold()

        if tag and _RULE_ID_PATTERN.fullmatch(tag):
            tags.append(tag)

    for candidate in candidates:
        if isinstance(candidate, str):
            add_tag(candidate)

        elif isinstance(candidate, (list, tuple, set, frozenset)):
            values = (
                sorted(candidate)
                if isinstance(candidate, (set, frozenset))
                else candidate
            )

            for value in values:
                add_tag(value)

    section = str(item.get("section") or "")

    for match in _RULE_TAG_PATTERN.finditer(section):
        add_tag(match.group(1))

    return tuple(dict.fromkeys(tags))


def _contains_topic_label(text: str, topic_number: str) -> bool:
    pattern = rf"\btopic\s*#?\s*{re.escape(topic_number)}\b"
    return bool(re.search(pattern, text or "", re.IGNORECASE))

@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    source: str
    source_id: int | str | None
    similarity: float
    category: str
    content_kind: str
    title: str
    source_file: str
    source_path: str
    page_number: int | None
    section: str
    chunk_index: int | None
    document_id: int | None
    content_sha256: str
    rule_id: str
    topic_numbers: tuple[str, ...]
    match_reason: str = "similarity"
    rule_tags: tuple[str, ...] = ()
    source_key: str = ""
    scope: str = ""
    priority: int = 0
    version: int = 0
    source_version: int = 0
    extraction_method: str = ""
    extraction_status: str = "unknown"
    extraction_warnings: tuple[str, ...] = ()
    page_count: int | None = None
    source_encoding: str = ""
    chunking_strategy: str = ""
    chunk_content_kind: str = ""
    chunk_char_count: int = 0
    chunk_estimated_tokens: int = 0
    chunk_paragraph_count: int = 0
    chunk_overlap_chars: int = 0
    chunk_boundary_before: str = ""
    chunk_boundary_after: str = ""
    stable_chunk_id: str = ""

_TOPIC_NUMBER_PATTERN = re.compile(
    r"\btopic\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)


def _extract_topic_numbers(
    item: dict[str, Any],
) -> tuple[str, ...]:
    """Extract topic identifiers from the selected source chunk."""
    text = "\n".join((
        str(item.get("section") or ""),
        str(item.get("content") or ""),
    ))

    return tuple(dict.fromkeys(
        match.group(1)
        for match in _TOPIC_NUMBER_PATTERN.finditer(text)
    ))

class SemanticGuidanceRetriever:
    """Retrieve a bounded amount of relevant context."""

    def __init__(
        self,
        store: LearningStore,
        min_similarity: float = 0.28,
        min_document_similarity: float = 0.38,
    ) -> None:
        for value in (min_similarity, min_document_similarity):
            if not -1.0 <= value <= 1.0:
                raise ValueError("Similarity thresholds must be between -1 and 1.")

        self.store = store
        self.min_similarity = min_similarity
        self.min_document_similarity = min_document_similarity
        self.last_manifest: dict[str, Any] = {
            "status": "not_run",
            "candidate_count": 0,
            "selected_count": 0,
            "selected": [],
        }

    def retrieve(
        self,
        user_message: str,
    ) -> tuple[list[str], list[str], list[str]]:
        requested_topic = _requested_topic_number(user_message)

        explicit_page_reference = _explicit_source_page_reference(
            user_message
        )

        # Exact file/page requests do not need a semantic query embedding.
        matches = (
            []
            if explicit_page_reference
            else self.store.semantic_search(user_message, limit=30)
        )

        if requested_topic and not explicit_page_reference:
            matches = sorted(
                matches,
                key=lambda item: 0
                if (
                    str(item.get("source", "")) == "document_chunk"
                    and _contains_topic_label(
                        str(item.get("content", "")),
                        requested_topic,
                    )
                )
                else 1,
            )

        explicit_page_keys: set[tuple[str, int]] = set()
        explicit_reference_trace: dict[str, Any] | None = None
        explicit_reference_is_path = False

        if explicit_page_reference:
            requested_file, requested_pages = explicit_page_reference
            normalized_request = requested_file.replace("\\", "/").strip()
            explicit_reference_is_path = "/" in normalized_request

            requested_source_key = (
                normalized_request.rstrip("/").casefold()
                if explicit_reference_is_path
                else Path(normalized_request).name.casefold()
            )

            pinned_matches = self.store.get_document_chunks_by_source_pages(
                requested_file,
                requested_pages,
            )

            explicit_page_keys = {
                key
                for item in pinned_matches
                if (
                    key := _document_page_key(
                        item,
                        include_path=explicit_reference_is_path,
                    )
                ) is not None
            }

            matched_pages = sorted({
                page for _, page in explicit_page_keys
            })
            missing_pages = sorted(
                set(requested_pages) - set(matched_pages)
            )

            if not explicit_page_keys:
                # Never substitute a similar-looking document for an
                # unresolved or ambiguous explicitly requested source.
                matches = [
                    item for item in matches
                    if item.get("source") != "document_chunk"
                ]
                reference_status = "unresolved"

            else:
                remaining_matches: list[dict[str, Any]] = []

                for item in matches:
                    if item.get("source") != "document_chunk":
                        remaining_matches.append(item)
                        continue

                    key = _document_page_key(
                        item,
                        include_path=explicit_reference_is_path,
                    )

                    if key is None:
                        continue

                    # Keep additional pages from the same source only.
                    if (
                        key[0] == requested_source_key
                        and key not in explicit_page_keys
                    ):
                        remaining_matches.append(item)

                matches = pinned_matches + remaining_matches

                reference_status = (
                    "partially_matched"
                    if missing_pages
                    else "matched"
                )

            explicit_reference_trace = {
                "status": reference_status,
                "file_reference": Path(
                    normalized_request
                ).name,
                "requested_pages": sorted(set(requested_pages)),
                "matched_pages": matched_pages,
                "missing_pages": missing_pages,
            }

        behavior_guidance: list[str] = []
        playbook_guidance: list[str] = []
        feedback_lessons: list[str] = []
        document_context: list[str] = []
        document_context_limit = (
            12 if explicit_page_reference else 2
        )

        evidence: list[RetrievedEvidence] = []
        seen: set[tuple[str, ...]] = set()

        rejected = {
            "invalid_score": 0,
            "below_threshold": 0,
            "duplicate": 0,
            "capacity": 0,
            "empty_content": 0,
        }

        def record_evidence(
            item: dict[str, Any],
            similarity: float,
            *,
            reason: str = "similarity",
        ) -> None:
            def optional_int(value: Any) -> int | None:
                try:
                    return int(value) if value is not None else None
                except (TypeError, ValueError):
                    return None

            raw_id = item.get("source_id")
            source_id: int | str | None = (
                raw_id
                if isinstance(raw_id, (int, str))
                else None
            )

            metadata = item.get("metadata")
            if not isinstance(metadata, dict):  
                metadata = {}

            chunking_metadata = metadata.get("chunking")

            if not isinstance(chunking_metadata, dict):
                chunking_metadata = {}

            rule_tags = _extract_rule_tags(item)

            item_source = str(item.get("source", ""))
            source_file = str(item.get("source_file", ""))
            title = str(item.get("title", ""))

            base_name = Path(
                (source_file or title).replace("\\", "/")
            ).name.casefold()

            if item_source == "behavior_guide":
                content_kind = "behavioral_guidance"

            elif item_source == "feedback_lesson":
                content_kind = "reviewed_lesson"

            elif item_source == "system_a_lesson":
                content_kind = "approved_system_a_lesson"

            elif base_name == "hinglish_playbook.md":
                content_kind = "behavioral_guidance"

            else:
                content_kind = str(
                    item.get("content_kind")
                    or metadata.get("content_kind")
                    or "document_knowledge"
                )

            rule_id = str(
                item.get("rule_id")
                or metadata.get("rule_id")
                or (
                    rule_tags[0]
                    if len(rule_tags) == 1
                    else ""
                )
            ).strip().casefold()

            source_key = str(
                item.get("source_key")
                or item.get("guide_key")
                or metadata.get("source_key")
                or rule_id
                or ""
            )

            evidence.append(
                RetrievedEvidence(
                    source=str(item.get("source", "")),
                    source_id=source_id,
                    similarity=round(similarity, 5),
                    category=str(item.get("category", "general")),
                    content_kind=content_kind,
                    title=str(item.get("title", "")),
                    source_file=str(item.get("source_file", "")),
                    source_path=str(item.get("source_path", "")),
                    page_number=optional_int(item.get("page_number")),
                    section=str(item.get("section", "")),
                    chunk_index=optional_int(item.get("chunk_index")),
                    document_id=(
                        optional_int(item.get("source_parent_id"))
                        if item.get("source") == "document_chunk"
                        else None
                    ),
                    content_sha256=str(
                        item.get("content_sha256", "")
                    ),
                    rule_id=rule_id,
                    match_reason=reason,
                    rule_tags=rule_tags,
                    source_key=source_key,
                    source_version=(
                        optional_int(item.get("source_version")) or 0
                    ),
                    extraction_method=str(
                        item.get("extraction_method") or ""
                    ),
                    extraction_status=str(
                        item.get("extraction_status") or "unknown"
                    ),
                    extraction_warnings=tuple(
                        str(value)
                        for value in (
                            item.get("extraction_warnings") or []
                        )[:10]
                        if isinstance(value, str)
                    ),
                    page_count=optional_int(
                        item.get("page_count")
                    ),
                    source_encoding=str(
                        item.get("source_encoding") or ""
                    ),
                    chunking_strategy=str(
                        chunking_metadata.get("strategy") or ""
                    ),
                    chunk_content_kind=str(
                        chunking_metadata.get("content_kind") or ""
                    ),
                    chunk_char_count=(
                        optional_int(chunking_metadata.get("char_count")) or 0
                    ),
                    chunk_estimated_tokens=(
                        optional_int(chunking_metadata.get("estimated_tokens")) or 0
                    ),
                    chunk_paragraph_count=(
                        optional_int(chunking_metadata.get("paragraph_count")) or 0
                    ),
                    chunk_overlap_chars=(
                        optional_int(chunking_metadata.get("overlap_chars")) or 0
                    ),
                    chunk_boundary_before=str(
                        chunking_metadata.get("boundary_before") or ""
                    ),
                    chunk_boundary_after=str(
                        chunking_metadata.get("boundary_after") or ""
                    ),
                    stable_chunk_id=str(
                        chunking_metadata.get("stable_chunk_id") or ""
                    ),
                    scope=str(item.get("scope") or ""),
                    priority=optional_int(item.get("priority")) or 0,
                    version=optional_int(item.get("version")) or 0,
                    topic_numbers=_extract_topic_numbers(item),               
                )
            )

        for item in matches:
            try:
                similarity = float(item["similarity"])
            except (KeyError, TypeError, ValueError):
                rejected["invalid_score"] += 1
                continue

            if (
                not math.isfinite(similarity)
                or not -1.0001 <= similarity <= 1.0001
            ):
                rejected["invalid_score"] += 1
                continue

            similarity = max(-1.0, min(1.0, similarity))

            source = str(item.get("source", ""))
            category = str(item.get("category", "general"))
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()

            if not content:
                rejected["empty_content"] += 1
                continue

            normalized = _normalize_text(content)

            if source == "document_chunk":
                file_name = str(
                    item.get("source_file") or title or "Unknown source"
                )
                base_name = Path(
                    file_name.replace("\\", "/")
                ).name.casefold()

                if base_name == "hinglish_playbook.md":
                    if similarity < self.min_similarity:
                        rejected["below_threshold"] += 1
                        continue

                    if len(playbook_guidance) >= 2:
                        rejected["capacity"] += 1
                        continue

                    key = (source, normalized)

                    if key in seen:
                        rejected["duplicate"] += 1
                        continue

                    section = str(item.get("section") or "").strip()
                    rendered = (
                        "Hinglish Playbook guidance "
                        f"({section or 'General'}): {content}"
                    )

                    seen.add(key)
                    playbook_guidance.append(rendered)
                    record_evidence(
                        item,
                        similarity,
                        reason="playbook_guidance_similarity",
                    )
                    continue

                exact_topic_match = bool(
                    requested_topic
                    and _contains_topic_label(content, requested_topic)
                )
                page_key = _document_page_key(
                    item,
                    include_path=explicit_reference_is_path,
                )

                explicit_source_page_match = (
                    page_key is not None
                    and page_key in explicit_page_keys
                )

                if (
                    similarity < self.min_document_similarity
                    and not exact_topic_match
                    and not explicit_source_page_match
                ):
                    rejected["below_threshold"] += 1
                    continue

                if len(document_context) >= document_context_limit:
                    rejected["capacity"] += 1
                    continue

                page = item.get("page_number")
                section = str(item.get("section") or "").strip()

                citation = f"Source: {file_name}"

                if page is not None:
                    citation += f", page {page}"
                if section:
                    citation += f", section: {section}"

                extraction_status = str(
                    item.get("extraction_status") or "unknown"
                )

                if extraction_status == "partial":
                    citation += (
                        " (partial extraction; "
                        "some document content may be missing)"
                    )

                rendered = (
                    f"{citation}\nExcerpt: {content}"
                )

                key = (
                    source,
                    str(
                        item.get("source_key")
                        or item.get("source_path")
                        or file_name
                    ).casefold(),
                    str(page or ""),
                    section.casefold(),
                    normalized,
                )

                if key in seen:
                    rejected["duplicate"] += 1
                    continue

                seen.add(key)
                document_context.append(rendered)

                record_evidence(
                    item,
                    similarity,
                    reason=(
                        "explicit_source_page"
                        if explicit_source_page_match
                        else (
                            "exact_topic_match"
                            if exact_topic_match
                            else "similarity"
                        )       
                    ),      
                )
                continue

            if similarity < self.min_similarity:
                rejected["below_threshold"] += 1
                continue

            key = (source, normalized)

            if key in seen:
                rejected["duplicate"] += 1
                continue

            if source == "behavior_guide":
                if len(behavior_guidance) >= 2:
                    rejected["capacity"] += 1
                    continue

                behavior_guidance.append(
                    f"{title or category}: {content}"
                )

            elif source == "system_a_lesson":
                if len(behavior_guidance) >= 2:
                    rejected["capacity"] += 1
                    continue

                rule_id = str(
                    item.get("rule_id")
                    or item.get("source_key")
                    or ""
                ).strip()

                scope = str(item.get("scope") or "general")

                priority = max(
                    1,
                    min(int(item.get("priority") or 1), 100),
                )

                behavior_guidance.append(
                    "Approved Zara learned rule "
                    f"({scope}/{rule_id or category}; "
                    f"priority {priority}): {content}"
                )
                
            elif source == "feedback_lesson":
                if len(feedback_lessons) >= 1:
                    rejected["capacity"] += 1
                    continue

                feedback_lessons.append(
                    f"Reviewed correction ({category}): {content}"
                )

            else:
                continue

            seen.add(key)
            record_evidence(item, similarity)

        self.last_manifest = {
            "status": (
                "ok" if evidence else "no_relevant_matches"
            ),
            "candidate_count": len(matches),
            "selected_count": len(evidence),
            "thresholds": {
                "behavior_and_playbook": self.min_similarity,
                "documents": self.min_document_similarity,
            },
            "rejected": rejected,
            "selected_by_source": {
                source: sum(
                    1 for item in evidence if item.source == source
                )
                for source in {
                    item.source for item in evidence
                }
            },
            "selected": [
                asdict(item) for item in evidence
            ],
        }

        if explicit_reference_trace is not None:
            self.last_manifest["explicit_source_page_reference"] = (
                explicit_reference_trace
            )

        return (
            playbook_guidance + behavior_guidance,
            feedback_lessons,
            document_context,
        )