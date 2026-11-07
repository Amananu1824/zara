"""Select relevant guidance and document excerpts for Zara."""
from __future__ import annotations

"""Hybrid retrieval engine for Zara's guidance and documents."""
from __future__ import annotations

import logging
import math
import re
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from app.learning_ai.embedding_client import (
    EmbeddingCancelled,
    EmbeddingError,
)

from app.learning_ai.learning_store import (
    DatabaseOperationCancelled,
    LearningStore,
)

from app.learning_ai.evidence_guard import (
    sanitize_untrusted_excerpt,
)

logger = logging.getLogger(__name__)


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
    evidence_id: str = ""
    excerpt: str = ""
    retrieval_score: float = 0.0
    semantic_similarity: float = 0.0
    keyword_score: float = 0.0
    updated_at_epoch: float = 0.0

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

_RETRIEVAL_TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9_]{2,}|[\u0900-\u097F]{2,}"
)

_RETRIEVAL_STOP_WORDS = frozenset({
    "please",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "what",
    "which",
    "how",
    "can",
    "could",
    "would",
    "you",
    "your",
    "me",
    "my",
    "tell",
    "show",
    "give",
    "explain",
    "find",
    "search",
    "for",
    "from",
    "about",
    "mujhe",
    "batao",
    "bataye",
    "samjhao",
    "kya",
    "hai",
    "hain",
    "ka",
    "ke",
    "ki",
    "ko",
    "mein",
    "me",
    "par",
    "se",
})


def _refine_retrieval_query(query: str) -> str:
    """
    Deterministically remove conversational filler.

    Preserve quoted phrases and explicit web-search operators.
    Never call an LLM for this operation.
    """
    original = re.sub(r"\s+", " ", str(query or "")).strip()

    if not original:
        return ""

    # Preserve user-specified phrase/boolean query semantics.
    if (
        '"' in original
        or re.search(
            r"\bOR\b|(?:^|\s)-",
            original,
            re.IGNORECASE,
        )
    ):
        return original

    tokens = _RETRIEVAL_TOKEN_PATTERN.findall(
        original.casefold()
    )

    meaningful = [
        token
        for token in tokens
        if token not in _RETRIEVAL_STOP_WORDS
    ]

    if not meaningful:
        return original

    return " ".join(meaningful)


def _normalize_metadata_filters(
    filters: dict[str, Any] | None,
) -> dict[str, set[Any]]:
    """
    Validate a small allowlist of retrieval metadata filters.

    Unsupported fields are rejected rather than passed to SQL.
    """
    if filters is None:
        return {}

    if not isinstance(filters, dict):
        raise ValueError("Metadata filters must be an object.")

    allowed_fields = {
        "source",
        "category",
        "source_key",
        "source_version",
        "page_number",
    }

    unknown_fields = set(filters) - allowed_fields

    if unknown_fields:
        raise ValueError(
            "Unsupported retrieval metadata filter."
        )

    numeric_fields = {
        "source_version",
        "page_number",
    }

    allowed_sources = {
        "behavior_guide",
        "feedback_lesson",
        "system_a_lesson",
        "document_chunk",
    }

    normalized: dict[str, set[Any]] = {}

    for field, raw_value in filters.items():
        values = (
            list(raw_value)
            if isinstance(raw_value, (list, tuple, set, frozenset))
            else [raw_value]
        )

        if not values or len(values) > 20:
            raise ValueError(
                f"Invalid number of values for {field}."
            )

        converted: set[Any] = set()

        for value in values:
            if field in numeric_fields:
                if isinstance(value, bool):
                    raise ValueError(
                        f"{field} must contain integers."
                    )

                try:
                    number = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{field} must contain integers."
                    ) from exc

                if number < 1:
                    raise ValueError(
                        f"{field} values must be positive."
                    )

                converted.add(number)

            else:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{field} must contain non-empty strings."
                    )

                text = value.strip().casefold()

                if field == "source" and text not in allowed_sources:
                    raise ValueError(
                        "Unsupported retrieval source."
                    )

                if field == "source_key":
                    text = text.replace("\\", "/")

                converted.add(text)

        normalized[field] = converted

    return normalized


def _matches_metadata_filters(
    item: dict[str, Any],
    filters: dict[str, set[Any]],
) -> bool:
    """Return True only when every supplied filter matches."""
    for field, allowed_values in filters.items():
        raw_value = item.get(field)

        if raw_value is None:
            return False

        if field in {"source_version", "page_number"}:
            try:
                actual_value = int(raw_value)
            except (TypeError, ValueError):
                return False

        else:
            actual_value = str(raw_value).strip().casefold()

            if field == "source_key":
                actual_value = actual_value.replace("\\", "/")

        if actual_value not in allowed_values:
            return False

    return True


def _retrieval_identity(
    item: dict[str, Any],
) -> tuple[str, str]:
    """Create a stable identity across semantic and lexical channels."""
    source = str(item.get("source") or "unknown")
    source_id = item.get("source_id")

    if source_id is None:
        source_id = (
            item.get("content_sha256")
            or _normalize_text(str(item.get("content") or ""))
        )

    return source, str(source_id)


def _fuse_ranked_results(
    semantic_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """
    Combine independently ranked channels using reciprocal rank fusion.

    A deterministic second-stage reranker uses query coverage,
    title relevance, source recency and approved lesson priority.
    """
    fused: dict[tuple[str, str], dict[str, Any]] = {}

    for rank, raw_item in enumerate(semantic_results, 1):
        item = dict(raw_item)
        key = _retrieval_identity(item)

        try:
            similarity = float(item.get("similarity", 0.0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(similarity):
            continue

        item["semantic_rank"] = rank
        item["semantic_similarity"] = similarity
        item["keyword_rank"] = None
        item["keyword_score"] = 0.0

        fused[key] = item

    for rank, raw_item in enumerate(keyword_results, 1):
        item = dict(raw_item)
        key = _retrieval_identity(item)

        try:
            keyword_score = float(item.get("keyword_score", 0.0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(keyword_score) or keyword_score <= 0:
            continue

        existing = fused.get(key)

        if existing is None:
            existing = item
            existing["semantic_rank"] = None
            existing["semantic_similarity"] = None
            existing["similarity"] = 0.0
            fused[key] = existing

        else:
            # Preserve semantic scores while filling missing metadata.
            for field, value in item.items():
                if (
                    field not in existing
                    or existing[field] is None
                    or existing[field] == ""
                ):
                    existing[field] = value

        existing["keyword_rank"] = rank
        existing["keyword_score"] = keyword_score

    refined_query = _refine_retrieval_query(query)
    query_tokens = set(
        _RETRIEVAL_TOKEN_PATTERN.findall(
            refined_query.casefold()
        )
    ) - _RETRIEVAL_STOP_WORDS

    now = time.time()
    max_rrf = 2.0 / 61.0

    ranked_results: list[dict[str, Any]] = []

    for item in fused.values():
        semantic_rank = item.get("semantic_rank")
        keyword_rank = item.get("keyword_rank")

        semantic_contribution = (
            1.0 / (60 + semantic_rank)
            if semantic_rank is not None
            else 0.0
        )

        keyword_contribution = (
            1.0 / (60 + keyword_rank)
            if keyword_rank is not None
            else 0.0
        )

        rrf_score = min(
            1.0,
            (
                semantic_contribution + keyword_contribution
            ) / max_rrf,
        )

        title = str(item.get("title") or "")
        text = " ".join(
            str(item.get(field) or "")
            for field in (
                "title",
                "content",
                "section",
                "category",
                "scope",
                "source_key",
                "rule_id",
            )
        )

        text_tokens = set(
            _RETRIEVAL_TOKEN_PATTERN.findall(
                text.casefold()
            )
        )

        title_tokens = set(
            _RETRIEVAL_TOKEN_PATTERN.findall(
                title.casefold()
            )
        )

        coverage = (
            len(query_tokens & text_tokens) / len(query_tokens)
            if query_tokens else 0.0
        )

        title_coverage = (
            len(query_tokens & title_tokens) / len(query_tokens)
            if query_tokens else 0.0
        )

        normalized_text = _normalize_text(text)
        normalized_query = _normalize_text(refined_query)

        phrase_score = (
            1.0
            if (
                len(normalized_query) >= 8
                and normalized_query in normalized_text
            )
            else 0.0
        )

        try:
            updated_at = float(
                item.get("updated_at_epoch") or 0.0
            )
        except (TypeError, ValueError):
            updated_at = 0.0

        age_days = (
            max(0.0, (now - updated_at) / 86400.0)
            if updated_at > 0
            else None
        )

        recency_score = (
            0.025 * math.exp(-age_days / 365.0)
            if age_days is not None
            else 0.0
        )

        try:
            priority = max(
                0,
                min(100, int(item.get("priority") or 0)),
            )
        except (TypeError, ValueError):
            priority = 0

        priority_score = priority * 0.00015

        relevance_score = (
            0.76 * rrf_score
            + 0.13 * coverage
            + 0.04 * title_coverage
            + 0.04 * phrase_score
            + recency_score
            + priority_score
        )

        relevance_score = max(
            0.0,
            min(1.0, relevance_score),
        )

        item["retrieval_score"] = round(
            relevance_score,
            6,
        )

        item["match_reason"] = (
            "hybrid"
            if semantic_rank is not None and keyword_rank is not None
            else (
                "semantic"
                if semantic_rank is not None
                else "keyword"
            )
        )

        item["similarity"] = (
            float(item["semantic_similarity"])
            if item.get("semantic_similarity") is not None
            else 0.0
        )

        item["recency_score"] = round(
            recency_score,
            6,
        )

        item["query_coverage"] = round(
            coverage,
            6,
        )

        ranked_results.append(item)

    ranked_results.sort(
        key=lambda item: (
            float(item.get("retrieval_score") or 0.0),
            int(item.get("priority") or 0),
            float(item.get("updated_at_epoch") or 0.0),
            str(item.get("source") or ""),
            str(item.get("source_id") or ""),
        ),
        reverse=True,
    )

    return ranked_results

class SemanticGuidanceRetriever:
    """Retrieve a bounded amount of relevant context."""

    def __init__(
        self,
        store: LearningStore,
        min_similarity: float = 0.28,
        min_document_similarity: float = 0.38,
        min_keyword_score: float = 0.005,
        min_relevance_score: float = 0.22,
    ) -> None:
        for value in (
            min_similarity,
            min_document_similarity,
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    "Similarity thresholds must be between -1 and 1."
                )

        if not 0.0 <= min_keyword_score <= 1.0:
            raise ValueError(
                "Keyword score threshold must be between 0 and 1."
            )

        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError(
                "Relevance threshold must be between 0 and 1."
            )

        self.store = store
        self.min_similarity = min_similarity
        self.min_document_similarity = min_document_similarity
        self.min_keyword_score = min_keyword_score
        self.min_relevance_score = min_relevance_score

        self.last_manifest: dict[str, Any] = {
            "status": "not_run",
            "algorithm": "hybrid_rrf_v1",
            "candidate_count": 0,
            "selected_count": 0,
            "selected": [],
            "retrieval_engine": {},
        }

    def retrieve(
        self,
        user_message: str,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        requested_topic = _requested_topic_number(user_message)

        started_at = perf_counter()

        normalized_filters = _normalize_metadata_filters(
            metadata_filters
        )

        requested_topic = _requested_topic_number(
            user_message
        )

        explicit_page_reference = (
            _explicit_source_page_reference(user_message)
        )

        refined_query = _refine_retrieval_query(
            user_message
        )

        semantic_matches: list[dict[str, Any]] = []
        keyword_matches: list[dict[str, Any]] = []

        semantic_error: str | None = None
        keyword_error: str | None = None

        metadata_filter_rejections = 0

        semantic_status = "not_requested"
        keyword_status = "not_requested"

        if not explicit_page_reference:

            # CHANNEL 1: semantic retrieval.
            try:
                semantic_matches = self.store.semantic_search(
                    user_message,
                    limit=30,
                )
                semantic_status = "available"

            except DatabaseOperationCancelled:
                raise

            except EmbeddingCancelled:
                raise

            except EmbeddingError as exc:
                semantic_status = "embedding_unavailable"
                semantic_error = type(exc).__name__

                logger.warning(
                    "Semantic retrieval unavailable (%s). "
                    "Continuing with keyword retrieval.",
                    semantic_error,
                )

            # CHANNEL 2: PostgreSQL full-text keyword retrieval.
            try:
                keyword_matches = self.store.keyword_search(
                    refined_query,
                    limit=30,
                )
                keyword_status = "available"

            except DatabaseOperationCancelled:
                raise

            except Exception as exc:
                keyword_status = "unavailable"
                keyword_error = type(exc).__name__

                logger.warning(
                    "Keyword retrieval unavailable (%s).",
                    keyword_error,
                )

                # Keep semantic-only retrieval available when FTS fails.
                if not semantic_matches:
                    raise RuntimeError(
                        "Hybrid retrieval has no available search channel."
                    ) from exc

            # Apply allowlisted metadata filters before rank fusion.
            if normalized_filters:
                before_filter = (
                    len(semantic_matches)
                    + len(keyword_matches)
                )

                semantic_matches = [
                    item
                    for item in semantic_matches
                    if _matches_metadata_filters(
                        item,
                        normalized_filters,
                    )
                ]

                keyword_matches = [
                    item
                    for item in keyword_matches
                    if _matches_metadata_filters(
                        item,
                        normalized_filters,
                    )
                ]

                metadata_filter_rejections += (
                    before_filter
                    - len(semantic_matches)
                    - len(keyword_matches)
                )

            # Merge both result channels and rerank deterministically.
            matches = _fuse_ranked_results(
                semantic_matches,
                keyword_matches,
                refined_query,
            )

        else:
            # Exact source/page lookup is authoritative.
            # Avoid unnecessary model and keyword searches.
            matches = []

        semantic_candidate_count = len(semantic_matches)
        keyword_candidate_count = len(keyword_matches)

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

        if normalized_filters:
            before_page_filter = len(matches)

            matches = [
                item
                for item in matches
                if _matches_metadata_filters(
                    item,
                    normalized_filters,
                )
            ]

            metadata_filter_rejections += (
                before_page_filter - len(matches)
            )
            
        behavior_guidance: list[str] = []
        playbook_guidance: list[str] = []
        feedback_lessons: list[str] = []
        document_context: list[str] = []
        document_context_limit = (
            12 if explicit_page_reference else 2
        )

        evidence: list[RetrievedEvidence] = []
        seen: set[tuple[str, ...]] = set()
        document_source_counts: dict[str, int] = {}

        rejected = {
            "invalid_score": 0,
            "below_threshold": 0,
            "duplicate": 0,
            "capacity": 0,
            "empty_content": 0,
            "diversity": 0,
            "metadata_filter": metadata_filter_rejections,
        }

        def passes_relevance_threshold(
            threshold: float,
            *,
            similarity: float,
            semantic_available: bool,
            keyword_score: float,
            retrieval_score: float,
        ) -> bool:
            semantic_pass = (
                semantic_available
                and similarity >= threshold
            )

            keyword_pass = (
                keyword_score >= self.min_keyword_score
                and retrieval_score >= max(
                    self.min_relevance_score,
                    threshold * 0.65,
                )
            )

            return semantic_pass or keyword_pass
        
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

            evidence_id = ""

            if (
                item_source == "document_chunk"
                and content_kind == "document_knowledge"
            ):
                existing_document_evidence = sum(
                    1
                    for existing in evidence
                    if existing.evidence_id
                )

                evidence_id = (
                    f"D{existing_document_evidence + 1}"
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
                    evidence_id=evidence_id,
                    excerpt=(
                        sanitize_untrusted_excerpt(
                            item.get("content"),
                            limit=5000,
                        )
                        if evidence_id
                        else ""
                    ),
                    scope=str(item.get("scope") or ""),
                    priority=optional_int(item.get("priority")) or 0,
                    version=optional_int(item.get("version")) or 0,
                    topic_numbers=_extract_topic_numbers(item),               
                    retrieval_score=float(
                        item.get("retrieval_score") or 0.0
                    ),
                    semantic_similarity=float(
                        item.get("semantic_similarity")
                        if item.get("semantic_similarity") is not None
                        else similarity
                    ),
                    keyword_score=float(
                        item.get("keyword_score") or 0.0
                    ),
                    updated_at_epoch=float(
                        item.get("updated_at_epoch") or 0.0
                    ),                
                )
            )

        for item in matches:
            semantic_rank = item.get("semantic_rank")

            semantic_available = (
                semantic_rank is not None
                or (
                    "semantic_rank" not in item
                    and "keyword_rank" not in item
                )
            )

            raw_semantic_score = item.get(
                "semantic_similarity"
            )

            if raw_semantic_score is None:
                raw_semantic_score = item.get(
                    "similarity",
                    0.0,
                )

            try:
                similarity = float(raw_semantic_score)
                retrieval_score = float(
                    item.get("retrieval_score") or 0.0
                )
                keyword_score = float(
                    item.get("keyword_score") or 0.0
                )

            except (TypeError, ValueError):
                rejected["invalid_score"] += 1
                continue

            if (
                not math.isfinite(similarity)
                or not -1.0001 <= similarity <= 1.0001
                or not math.isfinite(retrieval_score)
                or not 0.0 <= retrieval_score <= 1.0001
                or not math.isfinite(keyword_score)
                or keyword_score < 0.0
            ):
                rejected["invalid_score"] += 1
                continue

            similarity = max(
                -1.0,
                min(1.0, similarity),
            )

            try:
                retrieval_score = float(
                    item.get("retrieval_score") or 0.0
                )

                keyword_score = float(
                    item.get("keyword_score") or 0.0
                )

            except (TypeError, ValueError):
                rejected["invalid_score"] += 1
                continue

            semantic_available = (
                item.get("semantic_rank") is not None
                or (
                    "semantic_rank" not in item
                    and "keyword_rank" not in item
                )
            )

            if (
                not math.isfinite(retrieval_score)
                or not 0.0 <= retrieval_score <= 1.0001
                or not math.isfinite(keyword_score)
                or keyword_score < 0.0
            ):
                rejected["invalid_score"] += 1
                continue

            retrieval_score = max(
                0.0,
                min(1.0, retrieval_score),
            )

            retrieval_score = max(
                0.0,
                min(1.0, retrieval_score),
            )
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

                    if not passes_relevance_threshold(
                        self.min_similarity,
                        similarity=similarity,
                        semantic_available=semantic_available,
                        keyword_score=keyword_score,
                        retrieval_score=retrieval_score,
                    ):
                        rejected["below_threshold"] += 1
                        continue

                    if len(playbook_guidance) >= 2:
                        rejected["capacity"] += 1
                        continue

                    key = (source, normalized)

                    if key in seen:
                        rejected["duplicate"] += 1
                        continue

                    section = str(
                        item.get("section") or ""
                    ).strip()

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
                    not passes_relevance_threshold(
                        self.min_document_similarity,
                        similarity=similarity,
                        semantic_available=semantic_available,
                        keyword_score=keyword_score,
                        retrieval_score=retrieval_score,
                    )
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

                document_evidence_id = (
                    f"D{1 + sum(
                        1
                        for existing in evidence
                        if existing.evidence_id
                    )}"
                )

                safe_excerpt = sanitize_untrusted_excerpt(
                    content,
                    limit=5000,
                )

                rendered = (
                    f"[{document_evidence_id}] "
                    f"<ZARA_UNTRUSTED_DOCUMENT "
                    f"id=\"{document_evidence_id}\">\n"
                    f"{citation}\n"
                    f"Excerpt: {safe_excerpt}\n"
                    f"</ZARA_UNTRUSTED_DOCUMENT>"
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

                document_source_key = str(
                    item.get("source_key")
                    or item.get("source_path")
                    or file_name
                ).replace("\\", "/").casefold()

                if (
                    not explicit_page_reference
                    and document_source_counts.get(
                        document_source_key,
                        0,
                    ) >= 1
                ):
                    rejected["diversity"] += 1
                    continue

                seen.add(key)
                document_context.append(rendered)

                document_source_counts[
                    document_source_key
                ] = document_source_counts.get(
                    document_source_key,
                    0,
                ) + 1

                record_evidence(
                    item,
                    similarity,
                    reason=(
                        "explicit_source_page"
                        if explicit_source_page_match
                        else (
                            "exact_topic_match"
                            if exact_topic_match
                            else str(item.get("match_reason") or "semantic")
                        )       
                    ),      
                )
                continue

            if not passes_relevance_threshold(
                self.min_similarity,
                similarity=similarity,
                semantic_available=semantic_available,
                keyword_score=keyword_score,
                retrieval_score=retrieval_score,
            ):
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
            record_evidence(
                item,
                similarity,
                reason=str(
                    item.get("match_reason") or "semantic"
                ),
            )
        if semantic_error and keyword_matches:
            retrieval_status = "keyword_fallback"
        elif keyword_error and semantic_matches:
            retrieval_status = "semantic_fallback"
        elif evidence:
            retrieval_status = "ok"
        else:
            retrieval_status = "no_relevant_matches"

        retrieval_latency_ms = round(
            (perf_counter() - started_at) * 1000.0,
            3,
        )

        self.last_manifest = {
            "status": retrieval_status,

            "algorithm": "hybrid_rrf_v1",

            "candidate_count": len(matches),
            "selected_count": len(evidence),

            "thresholds": {
                "behavior_and_playbook": self.min_similarity,
                "documents": self.min_document_similarity,
                "keyword_score": self.min_keyword_score,
                "minimum_relevance": self.min_relevance_score,
            },

            "retrieval_engine": {
                "semantic_status": semantic_status,
                "keyword_status": keyword_status,

                "semantic_candidates": semantic_candidate_count,
                "keyword_candidates": keyword_candidate_count,

                "semantic_error": semantic_error,
                "keyword_error": keyword_error,

                "hybrid_fusion": "reciprocal_rank_fusion",
                "reranker": "deterministic_v1",

                "query_refinement_applied": (
                    refined_query != str(user_message or "").strip()
                ),

                "metadata_filter_keys": sorted(
                    normalized_filters.keys()
                ),

                "metadata_filter_rejections": (
                    metadata_filter_rejections
                ),

                "latency_ms": retrieval_latency_ms,
            },

            "rejected": rejected,

            "selected_by_source": {
                source: sum(
                    1
                    for item in evidence
                    if item.source == source
                )
                for source in {
                    item.source
                    for item in evidence
                }
            },

            "selected": [
                asdict(item)
                for item in evidence
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