"""Evidence registry, citation validation and grounded-answer guard."""

from __future__ import annotations

import re

from pathlib import Path
from typing import Any


_CITATION_PATTERN = re.compile(
    r"\[(D\d{1,3})\]",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(
    r"[a-z0-9]{3,}|[\u0900-\u097f]{2,}",
    re.IGNORECASE,
)

_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with",
    "from", "are", "was", "were", "has", "have",
    "had", "you", "your", "his", "her", "its",
    "not", "but", "into", "than", "then", "also",
    "can", "could", "would", "should", "will",
    "shall", "about", "according", "document",
    "source", "page", "section", "excerpt",
    "mujhe", "aap", "unka", "unki", "hai",
    "hain", "tha", "thi", "aur", "yeh", "woh",
    "mein", "ke", "ki", "ka", "ko", "se",
    "par", "liye", "kya", "kaise",
})

_DOCUMENT_QUERY_PATTERN = re.compile(
    r"\b(?:pdf|docx?|file|document|uploaded|attachment|"
    r"page\s*\d+|according\s+to|in\s+the\s+"
    r"(?:file|document|pdf)|from\s+(?:the\s+)?"
    r"(?:file|document)|source\s+document)\b|"
    r"\b(?:contract|manual|notes|report|uploaded\s+"
    r"(?:file|document|contract))\b|"
    r"\.(?:pdf|docx?|txt|md)\b",
    re.IGNORECASE,
)

_UNTRUSTED_INSTRUCTION_PATTERN = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|system|developer)\s+instructions|"
    r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt|"
    r"override\s+(?:the\s+)?(?:system|developer)\s+(?:message|policy)|"
    r"call\s+(?:a\s+)?tool|execute\s+(?:this\s+)?command|"
    r"delete\s+(?:all\s+)?(?:data|files|records)|"
    r"send\s+(?:the\s+)?(?:password|api\s+key|secret)",
    re.IGNORECASE,
)

_NUMBER_PATTERN = re.compile(
    r"(?<![a-z])"
    r"(?:\d{1,4}(?:[,.]\d{3})*(?:\.\d+)?\s*%?|"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}-\d{2}-\d{2})"
    r"(?![a-z])",
    re.IGNORECASE,
)


def is_document_question(question: str) -> bool:
    return bool(
        _DOCUMENT_QUERY_PATTERN.search(
            str(question or "")
        )
    )


def sanitize_untrusted_excerpt(
    value: Any,
    limit: int = 5000,
) -> str:
    """Prevent excerpt text from breaking Zara's evidence delimiters."""

    text = str(value or "")
    text = text.replace("\x00", " ").strip()

    text = re.sub(
        r"</?ZARA_UNTRUSTED_DOCUMENT\b",
        lambda match: (
            match.group(0)
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        ),
        text,
        flags=re.IGNORECASE,
    )

    return text[:limit]


def build_document_evidence_registry(
    manifest: dict[str, Any] | None,
    included_context: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a per-turn registry from actual selected document chunks.

    Only chunks included in the current prompt can be cited.
    """

    if not isinstance(manifest, dict):
        return []

    selected = manifest.get("selected", [])

    if not isinstance(selected, list):
        return []

    visible_context = "\n".join(
        str(item)
        for item in (included_context or [])
    )

    visible_ids = {
        value.upper()
        for value in _CITATION_PATTERN.findall(
            visible_context
        )
    }

    registry: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in selected:

        if not isinstance(item, dict):
            continue

        if item.get("source") != "document_chunk":
            continue

        if item.get("content_kind") != "document_knowledge":
            continue

        evidence_id = str(
            item.get("evidence_id") or ""
        ).upper()

        if not re.fullmatch(r"D\d{1,3}", evidence_id):
            continue

        if (
            included_context is not None
            and evidence_id not in visible_ids
        ):
            continue

        if evidence_id in seen:
            continue

        excerpt = sanitize_untrusted_excerpt(
            item.get("excerpt")
        )

        if not excerpt:
            continue

        source_file = str(
            item.get("source_file") or ""
        ).strip()

        source_path = str(
            item.get("source_path") or ""
        ).strip()

        title = str(
            item.get("title") or ""
        ).strip()

        normalized_source_path = source_path.replace(
            "\\",
            "/",
        )

        display_name = (
            source_file
            or (
                normalized_source_path.rsplit("/", 1)[-1]
                if normalized_source_path
                else title
            )
        )

        try:
            source_version = int(
                item.get("source_version") or 0
            )
        except (TypeError, ValueError):
            source_version = 0

        try:
            retrieval_score = float(
                item.get("retrieval_score")
                or item.get("similarity")
                or 0.0
            )
        except (TypeError, ValueError):
            retrieval_score = 0.0

        registry.append({
            "evidence_id": evidence_id,
            "source": "document_chunk",
            "source_id": item.get("source_id"),
            "document_id": item.get("document_id"),
            "source_key": str(
                item.get("source_key") or ""
            ),
            "source_version": source_version,
            "source_file": display_name[:240],
            "page_number": item.get("page_number"),
            "section": str(
                item.get("section") or ""
            )[:300],
            "chunk_index": item.get("chunk_index"),
            "stable_chunk_id": str(
                item.get("stable_chunk_id") or ""
            ),
            "content_sha256": str(
                item.get("content_sha256") or ""
            ),
            "retrieval_score": max(
                0.0,
                min(1.0, retrieval_score),
            ),
            "excerpt": excerpt,
            "contains_instruction_like_text": bool(
                _UNTRUSTED_INSTRUCTION_PATTERN.search(
                    excerpt
                )
            ),
        })

        seen.add(evidence_id)

    return registry[:12]


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_PATTERN.findall(
            str(text or "")
        )
        if token.casefold() not in _STOP_WORDS
    }


def _split_claims(paragraph: str) -> list[str]:
    text = _CITATION_PATTERN.sub(
        "",
        paragraph,
    ).strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?।])\s+",
        text,
    )

    return [
        part.strip(" \t-*•.)")
        for part in parts
        if len(_tokens(part)) >= 2
    ]


def _possible_numeric_conflicts(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Detect possible contradictory numeric/date statements.

    This is a conservative heuristic, not a semantic contradiction proof.
    """

    conflicts: list[dict[str, Any]] = []

    for index, first in enumerate(evidence):

        first_tokens = _tokens(
            first.get("excerpt", "")
        )

        first_numbers = set(
            _NUMBER_PATTERN.findall(
                first.get("excerpt", "")
            )
        )

        if not first_numbers:
            continue

        for second in evidence[index + 1:]:

            first_source_key = str(
                first.get("source_key") or ""
            )

            second_source_key = str(
                second.get("source_key") or ""
            )

            if first_source_key == second_source_key:
                continue

            second_tokens = _tokens(
                second.get("excerpt", "")
            )

            second_numbers = set(
                _NUMBER_PATTERN.findall(
                    second.get("excerpt", "")
                )
            )

            shared = first_tokens & second_tokens

            if (
                len(shared) >= 3
                and second_numbers
                and first_numbers != second_numbers
            ):
                conflicts.append({
                    "evidence_ids": [
                        first["evidence_id"],
                        second["evidence_id"],
                    ],
                    "shared_terms": sorted(shared)[:8],
                    "numeric_values": {
                        first["evidence_id"]: sorted(
                            first_numbers
                        )[:8],
                        second["evidence_id"]: sorted(
                            second_numbers
                        )[:8],
                    },
                    "type": "possible_numeric_conflict",
                })

            if len(conflicts) >= 5:
                return conflicts

    return conflicts


def format_visible_citations(
    answer: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Convert internal [D1] markers to readable source references."""

    registry = {
        item["evidence_id"]: item
        for item in evidence
        if isinstance(item, dict)
        and item.get("evidence_id")
    }

    def replace(match: re.Match[str]) -> str:

        evidence_id = match.group(1).upper()
        item = registry.get(evidence_id)

        if item is None:
            return match.group(0)

        label = str(
            item.get("source_file") or "document"
        )

        details = [label]

        page = item.get("page_number")

        if page is not None:
            details.append(f"page {page}")

        section = str(
            item.get("section") or ""
        ).strip()

        if section:
            details.append(f"section {section}")

        stable_chunk_id = str(
            item.get("stable_chunk_id") or ""
        ).strip()

        chunk_index = item.get("chunk_index")

        if stable_chunk_id:
            details.append(
                f"chunk {stable_chunk_id[:16]}"
            )
        elif chunk_index is not None:
            details.append(
                f"chunk index {chunk_index}"
            )

        return "[" + ", ".join(details) + "]"

    return _CITATION_PATTERN.sub(
        replace,
        str(answer or ""),
    )


def verify_grounded_answer(
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate citation references and claim/source lexical overlap.

    No model, network, database or tool call is made here.
    """

    answer = str(answer or "").strip()
    document_required = is_document_question(question)

    known = {
        item["evidence_id"]: item
        for item in evidence
        if isinstance(item, dict)
        and item.get("evidence_id")
    }

    conflicts = _possible_numeric_conflicts(
        evidence
    )

    used_ids = {
        value.upper()
        for value in _CITATION_PATTERN.findall(answer)
    }

    unknown_ids = sorted(
        used_ids - set(known)
    )

    if document_required and not known:
        return {
            "status": "insufficient_evidence",
            "blocking": True,
            "confidence": "none",
            "claim_source_map": [],
            "unknown_citations": unknown_ids,
            "possible_conflicts": conflicts,
            "safe_response": (
                "Is document-based question ke liye relevant "
                "verified passage retrieve nahi hua. Main bina "
                "source evidence ke answer guess nahi karungi."
            ),
        }

    paragraphs = [
        part.strip()
        for part in re.split(
            r"\n\s*\n",
            answer,
        )
        if part.strip()
    ]

    claim_source_map: list[dict[str, Any]] = []
    missing_citations: list[str] = []
    unsupported_claims: list[str] = []
    support_scores: list[float] = []

    for paragraph in paragraphs:

        paragraph_ids = {
            value.upper()
            for value in _CITATION_PATTERN.findall(
                paragraph
            )
        }

        paragraph_ids &= set(known)

        for claim in _split_claims(paragraph):

            claim_tokens = _tokens(claim)
            local_ids = paragraph_ids

            if not local_ids:

                if document_required:
                    missing_citations.append(
                        claim[:180]
                    )

                continue

            mappings: list[dict[str, Any]] = []
            best_overlap = 0.0

            for evidence_id in sorted(local_ids):

                source = known[evidence_id]

                source_tokens = _tokens(
                    source.get("excerpt", "")
                )

                shared = claim_tokens & source_tokens

                overlap = (
                    len(shared)
                    / max(1, len(claim_tokens))
                )

                best_overlap = max(
                    best_overlap,
                    overlap,
                )

                mappings.append({
                    "evidence_id": evidence_id,
                    "source_file": source.get(
                        "source_file"
                    ),
                    "page_number": source.get(
                        "page_number"
                    ),
                    "section": source.get(
                        "section"
                    ),
                    "source_version": source.get(
                        "source_version"
                    ),
                    "chunk_index": source.get(
                        "chunk_index"
                    ),
                    "stable_chunk_id": source.get(
                        "stable_chunk_id"
                    ),
                    "content_sha256": source.get(
                        "content_sha256"
                    ),
                    "shared_terms": sorted(shared)[:12],
                    "lexical_overlap": round(
                        overlap,
                        4,
                    ),
                })

            if best_overlap <= 0.0:
                unsupported_claims.append(
                    claim[:180]
                )
            else:
                support_scores.append(best_overlap)

            claim_source_map.append({
                "claim": claim[:300],
                "sources": mappings,
                "lexical_support": round(
                    best_overlap,
                    4,
                ),
            })

    if (
        document_required
        and not claim_source_map
        and not missing_citations
    ):
        missing_citations.append(
            answer[:180]
        )

    blocking = bool(
        unknown_ids
        or missing_citations
        or unsupported_claims
        or (
            document_required
            and conflicts
        )
    )

    if blocking:

        if conflicts:

            conflict_ids = sorted({
                evidence_id
                for item in conflicts
                for evidence_id in item["evidence_ids"]
            })

            refs = " ".join(
                f"[{item}]"
                for item in conflict_ids
            )

            safe_response = (
                "Retrieved documents mein kuch numeric/date "
                "details aapas mein match nahi kar rahi hain. "
                "Main bina authoritative source/version confirm "
                f"kiye ek value ko sahi nahi bataungi. References: {refs}"
            )

        else:

            valid_refs = " ".join(
                f"[{item}]"
                for item in sorted(known)
            )

            suffix = (
                f" Available references: {valid_refs}"
                if valid_refs else ""
            )

            safe_response = (
                "Generated answer ke sabhi factual claims ko "
                "retrieved document evidence se reliably map "
                "nahi kar payi. Isliye unsupported details "
                "provide nahi karungi."
                + suffix
            )

    else:
        safe_response = answer

    average_support = (
        sum(support_scores) / len(support_scores)
        if support_scores
        else 0.0
    )

    best_retrieval_score = max(
        (
            float(item.get("retrieval_score") or 0.0)
            for item in evidence
        ),
        default=0.0,
    )

    if not evidence:
        confidence = "none"

    elif (
        average_support >= 0.30
        and best_retrieval_score >= 0.55
    ):
        confidence = "high_heuristic"

    elif average_support >= 0.12:
        confidence = "medium_heuristic"

    else:
        confidence = "low_heuristic"

    if conflicts and document_required:
        status = "conflict_detected"

    elif blocking:
        status = "blocked_unverified_claims"

    elif document_required:
        status = "verified_lexical_support"

    elif used_ids:
        status = "citations_validated"

    else:
        status = "not_required"

    return {
        "status": status,
        "blocking": blocking,
        "confidence": confidence,
        "claim_source_map": claim_source_map[:30],
        "unknown_citations": unknown_ids,
        "missing_citations": missing_citations[:10],
        "unsupported_claims": unsupported_claims[:10],
        "possible_conflicts": conflicts,
        "average_lexical_support": round(
            average_support,
            4,
        ),
        "evidence_count": len(evidence),
        "used_citations": sorted(used_ids),
        "untrusted_instruction_evidence_ids": [
            item["evidence_id"]
            for item in evidence
            if item.get(
                "contains_instruction_like_text"
            )
        ],
        "safe_response": safe_response,
    }