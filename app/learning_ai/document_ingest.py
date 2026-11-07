"""Ingest supported documents into PostgreSQL and index their chunks."""
from __future__ import annotations

import codecs
import csv
import hashlib
import json
import mimetypes
import re
import unicodedata

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from html.parser import HTMLParser
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.core.config import settings
from app.learning_ai.learning_store import LearningStore, _vector_literal


MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_DOCUMENT_CHUNKS = 500
DOCUMENT_TRASH_RETENTION_DAYS = 30
MAX_DOCUMENT_PURGE_BATCH = 100
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 160
EMBEDDING_BATCH_SIZE = 8

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".htm",
}

MAX_EXTRACTED_CHARS = 500_000
MAX_STRUCTURED_RECORDS = 10_000
MAX_JSON_DEPTH = 32


@dataclass(frozen=True)
class TextSegment:
    text: str
    page_number: int | None = None
    section: str = ""


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    page_number: int | None
    section: str
    metadata: dict[str, Any] | None = None

@dataclass(frozen=True)
class ExtractionResult:
    segments: list[TextSegment]
    page_count: int | None
    method: str
    status: str
    warnings: tuple[str, ...]
    metadata: dict[str, Any]
    source_encoding: str | None

_PLAYBOOK_RULE_TAG_PATTERN = re.compile(
    r"\bquick behaviour card\s*[—–-]\s*`"
    r"([a-z][a-z0-9_.-]*)`",
    re.IGNORECASE,
)


def _extract_playbook_rule_tags(
    section: str,
) -> tuple[str, ...]:
    """Extract stable identifiers from a Playbook section heading."""
    match = _PLAYBOOK_RULE_TAG_PATTERN.search(
        str(section or "")
    )

    if not match:
        return ()

    return (match.group(1).casefold(),)


_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?।])\s+"
)

_LIST_LINE_RE = re.compile(
    r"^\s*(?:[-*+•▪◦]\s+|\d+[.)]\s+)"
)

_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


def _classify_chunk_block(text: str) -> str:
    """Identify prose, code, tables, lists and structured records."""

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return "prose"

    if any(
        line.startswith(("```", "~~~"))
        for line in lines
    ):
        return "code"

    markdown_table_rows = sum(
        1
        for line in lines
        if line.startswith("|") and line.endswith("|")
    )

    labelled_table_rows = sum(
        1
        for line in lines
        if re.match(
            r"^(?:header|row\s+\d+)\s*:",
            line,
            re.IGNORECASE,
        )
    )

    table_heading = bool(
        re.match(
            r"^table\s+\d+\s*:",
            lines[0],
            re.IGNORECASE,
        )
    )

    html_table_rows = sum(
        1
        for line in lines
        if " | " in line
    )

    if (
        table_heading
        or markdown_table_rows >= 2
        or labelled_table_rows >= 2
        or html_table_rows >= 2
        or any(
            _TABLE_SEPARATOR_RE.match(line)
            for line in lines
        )
    ):
        return "table"

    if all(
        re.match(
            r"^csv record\s+\d+\s*:",
            line,
            re.IGNORECASE,
        )
        for line in lines
    ):
        return "record"

    if all(_LIST_LINE_RE.match(line) for line in lines):
        return "list"

    return "prose"


def _split_by_words(
    text: str,
    max_chars: int = CHUNK_SIZE,
) -> list[str]:
    """Hard fallback when a sentence or structural line is too large."""
    words = text.split()

    if not words:
        return []

    pieces: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            for offset in range(0, len(word), max_chars):
                pieces.append(word[offset:offset + max_chars])

            continue

        candidate = f"{current} {word}".strip()

        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def _split_block_into_units(
    block: str,
    content_kind: str,
) -> list[tuple[str, str]]:
    """
    Split oversized blocks at natural boundaries.

    The second value describes the boundary before that unit.
    """
    block = block.strip()

    if not block:
        return []

    if len(block) <= CHUNK_SIZE:
        return [(block, "paragraph")]

    if content_kind in {"code", "table", "list", "record"}:
        lines = [
            line.rstrip()
            for line in block.splitlines()
            if line.strip()
        ]

        source_units = lines if len(lines) > 1 else [block]
        result: list[tuple[str, str]] = []

        for line in source_units:
            if len(line) <= CHUNK_SIZE:
                result.append((line, "line"))
            else:
                result.extend(
                    (piece, "hard_limit")
                    for piece in _split_by_words(line)
                )

        return result

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(block)
        if sentence.strip()
    ]

    result: list[tuple[str, str]] = []

    for index, sentence in enumerate(sentences):
        boundary = "paragraph" if index == 0 else "sentence"

        if len(sentence) <= CHUNK_SIZE:
            result.append((sentence, boundary))
        else:
            result.extend(
                (piece, "hard_limit")
                for piece in _split_by_words(sentence)
            )

    return result


def _join_chunk_parts(
    parts: list[str],
    kinds: set[str],
) -> str:
    """Keep table, list and code lines together."""
    if kinds in (
        {"code"},
        {"table"},
        {"list"},
        {"record"},
    ):
        separator = "\n"
    else:
        separator = "\n\n"

    return separator.join(parts).strip()


def _make_chunk_overlap(text: str) -> str:
    """Take a small sentence-aware tail for prose continuity."""
    if not text:
        return ""

    if len(text) <= CHUNK_OVERLAP:
        return text.strip()

    tail = text[-CHUNK_OVERLAP:]

    boundary_positions = [
        tail.rfind(". "),
        tail.rfind("? "),
        tail.rfind("! "),
        tail.rfind("। "),
    ]

    boundary = max(boundary_positions)

    if boundary >= 0:
        tail = tail[boundary + 2:]
    else:
        first_space = tail.find(" ")
        if first_space >= 0:
            tail = tail[first_space + 1:]

    return tail.strip()


def _pack_chunk_units(
    units: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """Pack structural units into bounded chunks."""
    packed: list[tuple[str, dict[str, Any]]] = []

    current_parts: list[str] = []
    current_kinds: set[str] = set()
    current_paragraph_ids: set[int] = set()
    current_unit_count = 0
    current_overlap_chars = 0
    current_boundary_before = "group_start"

    def flush(boundary_after: str) -> None:
        if not current_parts:
            return

        chunk_text = _join_chunk_parts(
            current_parts,
            current_kinds,
        )

        if not chunk_text:
            return

        content_kind = (
            next(iter(current_kinds))
            if len(current_kinds) == 1
            else "mixed"
        )

        metadata = {
            "strategy": "structure_aware_v1",
            "content_kind": content_kind,
            "boundary_before": current_boundary_before,
            "boundary_after": boundary_after,
            "paragraph_count": len(current_paragraph_ids),
            "unit_count": current_unit_count,
            "char_count": len(chunk_text),
            "estimated_tokens": max(
                1,
                (len(chunk_text) + 3) // 4,
            ),
            "overlap_chars": current_overlap_chars,
        }

        packed.append((chunk_text, metadata))

    for unit in units:
        unit_text = str(unit["text"]).strip()

        if not unit_text:
            continue

        unit_kind = str(unit["kind"])
        unit_boundary = str(unit["boundary"])
        unit_paragraph = int(unit["paragraph_id"])

        # Do different structures ko ek chunk mein mix mat karo.
        if current_parts and current_kinds != {unit_kind}:
            flush("structure_boundary")

            current_parts = []
            current_kinds = set()
            current_paragraph_ids = set()
            current_unit_count = 0
            current_overlap_chars = 0
            current_boundary_before = unit_boundary

        separator = (
            "\n"
            if current_kinds == {unit_kind}
            and unit_kind in {"code", "table", "list"}
            else "\n\n"
        )

        projected_size = (
            len(_join_chunk_parts(current_parts, current_kinds))
            + len(separator)
            + len(unit_text)
            if current_parts
            else len(unit_text)
        )

        if current_parts and projected_size > CHUNK_SIZE:
            previous_text = _join_chunk_parts(
                current_parts,
                current_kinds,
            )

            # Overlap is used only for ordinary prose.
            carry = (
                _make_chunk_overlap(previous_text)
                if current_kinds == {"prose"}
                else ""
            )

            flush("size_limit")

            current_parts = [carry] if carry else []
            current_kinds = {"prose"} if carry else set()
            current_paragraph_ids = set()
            current_unit_count = 0
            current_overlap_chars = len(carry)

            current_boundary_before = (
                f"overlap_context+{unit_boundary}"
                if carry
                else unit_boundary
            )

            separator = (
                "\n"
                if current_kinds == {unit_kind}
                and unit_kind in {"code", "table", "list"}
                else "\n\n"
            )

            projected_size = (
                len(_join_chunk_parts(current_parts, current_kinds))
                + len(separator)
                + len(unit_text)
                if current_parts
                else len(unit_text)
            )

            # If overlap leaves insufficient room, discard the overlap.
            if current_parts and projected_size > CHUNK_SIZE:
                current_parts = []
                current_kinds = set()
                current_paragraph_ids = set()
                current_unit_count = 0
                current_overlap_chars = 0
                current_boundary_before = unit_boundary

        if (
            not current_parts
            and current_unit_count == 0
            and current_boundary_before == "group_start"
        ):
            current_boundary_before = "group_start"

        current_parts.append(unit_text)
        current_kinds.add(unit_kind)
        current_paragraph_ids.add(unit_paragraph)
        current_unit_count += 1

    flush("group_end")

    return packed


def _build_chunks(
    segments: list[TextSegment],
) -> list[DocumentChunk]:
    """
    Structure-aware chunking.

    Never combines different source pages or section groups.
    """
    groups: list[
        tuple[int | None, str, list[str]]
    ] = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        key = (segment.page_number, segment.section)

        if groups and groups[-1][:2] == key:
            groups[-1][2].append(text)
        else:
            groups.append(
                (segment.page_number, segment.section, [text])
            )

    chunks: list[DocumentChunk] = []

    for page_number, section, texts in groups:
        combined = "\n\n".join(texts)

        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(
                r"\n[ \t]*\n+",
                combined,
            )
            if paragraph.strip()
        ]

        units: list[dict[str, Any]] = []

        for paragraph_id, paragraph in enumerate(paragraphs):
            content_kind = _classify_chunk_block(paragraph)

            for unit_text, boundary in _split_block_into_units(
                paragraph,
                content_kind,
            ):
                units.append({
                    "text": unit_text,
                    "kind": content_kind,
                    "boundary": boundary,
                    "paragraph_id": paragraph_id,
                })

        for chunk_text, metadata in _pack_chunk_units(units):
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    page_number=page_number,
                    section=section,
                    metadata=metadata,
                )
            )

    return chunks


def _chunking_summary(
    chunks: list[DocumentChunk],
) -> dict[str, Any]:
    """Build a compact summary for ingestion output."""
    lengths = [len(chunk.text) for chunk in chunks]
    content_kinds: dict[str, int] = {}
    estimated_tokens = 0

    for chunk in chunks:
        metadata = chunk.metadata or {}

        kind = str(
            metadata.get("content_kind", "unknown")
        )

        content_kinds[kind] = content_kinds.get(kind, 0) + 1

        estimated_tokens += int(
            metadata.get("estimated_tokens", 0) or 0
        )

    return {
        "strategy": "structure_aware_v1",
        "total_chunks": len(chunks),
        "content_kinds": content_kinds,
        "average_chars": (
            round(sum(lengths) / len(lengths))
            if lengths else 0
        ),
        "max_chars": max(lengths, default=0),
        "estimated_tokens_total": estimated_tokens,
    }

def _extract_document(
    path: Path,
    raw: bytes,
) -> ExtractionResult:
    """Extract structured, source-aware text from supported documents."""

    extension = path.suffix.casefold()

    segments: list[TextSegment] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}
    page_count: int | None = None
    source_encoding: str | None = None

    # -------------------------------------------------
    # PDF
    # -------------------------------------------------

    if extension == ".pdf":

        try:
            reader = PdfReader(
                BytesIO(raw),
                strict=False,
            )

        except Exception as exc:
            raise ValueError(
                f"PDF container could not be opened: {type(exc).__name__}"
            ) from exc

        if reader.is_encrypted:
            raise ValueError(
                "Encrypted PDF is not supported."
            )

        page_count = len(reader.pages)
        pages_with_text = 0
        pages_without_text = 0
        layout_fallbacks = 0

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:
                try:
                    text = page.extract_text(
                        extraction_mode="layout"
                    ) or ""

                except (TypeError, NotImplementedError):
                    text = page.extract_text() or ""
                    layout_fallbacks += 1

                except Exception:
                    text = page.extract_text() or ""
                    layout_fallbacks += 1

            except Exception:
                text = ""
                layout_fallbacks += 1

            text = _clean_extracted_text(text).strip()

            if not text:
                pages_without_text += 1
                continue

            pages_with_text += 1

            segments.append(
                TextSegment(
                    text=text,
                    page_number=page_number,
                    section=f"Page {page_number}",
                )
            )

        if pages_without_text:
            warnings.append(
                f"pages_without_selectable_text:{pages_without_text}"
            )

        if layout_fallbacks:
            warnings.append(
                f"pdf_layout_fallback_count:{layout_fallbacks}"
            )

        if not segments:
            raise ValueError(
                "PDF has no selectable text. It may be scanned; "
                "OCR is not available in the current extraction engine."
            )

        if pages_without_text:
            warnings.append("ocr_may_be_required_for_missing_pages")

        metadata = {
            "total_pages": page_count,
            "pages_with_text": pages_with_text,
            "pages_without_text": pages_without_text,
            "layout_fallbacks": layout_fallbacks,
        }

        method = "pypdf_layout"

    # -------------------------------------------------
    # DOCX
    # -------------------------------------------------

    elif extension == ".docx":

        try:
            document = Document(BytesIO(raw))

        except Exception as exc:
            raise ValueError(
                f"DOCX could not be opened: {type(exc).__name__}"
            ) from exc

        current_section = path.stem
        paragraph_count = 0
        table_count = 0
        header_count = 0
        footer_count = 0

        seen_story_text: set[str] = set()

        def add_paragraph(
            paragraph: Paragraph,
            section_name: str,
            *,
            is_list_item: bool = False,
        ) -> None:
            nonlocal paragraph_count

            text = _clean_extracted_text(
                paragraph.text
            ).strip()

            if not text:
                return

            if is_list_item and not _LIST_LINE_RE.match(text):
                text = f"- {text}"

            paragraph_count += 1

            segments.append(
                TextSegment(
                    text=text,
                    section=section_name,
                )
            )

        def add_table(
            table: Table,
            section_name: str,
        ) -> None:
            nonlocal table_count

            table_count += 1
            rows: list[str] = []

            for row_index, row in enumerate(table.rows):

                cells: list[str] = []
                seen_cells: set[int] = set()

                for cell in row.cells:
                    identity = id(cell._tc)

                    if identity in seen_cells:
                        continue

                    seen_cells.add(identity)

                    cell_text = _clean_extracted_text(
                        cell.text
                    ).strip()

                    if cell_text:
                        cells.append(
                            cell_text.replace("\n", " / ")
                        )
                    else:
                        cells.append("")

                if any(cells):
                    row_label = (
                        "Header"
                        if row_index == 0
                        else f"Row {row_index}"
                    )

                    rows.append(
                        f"{row_label}: " + " | ".join(cells)
                    )

            if rows:
                segments.append(
                    TextSegment(
                        text=(
                            f"Table {table_count}:\n"
                            + "\n".join(rows)
                        ),
                        section=section_name,
                    )
                )

        # Preserve body paragraphs and tables in document order.
        for block in document.iter_inner_content():

            if isinstance(block, Paragraph):

                style_name = ""

                try:
                    style_name = (
                        block.style.name.casefold()
                    )
                except (AttributeError, KeyError):
                    pass

                if style_name.startswith(
                    ("heading", "title", "subtitle")
                ):
                    heading = _clean_extracted_text(
                        block.text
                    ).strip()

                    if heading:
                        current_section = heading

                add_paragraph(
                    block,
                    current_section,
                    is_list_item=style_name.startswith(
                        (
                            "list bullet",
                            "list number",
                            "list paragraph",
                        )
                    ),
                )

            elif isinstance(block, Table):
                add_table(
                    block,
                    current_section,
                )

        # Extract section headers and footers separately.
        for section_index, section in enumerate(
            document.sections,
            start=1,
        ):

            for story_name, story in (
                ("Header", section.header),
                ("Footer", section.footer),
                ("First page header", section.first_page_header),
                ("First page footer", section.first_page_footer),
            ):

                story_blocks = []

                for paragraph in story.paragraphs:
                    text = _clean_extracted_text(
                        paragraph.text
                    ).strip()

                    if text:
                        story_blocks.append(text)

                for table in story.tables:
                    for row in table.rows:
                        cells = [
                            _clean_extracted_text(cell.text).strip()
                            for cell in row.cells
                        ]

                        if any(cells):
                            story_blocks.append(
                                " | ".join(cells)
                            )

                story_text = "\n".join(story_blocks).strip()

                if (
                    not story_text
                    or story_text in seen_story_text
                ):
                    continue

                seen_story_text.add(story_text)

                segments.append(
                    TextSegment(
                        text=story_text,
                        section=(
                            f"{story_name} {section_index}"
                        ),
                    )
                )

                if story_name == "Header":
                    header_count += 1

                elif story_name == "Footer":
                    footer_count += 1

        image_count = len(document.inline_shapes)

        if image_count:
            warnings.append(
                f"embedded_images_not_ocr_processed:{image_count}"
            )

        if not segments:
            raise ValueError(
                "DOCX contains no extractable text."
            )

        metadata = {
            "paragraph_count": paragraph_count,
            "table_count": table_count,
            "header_blocks": header_count,
            "footer_blocks": footer_count,
            "embedded_images": image_count,
            "section_count": len(document.sections),
        }

        method = "python_docx_structured"

    # -------------------------------------------------
    # TEXT AND MARKDOWN
    # -------------------------------------------------

    elif extension in {".txt", ".md"}:

        text, source_encoding = _decode_document_text(raw)

        if extension == ".txt":

            for paragraph in re.split(
                r"\n\s*\n",
                text,
            ):
                paragraph = _clean_extracted_text(
                    paragraph
                ).strip()

                if paragraph:
                    segments.append(
                        TextSegment(
                            text=paragraph,
                            section=path.stem,
                        )
                    )

            method = "utf_text_paragraphs"

        else:

            current_section = path.stem
            paragraph: list[str] = []
            in_fence = False
            fence_marker = ""

            def flush_markdown() -> None:
                if not paragraph:
                    return

                content = "\n".join(
                    paragraph
                ).strip()

                if content:
                    segments.append(
                        TextSegment(
                            text=content,
                            section=current_section,
                        )
                    )

                paragraph.clear()

            for line in text.splitlines():

                stripped = line.strip()

                fence_match = re.match(
                    r"^(```+|~~~+)",
                    stripped,
                )

                if fence_match:

                    marker = fence_match.group(1)

                    if not in_fence:
                        flush_markdown()
                        in_fence = True
                        fence_marker = marker[0]

                    elif marker.startswith(fence_marker * 3):
                        paragraph.append(line)
                        in_fence = False
                        fence_marker = ""
                        flush_markdown()
                        continue

                    paragraph.append(line)
                    continue

                if in_fence:
                    paragraph.append(line)
                    continue

                heading_match = re.match(
                    r"^(#{1,6})\s+(.+?)\s*#*\s*$",
                    line,
                )

                if heading_match:

                    flush_markdown()

                    heading_level = len(
                        heading_match.group(1)
                    )

                    current_section = (
                        f"H{heading_level}: "
                        f"{heading_match.group(2).strip()}"
                    )

                    segments.append(
                        TextSegment(
                            text=heading_match.group(2).strip(),
                            section=current_section,
                        )
                    )

                elif not stripped:
                    flush_markdown()

                else:
                    paragraph.append(line)

            flush_markdown()

            method = "markdown_structured"

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------

    elif extension == ".csv":

        text, source_encoding = _decode_document_text(raw)

        sample = text[:8192]

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|",
            )

        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(
            StringIO(text),
            dialect,
        )

        header: list[str] | None = None
        record_count = 0
        column_count = 0

        for row_index, row in enumerate(reader):

            if row_index >= MAX_STRUCTURED_RECORDS:
                warnings.append(
                    "csv_record_limit_reached"
                )
                break

            cells = [
                _clean_extracted_text(str(value)).strip()
                for value in row
            ]

            if not any(cells):
                continue

            if header is None:
                header = cells
                column_count = len(header)

                # Header alone is not a useful content record.
                continue

            values = []

            for column_index, value in enumerate(cells):

                column_name = (
                    header[column_index]
                    if column_index < len(header)
                    and header[column_index]
                    else f"Column {column_index + 1}"
                )

                if value:
                    values.append(
                        f"{column_name}: {value}"
                    )

            if values:
                record_count += 1

                segments.append(
                    TextSegment(
                        text=(
                            f"CSV record {record_count}: "
                            + "; ".join(values)
                        ),
                        section="CSV data",
                    )
                )

        if header is None:
            raise ValueError(
                "CSV contains no readable header row."
            )

        metadata = {
            "record_count": record_count,
            "column_count": column_count,
            "delimiter": getattr(dialect, "delimiter", ","),
        }

        method = "python_csv_structured"

    # -------------------------------------------------
    # JSON
    # -------------------------------------------------

    elif extension == ".json":

        text, source_encoding = _decode_document_text(raw)

        try:
            data = json.loads(text)

        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "JSON document is malformed."
            ) from exc

        stack = [("$", data, 0)]
        record_count = 0

        while stack:

            current_path, value, depth = stack.pop()

            if depth > MAX_JSON_DEPTH:
                warnings.append(
                    "json_maximum_depth_reached"
                )
                continue

            if isinstance(value, dict):

                for key, child in reversed(
                    list(value.items())
                ):
                    stack.append((
                        f"{current_path}.{str(key)[:100]}",
                        child,
                        depth + 1,
                    ))

            elif isinstance(value, list):

                for index, child in reversed(
                    list(enumerate(value))
                ):
                    stack.append((
                        f"{current_path}[{index}]",
                        child,
                        depth + 1,
                    ))

            else:

                if value is None:
                    rendered = "null"

                elif isinstance(value, str):
                    rendered = value

                elif isinstance(value, bool):
                    rendered = str(value).lower()

                else:
                    rendered = str(value)

                rendered = _clean_extracted_text(
                    rendered
                ).strip()

                if not rendered:
                    continue

                record_count += 1

                if record_count > MAX_STRUCTURED_RECORDS:
                    warnings.append(
                        "json_record_limit_reached"
                    )
                    break

                segments.append(
                    TextSegment(
                        text=(
                            f"{current_path}: {rendered}"
                        ),
                        section="JSON data",
                    )
                )

        metadata = {
            "scalar_value_count": min(
                record_count,
                MAX_STRUCTURED_RECORDS,
            ),
        }

        method = "python_json_structured"

    # -------------------------------------------------
    # HTML
    # -------------------------------------------------

    elif extension in {".html", ".htm"}:

        text, source_encoding = _decode_document_text(raw)

        parser = _StructuredHTMLParser()

        try:
            parser.feed(text)
            parser.close()

        except Exception as exc:
            raise ValueError(
                "HTML structure could not be parsed."
            ) from exc

        segments = parser.segments

        metadata = {
            "extracted_block_count": parser.block_count,
            "scripts_excluded": True,
            "styles_excluded": True,
        }

        method = "python_html_structured"

    else:
        raise ValueError(
            f"Unsupported extension: {extension}"
        )

    # -------------------------------------------------
    # FINAL EXTRACTION VALIDATION
    # -------------------------------------------------

    cleaned_segments = [
        TextSegment(
            text=_clean_extracted_text(item.text).strip(),
            page_number=item.page_number,
            section=_clean_extracted_text(
                item.section
            ).strip(),
        )
        for item in segments
        if item.text and item.text.strip()
    ]

    if not cleaned_segments:
        raise ValueError(
            "No usable text could be extracted."
        )

    total_characters = sum(
        len(item.text)
        for item in cleaned_segments
    )

    if total_characters > MAX_EXTRACTED_CHARS:
        raise ValueError(
            "Extracted text exceeds the 500,000-character limit. "
            "Split the source into smaller documents."
        )
    
    metadata.update({
        "segment_count": len(cleaned_segments),
        "extracted_characters": total_characters,
    })

    status = (
        "partial"
        if warnings
        else "complete"
    )

    return ExtractionResult(
        segments=cleaned_segments,
        page_count=page_count,
        method=method,
        status=status,
        warnings=tuple(warnings),
        metadata=metadata,
        source_encoding=source_encoding,
    )


def _extract_segments(
    path: Path,
    raw: bytes,
) -> tuple[list[TextSegment], int | None]:
    """Backward-compatible extraction interface."""

    result = _extract_document(path, raw)

    return (
        result.segments,
        result.page_count,
    )

def _clean_extracted_text(value: str | None) -> str:
    """Remove NUL and invalid control characters from extracted document text."""
    if not value:
        return ""

    return "".join(
        character
        for character in value
        if character in "\t\n\r"
        or unicodedata.category(character) not in {"Cc", "Cs"}
    )

def _decode_document_text(raw: bytes) -> tuple[str, str]:
    """Decode supported Unicode text without lossy character replacement."""

    if raw.startswith(codecs.BOM_UTF8):
        encoding = "utf-8-sig"

    elif raw.startswith(
        (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)
    ):
        encoding = "utf-16"

    else:
        encoding = "utf-8"

    try:
        text = raw.decode(encoding)

    except UnicodeDecodeError as exc:
        raise ValueError(
            "Document encoding is unsupported or invalid. "
            "Save the source as UTF-8 or UTF-16."
        ) from exc

    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
    )

    return text, encoding

class _StructuredHTMLParser(HTMLParser):
    """Extract readable HTML blocks while excluding executable content."""

    BLOCK_TAGS = {
        "article",
        "blockquote",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "section",
        "td",
        "th",
        "tr",
    }

    SKIP_TAGS = {
        "head",
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "template",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.segments: list[TextSegment] = []
        self.section = "HTML document"
        self.buffer: list[str] = []
        self.heading_tag: str | None = None
        self.heading_buffer: list[str] = []
        self.skip_depth = 0
        self.in_table_row = False
        self.block_count = 0

    def _flush(self) -> None:
        content = _clean_extracted_text(
            "".join(self.buffer)
        ).strip()

        self.buffer.clear()

        if not content:
            return

        self.segments.append(
            TextSegment(
                text=content,
                section=self.section,
            )
        )

        self.block_count += 1

    def handle_starttag(self, tag: str, attrs: list) -> None:
        tag = tag.casefold()

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth:
            return

        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self.heading_tag = tag
            self.heading_buffer.clear()
            return

        elif tag in {"tr", "p", "div", "section", "article", "li"}:
            self._flush()

        elif tag == "br":
            self._flush()

        elif tag in {"td", "th"} and self.in_table_row:
            if self.buffer and self.buffer[-1] != " | ":
                self.buffer.append(" | ")

        if tag == "tr":
            self.in_table_row = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()

        if tag in self.SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return

        if self.skip_depth:
            return

        if self.heading_tag == tag:
            heading = _clean_extracted_text(
                "".join(self.heading_buffer)
            ).strip()

            self.heading_buffer.clear()
            self.heading_tag = None

            if heading:
                self.section = heading[:200]

                self.segments.append(
                    TextSegment(
                        text=heading,
                        section=self.section,
                    )
                )

                self.block_count += 1

            return

        if tag in {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
        }:
            self._flush()

        if tag == "tr":
            self.in_table_row = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return

        cleaned = _clean_extracted_text(data)

        if cleaned:
            if self.heading_tag is not None:
                self.heading_buffer.append(cleaned)
            else:
                self.buffer.append(cleaned)

    def close(self) -> None:
        super().close()

        if self.heading_tag is not None:
            heading = _clean_extracted_text(
                "".join(self.heading_buffer)
            ).strip()

            self.heading_buffer.clear()
            self.heading_tag = None

            if heading:
                self.section = heading[:200]

                self.segments.append(
                    TextSegment(
                        text=heading,
                        section=self.section,
                    )
                )

                self.block_count += 1

        self._flush()

def _source_identity(path: Path) -> str:
    """Stable, privacy-conscious identity for one canonical local path."""
    normalized = str(path).replace("\\", "/").strip().lower()

    # MD5 is used only as a stable source identifier, never for integrity.
    return hashlib.md5(
        normalized.encode("utf-8")
    ).hexdigest()


def _extracted_content_hash(
    segments: list[TextSegment],
) -> str:
    content = "\n".join(
        (
            f"PAGE:{segment.page_number or ''}\n"
            f"SECTION:{segment.section}\n"
            f"{segment.text}"
        )
        for segment in segments
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

def _attach_stable_chunk_ids(
    chunks: list[DocumentChunk],
    source_key: str,
) -> list[DocumentChunk]:
    """Attach deterministic IDs that survive unchanged source re-ingestion."""
    occurrences: dict[str, int] = {}
    identified: list[DocumentChunk] = []

    for chunk in chunks:
        content_hash = hashlib.sha256(
            chunk.text.encode("utf-8")
        ).hexdigest()

        identity_base = "\x1f".join((
            source_key,
            str(chunk.page_number or ""),
            chunk.section,
            content_hash,
        ))

        occurrence = occurrences.get(identity_base, 0)
        occurrences[identity_base] = occurrence + 1

        stable_chunk_id = hashlib.sha256(
            f"{identity_base}\x1f{occurrence}".encode("utf-8")
        ).hexdigest()

        metadata = dict(chunk.metadata or {})
        metadata["stable_chunk_id"] = stable_chunk_id

        identified.append(
            DocumentChunk(
                text=chunk.text,
                page_number=chunk.page_number,
                section=chunk.section,
                metadata=metadata,
            )
        )

    return identified

class DocumentIngestor:
    """Extract, chunk, embed, and persist reference documents."""

    def __init__(
        self,
        store: LearningStore,
        *,
        allow_mutations: bool = False,
    ) -> None:
        self.store = store
        self.allow_mutations = bool(allow_mutations)

    def _require_mutation_permission(self) -> None:
        if not self.allow_mutations:
            raise PermissionError(
                "Document mutation is not authorized "
                "for this application context."
            )

    @staticmethod
    def _validate_document_id(
        document_id: int,
    ) -> int | None:
        if isinstance(document_id, bool):
            return None

        try:
            value = int(document_id)
        except (TypeError, ValueError):
            return None

        if value <= 0:
            return None

        return value

    @staticmethod
    def _normalize_actor(actor: str) -> str:
        actor = re.sub(
            r"[^a-zA-Z0-9_.:@-]",
            "_",
            str(actor or "local_owner").strip(),
        )

        return actor[:100] or "local_owner"

    def _acquire_document_source_lock(
        self,
        connection: Any,
        source_key: str,
        document_id: int,
    ) -> None:
        lock_key = (
            source_key
            if source_key
            else f"legacy-document:{document_id}"
        )

        connection.execute(
            """
            SELECT pg_advisory_xact_lock(
                hashtext(%s)::bigint
            )
            """,
            (
                f"zara-document-source:{lock_key}",
            ),
        )

    @staticmethod
    def _current_index_is_complete(
        connection: Any,
        document: dict[str, Any],
        chunks: list[DocumentChunk],
        *,
        category: str,
        language: str,
        model: str,
        model_version: str,
        dimensions: int,
        format_version: str,
        embedding_text_hashes: list[str],
        file_name: str,
    ) -> bool:

        if (
            document["ingestion_status"] != "ready"
            or document["category"] != category
            or document["language"] != language
            or document["file_name"] != file_name
        ):
            return False

        rows = connection.execute("""
            SELECT
                chunk_index,
                content,
                page_number,
                section,
                embedding,
                embedding_model,
                embedding_model_version,
                embedding_dimensions,
                embedding_format_version,
                embedding_text_hash,
                metadata
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index
        """, (
            document["id"],
        )).fetchall()

        if len(rows) != len(chunks):
            return False

        for index, (row, chunk) in enumerate(zip(rows, chunks)):

            if (
                int(row["chunk_index"]) != index
                or row["content"] != chunk.text
                or row["page_number"] != chunk.page_number
                or str(row["section"] or "") != chunk.section
                or row["embedding"] is None
                or row["embedding_model"] != model
                or row["embedding_model_version"] != model_version
                or int(row["embedding_dimensions"] or 0) != dimensions
                or row["embedding_format_version"] != format_version
                or row["embedding_text_hash"] != embedding_text_hashes[index]
            ):
                return False

            stored_metadata = row["metadata"] or {}
            stored_chunking = stored_metadata.get("chunking", {})

            expected_chunk_id = str(
                (chunk.metadata or {}).get("stable_chunk_id", "")
            )

            if (
                not isinstance(stored_chunking, dict)
                or not expected_chunk_id
                or stored_chunking.get("stable_chunk_id") != expected_chunk_id
            ):
                return False

        return True

    @staticmethod
    def _refresh_chunking_metadata(
        connection: Any,
        document_id: int,
        chunks: list[DocumentChunk],
    ) -> None:
        """Update chunking metadata without regenerating embeddings."""

        for index, chunk in enumerate(chunks):
            chunking_json = json.dumps(
                dict(chunk.metadata or {}),
                ensure_ascii=False,
            )

            connection.execute(
                """
                UPDATE document_chunks
                SET metadata = metadata
                    || jsonb_build_object('chunking', %s::jsonb)
                WHERE document_id = %s
                  AND chunk_index = %s
                """,
                (
                    chunking_json,
                    document_id,
                    index,
                ),
            )

    def ingest_file(
        self,
        file_path: str | Path,
        *,
        category: str = "general",
        language: str = "unknown",
        force: bool = False,
    ) -> dict[str, Any]:
        """Ingest one local document as an immutable source version."""

        self._require_mutation_permission()

        path = Path(file_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                "Selected document does not exist."
            )

        extension = path.suffix.casefold()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document extension: {extension}. "
                "Use PDF, DOCX, TXT, Markdown, CSV, JSON or HTML."
            )

        if not isinstance(force, bool):
            raise ValueError("force must be a boolean.")

        category = str(category).strip()[:100] or "general"
        language = str(language).strip()[:50] or "unknown"

        file_stat = path.stat()
        raw = path.read_bytes()

        if not raw:
            raise ValueError("The selected document is empty.")

        if len(raw) > MAX_FILE_BYTES:
            raise ValueError(
                "Document exceeds the configured 15 MB limit."
            )

        content_hash = hashlib.sha256(raw).hexdigest()
        source_key = _source_identity(path)

        media_type = (
            mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )

        extraction = _extract_document(path, raw)

        segments = extraction.segments
        page_count = extraction.page_count

        chunks = _attach_stable_chunk_ids(
            _build_chunks(segments),
            source_key,
        )
        chunking_summary = _chunking_summary(chunks)

        if not chunks:
            raise ValueError(
                "No usable text could be extracted from this document."
            )

        if len(chunks) > MAX_DOCUMENT_CHUNKS:
            raise ValueError(
                f"Document creates {len(chunks)} chunks. "
                f"Maximum allowed: {MAX_DOCUMENT_CHUNKS}."
            )

        extracted_hash = _extracted_content_hash(segments)

        embedding_client = self.store.embedding_client
        model = str(embedding_client.model)
        model_version = str(embedding_client.model_version)
        dimensions = int(embedding_client.dimensions)
        format_version = str(embedding_client.format_version)

        is_playbook_file = (
            path.name.casefold() == "hinglish_playbook.md"
        )

        embedding_documents: list[tuple[str, str]] = []
        embedding_text_hashes: list[str] = []

        for chunk in chunks:
            title = f"{path.stem} - {chunk.section or 'General'}"

            body = (
                f"Category: {category}\n"
                f"Language: {language}\n"
                f"Section: {chunk.section or 'General'}\n"
                f"Content:\n{chunk.text}"
            )

            embedding_documents.append((title, body))

            formatted_text = embedding_client.format_document(
                title,
                body,
            )

            embedding_text_hashes.append(
                hashlib.sha256(
                    formatted_text.encode("utf-8")
                ).hexdigest()
            )

        # -------------------------------------------------
        # 1. Fast duplicate check: no embedding call.
        # -------------------------------------------------

        with self.store._connect() as connection:

            existing = connection.execute("""
                SELECT *
                FROM documents
                WHERE source_key = %s
                  AND ingestion_status = 'ready'
                ORDER BY source_version DESC
                LIMIT 1
            """, (
                source_key,
            )).fetchone()

            if (
                existing is not None
                and not force
                and existing["content_sha256"] == content_hash
                and self._current_index_is_complete(
                    connection,
                    existing,
                    chunks,
                    category=category,
                    language=language,
                    model=model,
                    model_version=model_version,
                    dimensions=dimensions,
                    format_version=format_version,
                    embedding_text_hashes=embedding_text_hashes,
                    file_name=path.name,
                )
            ):
                self._refresh_chunking_metadata(
                    connection,
                    int(existing["id"]),
                    chunks,
                )

                return {
                    "status": "already_indexed",
                    "document_id": int(existing["id"]),
                    "source_version": int(existing["source_version"]),
                    "file_name": path.name,
                    "chunks": len(chunks),
                    "chunking": chunking_summary,
                    "pages": page_count,
                }

        # -------------------------------------------------
        # 2. Reuse cached embeddings; generate only missing vectors.
        # Completed batches are committed to the cache independently, so a
        # later retry can resume without repeating those embedding requests.
        # -------------------------------------------------

        vector_literals: list[str | None] = [None] * len(chunks)
        chunk_indexes_by_hash: dict[str, list[int]] = {}

        for index, text_hash in enumerate(embedding_text_hashes):
            chunk_indexes_by_hash.setdefault(text_hash, []).append(index)

        with self.store._connect() as connection:
            cached_rows = connection.execute(
                """
                SELECT
                    embedding_text_hash,
                    embedding::text AS embedding_literal
                FROM document_embedding_cache
                WHERE embedding_text_hash = ANY(%s)
                  AND embedding_model = %s
                  AND embedding_model_version = %s
                  AND embedding_dimensions = %s
                  AND embedding_format_version = %s
                """,
                (
                    list(chunk_indexes_by_hash),
                    model,
                    model_version,
                    dimensions,
                    format_version,
                ),
            ).fetchall()

        cached_by_hash: dict[str, str] = {}

        for cached_row in cached_rows:
            try:
                cached_values = json.loads(
                    str(cached_row["embedding_literal"])
                )

                if not isinstance(cached_values, list):
                    continue

                cached_by_hash[str(cached_row["embedding_text_hash"])] = (
                    _vector_literal(
                        [float(value) for value in cached_values]
                    )
                )

            except (TypeError, ValueError, json.JSONDecodeError):
                # A malformed cache row is ignored and regenerated below.
                continue

        for text_hash, cached_literal in cached_by_hash.items():
            for chunk_index in chunk_indexes_by_hash[text_hash]:
                vector_literals[chunk_index] = cached_literal

        missing_hashes = [
            text_hash
            for text_hash in chunk_indexes_by_hash
            if text_hash not in cached_by_hash
        ]

        generated_embedding_count = 0

        reused_embedding_count = sum(
            len(chunk_indexes_by_hash[text_hash])
            for text_hash in cached_by_hash
        )

        for offset in range(0, len(missing_hashes), EMBEDDING_BATCH_SIZE):
            batch_hashes = missing_hashes[
                offset:offset + EMBEDDING_BATCH_SIZE
            ]

            batch_indexes = [
                chunk_indexes_by_hash[text_hash][0]
                for text_hash in batch_hashes
            ]

            batch_documents = [
                embedding_documents[index]
                for index in batch_indexes
            ]

            batch_vectors = embedding_client.embed_documents(
                batch_documents
            )

            if len(batch_vectors) != len(batch_hashes):
                raise RuntimeError(
                    "Embedding batch returned an incorrect vector count."
                )

            cache_rows: list[tuple[str, str]] = []

            for text_hash, vector in zip(batch_hashes, batch_vectors):
                literal = _vector_literal(vector)

                for chunk_index in chunk_indexes_by_hash[text_hash]:
                    vector_literals[chunk_index] = literal

                cache_rows.append((text_hash, literal))
                generated_embedding_count += 1

            # Commit successful batch results before requesting the next one.
            with self.store._connect() as connection:
                for text_hash, literal in cache_rows:
                    connection.execute(
                        """
                        INSERT INTO document_embedding_cache (
                            embedding_text_hash,
                            embedding_model,
                            embedding_model_version,
                            embedding_dimensions,
                            embedding_format_version,
                            embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (
                            embedding_text_hash,
                            embedding_model,
                            embedding_model_version,
                            embedding_dimensions,
                            embedding_format_version
                        )
                        DO UPDATE SET last_used_at = NOW()
                        """,
                        (
                            text_hash,
                            model,
                            model_version,
                            dimensions,
                            format_version,
                            literal,
                        ),
                    )

        if any(literal is None for literal in vector_literals):
            raise RuntimeError(
                "Embedding cache/index build left one or more chunks "
                "without vectors."
            )

        complete_vector_literals = [
            str(literal) for literal in vector_literals
        ]

        # -------------------------------------------------
        # 3. Atomic source-version transaction.
        # -------------------------------------------------

        with self.store._connect() as connection:

            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                (f"zara-document-source:{source_key}",),
            )

            current = connection.execute("""
                SELECT *
                FROM documents
                WHERE source_key = %s
                  AND ingestion_status = 'ready'
                ORDER BY source_version DESC
                LIMIT 1
                FOR UPDATE
            """, (
                source_key,
            )).fetchone()

            # A second concurrent ingestion may have completed while
            # embeddings were being generated.
            if (
                current is not None
                and not force
                and current["content_sha256"] == content_hash
                and self._current_index_is_complete(
                    connection,
                    current,
                    chunks,
                    category=category,
                    language=language,
                    model=model,
                    model_version=model_version,
                    dimensions=dimensions,
                    format_version=format_version,
                    embedding_text_hashes=embedding_text_hashes,
                    file_name=path.name,
                )
            ):
                self._refresh_chunking_metadata(
                    connection,
                    int(current["id"]),
                    chunks,
                )
                return {
                    "status": "already_indexed",
                    "document_id": int(current["id"]),
                    "source_version": int(current["source_version"]),
                    "file_name": path.name,
                    "chunks": len(chunks),
                    "chunking": chunking_summary,
                    "pages": page_count,
                }

            version_row = connection.execute("""
                SELECT COALESCE(MAX(source_version), 0) + 1 AS next_version
                FROM documents
                WHERE source_key = %s
            """, (
                source_key,
            )).fetchone()

            source_version = int(
                version_row["next_version"]
            )

            previous_document_id = (
                int(current["id"])
                if current is not None
                else None
            )

            # Same binary content at another source is a duplicate,
            # but it retains an independent source record.
            duplicate = connection.execute("""
                SELECT id
                FROM documents
                WHERE content_sha256 = %s
                  AND source_key <> %s
                ORDER BY
                    CASE WHEN ingestion_status = 'ready'
                         THEN 0 ELSE 1 END,
                    updated_at DESC,
                    id DESC
                LIMIT 1
            """, (
                content_hash,
                source_key,
            )).fetchone()

            duplicate_of_document_id = (
                int(duplicate["id"])
                if duplicate is not None
                else None
            )

            # Deactivate previous source version before creating
            # the new ready version. Both actions share one transaction.
            if current is not None:

                connection.execute("""
                    UPDATE documents
                    SET
                        ingestion_status = 'superseded',
                        updated_at = NOW()
                    WHERE id = %s
                      AND ingestion_status = 'ready'
                """, (
                    previous_document_id,
                ))

            document = connection.execute("""
                INSERT INTO documents (
                    source_path,
                    file_name,
                    content_sha256,
                    category,
                    language,
                    ingestion_status,
                    source_key,
                    source_version,
                    supersedes_document_id,
                    duplicate_of_document_id,
                    extracted_text_sha256,
                    file_size_bytes,
                    media_type,
                    source_mtime_ns,
                    extraction_method,
                    extraction_status,
                    extraction_warnings,
                    extraction_metadata,
                    source_encoding,
                    page_count
                )
                VALUES (
                    %s, %s, %s, %s, %s, 'ready',
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s::jsonb, %s::jsonb, %s, %s
                )
                RETURNING id
            """, (
                str(path),
                path.name,
                content_hash,
                category,
                language,
                source_key,
                source_version,
                previous_document_id,
                duplicate_of_document_id,
                extracted_hash,
                len(raw),
                media_type,
                file_stat.st_mtime_ns,
                    extraction.method,
                    extraction.status,
                    json.dumps(
                        list(extraction.warnings),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        extraction.metadata,
                        ensure_ascii=False,
                    ),
                    extraction.source_encoding,
                    extraction.page_count,
            )).fetchone()


            document_id = int(document["id"])

            # Preserve all existing versions; do not DELETE old chunks.
            for index, (chunk, vector) in enumerate(
                zip(chunks, complete_vector_literals)
            ):

                rule_tags = (
                    _extract_playbook_rule_tags(chunk.section)
                    if is_playbook_file
                    else ()
                )

                metadata = {
                    "file_name": path.name,
                    "category": category,
                    "language": language,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "chunk_index": index,
                    "source_key": source_key,
                    "source_version": source_version,
                    "content_kind": (
                        "behavioral_guidance"
                        if is_playbook_file
                        else "document_knowledge"
                    ),
                    "rule_tags": list(rule_tags),
                    "chunking": dict(chunk.metadata or {}),
                }

                if len(rule_tags) == 1:
                    metadata["rule_id"] = rule_tags[0]

                connection.execute("""
                    INSERT INTO document_chunks (
                        document_id,
                        chunk_index,
                        content,
                        page_number,
                        section,
                        metadata,
                        embedding,
                        embedding_model,
                        embedding_model_version,
                        embedding_dimensions,
                        embedding_format_version,
                        embedding_text_hash
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s::jsonb, %s::vector, %s, %s, %s, %s, %s
                    )
                """, (
                    document_id,
                    index,
                    chunk.text,
                    chunk.page_number,
                    chunk.section,
                    json.dumps(metadata, ensure_ascii=False),
                    vector,
                    model,
                    model_version,
                    dimensions,
                    format_version,
                    embedding_text_hashes[index],
                ))

            event_type = (
                "duplicate_content_indexed"
                if duplicate_of_document_id is not None
                else "document_indexed"
            )

            connection.execute("""
                INSERT INTO document_ingestion_events (
                    document_id,
                    previous_document_id,
                    source_key,
                    event_type,
                    details
                )
                VALUES (
                    %s, %s, %s, %s, %s::jsonb
                )
            """, (
                document_id,
                previous_document_id,
                source_key,
                event_type,
                json.dumps({
                    "source_version": source_version,
                    "content_sha256": content_hash,
                    "extracted_text_sha256": extracted_hash,
                    "chunk_count": len(chunks),
                    "embedding_model": model,
                    "embedding_model_version": model_version,
                    "embedding_dimensions": dimensions,
                    "embedding_format_version": format_version,
                    "generated_embeddings": generated_embedding_count,
                    "reused_embeddings": reused_embedding_count,
                    "page_count": page_count,
                    "duplicate_of_document_id": (
                        duplicate_of_document_id
                    ),
                }),
            ))

            if previous_document_id is not None:

                connection.execute("""
                    INSERT INTO document_ingestion_events (
                        document_id,
                        previous_document_id,
                        source_key,
                        event_type,
                        details
                    )
                    VALUES (
                        %s, %s, %s,
                        'document_superseded',
                        %s::jsonb
                    )
                """, (
                    previous_document_id,
                    document_id,
                    source_key,
                    json.dumps({
                        "replacement_document_id": document_id,
                        "replacement_version": source_version,
                    }),
                ))

        return {
            "status": (
                "indexed_duplicate_content"
                if duplicate_of_document_id is not None
                else "indexed"
            ),
            "document_id": document_id,
            "source_version": source_version,
            "supersedes_document_id": previous_document_id,
            "duplicate_of_document_id": duplicate_of_document_id,
            "file_name": path.name,
            "chunks": len(chunks),
            "chunking": chunking_summary,
            "pages": page_count,
            "generated_embeddings": generated_embedding_count,
            "reused_embeddings": reused_embedding_count,
            "embedding_model_version": model_version,
            "extraction": {
                "method": extraction.method,
                "status": extraction.status,
                "warnings": list(extraction.warnings),
                "metadata": extraction.metadata,
                "source_encoding": extraction.source_encoding,
            },
        }

    def list_documents(self, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))

        with self.store._connect() as connection:
            rows = connection.execute("""
                SELECT
                    d.id,
                    NULL::text AS source_path,
                    d.source_version,
                    d.supersedes_document_id,
                    d.duplicate_of_document_id,
                    d.file_name,
                    d.file_size_bytes,
                    d.media_type,
                    d.extraction_status,
                    d.extraction_warnings,
                    d.extraction_metadata,
                    d.page_count,
                    d.category,
                    d.language,
                    d.ingestion_status,
                    d.created_at,
                    d.updated_at,
                    d.deleted_at,
                    d.deleted_by,
                    d.retention_until,
                    d.purged_at,
                    d.integrity_status,
                    d.last_integrity_check_at,
                    (
                        d.ingestion_status = 'superseded'
                    ) AS superseded_version,
                    COUNT(dc.id) AS chunk_count
                FROM documents AS d
                LEFT JOIN document_chunks AS dc
                    ON dc.document_id = d.id
                GROUP BY d.id
                ORDER BY d.updated_at DESC, d.id DESC
                LIMIT %s
            """, (
                limit,
            )).fetchall()

        return [dict(row) for row in rows]

    def archive_document(
        self,
        document_id: int,
        *,
        actor: str = "local_owner",
    ) -> dict[str, Any]:

        if isinstance(document_id, bool):
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        try:
            document_id = int(document_id)
        except (TypeError, ValueError):
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        if document_id <= 0:
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        self._require_mutation_permission()
        actor = self._normalize_actor(actor)

        with self.store._connect() as connection:
            identity = connection.execute("""
                SELECT source_key
                FROM documents
                WHERE id = %s
            """, (
                document_id,
            )).fetchone()

            if identity is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            source_key = str(identity["source_key"] or "")
            lock_key = source_key or f"legacy-document:{document_id}"

            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                (f"zara-document-source:{lock_key}",),
            )

            row = connection.execute("""
                SELECT id, source_key, ingestion_status
                FROM documents
                WHERE id = %s
                FOR UPDATE
            """, (
                document_id,
            )).fetchone()

            if row["ingestion_status"] == "archived":
                return {
                    "status": "unchanged",
                    "reason": "already_archived",
                    "document_id": document_id,
                }

            if row["ingestion_status"] != "ready":
                return {
                    "status": "blocked",
                    "reason": "only_active_documents_can_be_archived",
                }

            connection.execute("""
                UPDATE documents
                SET
                    ingestion_status = 'archived',
                    updated_at = NOW()
                WHERE id = %s
                  AND ingestion_status = 'ready'
            """, (
                document_id,
            ))

            connection.execute("""
                INSERT INTO document_ingestion_events (
                    document_id,
                    source_key,
                    event_type,
                    actor,
                    details
                )
                VALUES (
                    %s,
                    %s,
                    'document_archived',
                    %s,
                    %s::jsonb
                )
            """, (
                document_id,
                source_key,
                actor,
                json.dumps({
                    "reason": "explicit_archive_operation",
                }),
            ))

        return {
            "status": "archived",
            "document_id": document_id,
        }

    def restore_document(
        self,
        document_id: int,
        *,
        actor: str = "local_owner",
    ) -> dict[str, Any]:
        """
        Restore an archived, superseded or non-expired deleted version.

        Exactly one ready version can exist for a source key.
        """

        document_id = self._validate_document_id(
            document_id
        )

        if document_id is None:
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        actor = self._normalize_actor(actor)
        self._require_mutation_permission()

        with self.store._connect() as connection:

            identity = connection.execute("""
                SELECT source_key
                FROM documents
                WHERE id = %s
            """, (
                document_id,
            )).fetchone()

            if identity is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            source_key = str(
                identity["source_key"] or ""
            )

            self._acquire_document_source_lock(
                connection,
                source_key,
                document_id,
            )

            target = connection.execute("""
                SELECT
                    id,
                    source_key,
                    source_version,
                    ingestion_status,
                    retention_until,
                    (
                        retention_until IS NOT NULL
                        AND retention_until <= NOW()
                    ) AS retention_expired
                FROM documents
                WHERE id = %s
                FOR UPDATE
            """, (
                document_id,
            )).fetchone()

            if target is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            current_status = str(
                target["ingestion_status"]
            )

            if current_status == "ready":
                return {
                    "status": "unchanged",
                    "reason": "document_already_active",
                    "document_id": document_id,
                }

            if current_status == "purged":
                return {
                    "status": "blocked",
                    "reason": "permanently_purged",
                    "document_id": document_id,
                }

            if current_status == "deleted":
                if (
                    target["retention_expired"]
                    or target["retention_until"] is None
                ):
                    return {
                        "status": "blocked",
                        "reason": "retention_expired",
                        "document_id": document_id,
                    }

            elif current_status not in {
                "archived",
                "superseded",
            }:
                return {
                    "status": "blocked",
                    "reason": "document_version_not_restorable",
                    "document_id": document_id,
                }

            previous = None

            if source_key:
                previous = connection.execute("""
                    SELECT id
                    FROM documents
                    WHERE source_key = %s
                      AND ingestion_status = 'ready'
                      AND id <> %s
                    ORDER BY source_version DESC
                    LIMIT 1
                    FOR UPDATE
                """, (
                    source_key,
                    document_id,
                )).fetchone()

            previous_document_id = (
                int(previous["id"])
                if previous is not None
                else None
            )

            if previous_document_id is not None:

                connection.execute("""
                    UPDATE documents
                    SET
                        ingestion_status = 'superseded'
                    WHERE id = %s
                      AND ingestion_status = 'ready'
                """, (
                    previous_document_id,
                ))

                connection.execute("""
                    INSERT INTO document_ingestion_events (
                        document_id,
                        previous_document_id,
                        source_key,
                        event_type,
                        actor,
                        details
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        'document_superseded',
                        %s,
                        %s::jsonb
                    )
                """, (
                    previous_document_id,
                    document_id,
                    source_key,
                    actor,
                    json.dumps({
                        "reason": "another_version_restored",
                        "restored_document_id": document_id,
                    }),
                ))

            restored = connection.execute("""
                UPDATE documents
                SET
                    ingestion_status = 'ready',
                    deleted_at = NULL,
                    deleted_by = NULL,
                    retention_until = NULL,
                    integrity_status = 'unchecked'
                WHERE id = %s
                  AND ingestion_status = %s
                RETURNING id
            """, (
                document_id,
                current_status,
            )).fetchone()

            if restored is None:
                return {
                    "status": "conflict",
                    "reason": "restore_state_changed",
                    "document_id": document_id,
                }

            connection.execute("""
                INSERT INTO document_ingestion_events (
                    document_id,
                    previous_document_id,
                    source_key,
                    event_type,
                    actor,
                    details
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    'document_restored',
                    %s,
                    %s::jsonb
                )
            """, (
                document_id,
                previous_document_id,
                source_key,
                actor,
                json.dumps({
                    "restored_version": int(
                        target["source_version"]
                    ),
                    "previous_status": current_status,
                }),
            ))

        return {
            "status": "restored",
            "document_id": document_id,
            "source_version": int(
                target["source_version"]
            ),
            "previous_document_id": previous_document_id,
        }

    def document_history(
        self,
        document_id: int,
        *,
        limit: int = 30,
    ) -> list[dict[str, Any]]:

        try:
            document_id = int(document_id)
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            return []

        if document_id <= 0:
            return []

        with self.store._connect() as connection:
            rows = connection.execute("""
                SELECT
                    id,
                    document_id,
                    previous_document_id,
                    source_key,
                    event_type,
                    actor,
                    details,
                    created_at
                FROM document_ingestion_events
                WHERE document_id = %s
                   OR previous_document_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
            """, (
                document_id,
                document_id,
                limit,
            )).fetchall()

        return [dict(row) for row in rows]

    def delete_document(
        self,
        document_id: int,
        *,
        confirmed: bool = False,
        expected_source_version: int | None = None,
        retention_days: int = DOCUMENT_TRASH_RETENTION_DAYS,
        actor: str = "local_owner",
    ) -> dict[str, Any]:
        """
        Soft-delete a document.

        First call returns needs_confirmation.
        Second call with confirmed=True performs the deletion.

        The content remains recoverable during the retention window.
        """

        document_id = self._validate_document_id(
            document_id
        )

        if document_id is None:
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        if not isinstance(confirmed, bool):
            return {
                "status": "invalid",
                "reason": "invalid_confirmation",
            }

        try:
            retention_days = int(retention_days)
        except (TypeError, ValueError):
            return {
                "status": "invalid",
                "reason": "invalid_retention_days",
            }

        if not 1 <= retention_days <= 3650:
            return {
                "status": "invalid",
                "reason": "retention_days_out_of_range",
            }

        actor = self._normalize_actor(actor)

        if confirmed:
            try:
                expected_source_version = int(
                    expected_source_version
                )
            except (TypeError, ValueError):
                return {
                    "status": "invalid",
                    "reason": "expected_source_version_required",
                    "data_deleted": False,
                }

        if not confirmed:
            with self.store._connect() as connection:
                row = connection.execute("""
                    SELECT
                        id,
                        file_name,
                        source_version,
                        ingestion_status
                    FROM documents
                    WHERE id = %s
                """, (
                    document_id,
                )).fetchone()

            if row is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            if row["ingestion_status"] in {
                "deleted",
                "purged",
            }:
                return {
                    "status": "unchanged",
                    "reason": row["ingestion_status"],
                    "document_id": document_id,
                }

            return {
                "status": "needs_confirmation",
                "action": "document_delete",
                "document_id": document_id,
                "file_name": row["file_name"],
                "source_version": row["source_version"],
                "retention_days": retention_days,
                "confirmation_action": (
                    f"/document-delete-confirm "
                    f"{document_id} "
                    f"{row['source_version']}"
                ),
                "data_deleted": False,
            }

        self._require_mutation_permission()

        with self.store._connect() as connection:

            identity = connection.execute("""
                SELECT source_key
                FROM documents
                WHERE id = %s
            """, (
                document_id,
            )).fetchone()

            if identity is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            source_key = str(
                identity["source_key"] or ""
            )

            self._acquire_document_source_lock(
                connection,
                source_key,
                document_id,
            )

            row = connection.execute("""
                SELECT
                    id,
                    source_key,
                    file_name,
                    source_version,
                    ingestion_status
                FROM documents
                WHERE id = %s
                FOR UPDATE
            """, (
                document_id,
            )).fetchone()

            if row is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            status = str(row["ingestion_status"])

            if (
                int(row["source_version"])
                != expected_source_version
            ):
                return {
                    "status": "conflict",
                    "reason": "document_version_changed_since_preview",
                    "document_id": document_id,
                    "data_deleted": False,
                }
            
            if status == "deleted":
                return {
                    "status": "unchanged",
                    "reason": "already_deleted",
                    "document_id": document_id,
                }

            if status == "purged":
                return {
                    "status": "blocked",
                    "reason": "document_already_purged",
                    "document_id": document_id,
                }

            if status not in {
                "ready",
                "archived",
                "superseded",
            }:
                return {
                    "status": "blocked",
                    "reason": "document_state_not_deletable",
                    "document_id": document_id,
                }

            deleted = connection.execute("""
                UPDATE documents
                SET
                    ingestion_status = 'deleted',
                    deleted_at = NOW(),
                    deleted_by = %s,
                    retention_until = (
                        NOW() + make_interval(days => %s)
                    ),
                    integrity_status = 'unchecked'
                WHERE id = %s
                  AND ingestion_status = %s
                RETURNING retention_until
            """, (
                actor,
                retention_days,
                document_id,
                status,
            )).fetchone()

            if deleted is None:
                return {
                    "status": "conflict",
                    "reason": "document_state_changed",
                    "document_id": document_id,
                }

            connection.execute("""
                INSERT INTO document_ingestion_events (
                    document_id,
                    source_key,
                    event_type,
                    actor,
                    details
                )
                VALUES (
                    %s,
                    %s,
                    'document_deleted',
                    %s,
                    %s::jsonb
                )
            """, (
                document_id,
                source_key,
                actor,
                json.dumps({
                    "previous_status": status,
                    "retention_days": retention_days,
                    "soft_delete": True,
                }),
            ))

        return {
            "status": "deleted",
            "document_id": document_id,
            "retention_days": retention_days,
            "retention_until": deleted["retention_until"],
            "permanent_purge": False,
        }

    def preview_expired_document_purge(
        self,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Read-only preview of documents eligible for permanent purge."""

        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            return {
                "status": "invalid",
                "reason": "invalid_limit",
            }

        with self.store._connect() as connection:

            summary = connection.execute("""
                SELECT COUNT(*) AS eligible_count
                FROM documents
                WHERE ingestion_status = 'deleted'
                  AND retention_until IS NOT NULL
                  AND retention_until <= NOW()
            """).fetchone()

            rows = connection.execute("""
                SELECT
                    id,
                    file_name,
                    source_version,
                    deleted_at,
                    retention_until
                FROM documents
                WHERE ingestion_status = 'deleted'
                  AND retention_until IS NOT NULL
                  AND retention_until <= NOW()
                ORDER BY retention_until ASC, id ASC
                LIMIT %s
            """, (
                limit,
            )).fetchall()

        return {
            "status": "preview",
            "eligible_count": int(
                summary["eligible_count"]
            ),
            "documents": [
                dict(row)
                for row in rows
            ],
            "permanent_deletion": True,
            "confirmation_required": True,
        }

    def purge_expired_documents(
        self,
        *,
        confirmed: bool = False,
        batch_size: int = 25,
        actor: str = "local_owner",
    ) -> dict[str, Any]:
        """
        Permanently purge expired soft-deleted document content.

        The parent document and event records remain as minimal tombstones
        so historical foreign keys and audit trails remain valid.
        """

        if confirmed is not True:
            return {
                "status": "needs_confirmation",
                "reason": "permanent_purge_requires_confirmation",
                "preview": self.preview_expired_document_purge(),
            }

        self._require_mutation_permission()

        try:
            batch_size = max(
                1,
                min(int(batch_size), MAX_DOCUMENT_PURGE_BATCH),
            )
        except (TypeError, ValueError):
            return {
                "status": "invalid",
                "reason": "invalid_batch_size",
            }

        actor = self._normalize_actor(actor)

        purged: list[dict[str, Any]] = []
        skipped = 0

        with self.store._connect() as connection:

            # Do not lock candidate rows before advisory locks.
            # Ingestion uses advisory-lock -> row-lock ordering.
            candidates = connection.execute("""
                SELECT
                    id,
                    source_key
                FROM documents
                WHERE ingestion_status = 'deleted'
                  AND retention_until IS NOT NULL
                  AND retention_until <= NOW()
                ORDER BY retention_until ASC, id ASC
                LIMIT %s
            """, (
                batch_size,
            )).fetchall()

            for candidate in candidates:

                document_id = int(candidate["id"])
                source_key = str(
                    candidate["source_key"] or ""
                )

                self._acquire_document_source_lock(
                    connection,
                    source_key,
                    document_id,
                )

                row = connection.execute("""
                    SELECT
                        id,
                        source_key,
                        ingestion_status,
                        retention_until
                    FROM documents
                    WHERE id = %s
                    FOR UPDATE
                """, (
                    document_id,
                )).fetchone()

                if (
                    row is None
                    or row["ingestion_status"] != "deleted"
                ):
                    skipped += 1
                    continue

                eligible = connection.execute("""
                    SELECT (
                        retention_until IS NOT NULL
                        AND retention_until <= NOW()
                    ) AS is_eligible
                    FROM documents
                    WHERE id = %s
                """, (
                    document_id,
                )).fetchone()

                if not eligible["is_eligible"]:
                    skipped += 1
                    continue

                chunks_deleted = connection.execute("""
                    DELETE FROM document_chunks
                    WHERE document_id = %s
                """, (
                    document_id,
                )).rowcount

                # Delete only cache vectors no remaining document uses.
                connection.execute("""
                    DELETE FROM document_embedding_cache AS cache
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM document_chunks AS chunk
                        WHERE
                            chunk.embedding_text_hash =
                                cache.embedding_text_hash
                            AND chunk.embedding_model =
                                cache.embedding_model
                            AND chunk.embedding_model_version =
                                cache.embedding_model_version
                            AND chunk.embedding_dimensions =
                                cache.embedding_dimensions
                            AND chunk.embedding_format_version =
                                cache.embedding_format_version
                    )
                """)

                tombstone_hash = hashlib.sha256(
                    f"purged-document:{document_id}".encode("utf-8")
                ).hexdigest()

                tombstone_key = f"purged:{document_id}"

                connection.execute("""
                    UPDATE document_ingestion_events
                    SET
                        source_key = %s,
                        details = (
                            details
                            - 'content_sha256'
                            - 'extracted_text_sha256'
                            - 'source_path'
                            - 'file_path'
                            - 'file_name'
                            - 'extraction_metadata'
                        )
                    WHERE document_id = %s
                """, (
                    tombstone_key,
                    document_id,
                ))

                connection.execute("""
                    UPDATE documents
                    SET
                        ingestion_status = 'purged',
                        source_path = '',
                        source_key = %s,
                        file_name = '[purged]',
                        content_sha256 = %s,
                        extracted_text_sha256 = '',
                        file_size_bytes = 0,
                        media_type = 'application/octet-stream',
                        source_mtime_ns = NULL,
                        extraction_method = 'purged',
                        extraction_status = 'purged',
                        extraction_warnings = '[]'::jsonb,
                        extraction_metadata = '{}'::jsonb,
                        source_encoding = NULL,
                        page_count = NULL,
                        retention_until = NULL,
                        purged_at = NOW(),
                        integrity_status = 'purged',
                        last_integrity_check_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                      AND ingestion_status = 'deleted'
                """, (
                    tombstone_key,
                    tombstone_hash,
                    document_id,
                ))

                connection.execute("""
                    INSERT INTO document_ingestion_events (
                        document_id,
                        source_key,
                        event_type,
                        actor,
                        details
                    )
                    VALUES (
                        %s,
                        %s,
                        'document_purged',
                        %s,
                        %s::jsonb
                    )
                """, (
                    document_id,
                    tombstone_key,
                    actor,
                    json.dumps({
                        "chunks_deleted": int(
                            chunks_deleted or 0
                        ),
                        "content_removed": True,
                        "audit_tombstone_preserved": True,
                    }),
                ))

                purged.append({
                    "document_id": document_id,
                    "chunks_deleted": int(
                        chunks_deleted or 0
                    ),
                    "status": "purged",
                })

        with self.store._connect() as connection:
            remaining_row = connection.execute("""
                SELECT COUNT(*) AS remaining_count
                FROM documents
                WHERE ingestion_status = 'deleted'
                  AND retention_until IS NOT NULL
                  AND retention_until <= NOW()
            """).fetchone()

            remaining_count = int(
                remaining_row["remaining_count"]
            )

        return {
            "status": (
                "purged"
                if purged
                else "nothing_to_purge"
            ),
            "purged_count": len(purged),
            "skipped_count": skipped,
            "results": purged,
            "remaining_expired_documents": remaining_count,
            "remaining_batch_required": (
                remaining_count > 0
            ),
        }

    def verify_document_integrity(
        self,
        document_id: int,
        *,
        actor: str = "local_owner",
    ) -> dict[str, Any]:
        """
        Verify stored document chunks against immutable ingestion evidence.

        Checks:
        - Expected chunk count.
        - Stable chunk identifiers.
        - Missing embeddings.
        - Embedding model provenance.
        - Embedding dimensions and format.
        - Embedding input hashes.
        - Local source-file availability and content hash.

        This does not regenerate embeddings or change document content.
        """

        document_id = self._validate_document_id(
            document_id
        )

        if document_id is None:
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        actor = self._normalize_actor(actor)

        self._require_mutation_permission()

        with self.store._connect() as connection:

            identity = connection.execute("""
                SELECT source_key
                FROM documents
                WHERE id = %s
            """, (
                document_id,
            )).fetchone()

            if identity is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            source_key = str(
                identity["source_key"] or ""
            )

            self._acquire_document_source_lock(
                connection,
                source_key,
                document_id,
            )

            document = connection.execute("""
                SELECT
                    id,
                    source_key,
                    source_path,
                    file_name,
                    content_sha256,
                    source_version,
                    ingestion_status
                FROM documents
                WHERE id = %s
                FOR UPDATE
            """, (
                document_id,
            )).fetchone()

            if document is None:
                return {
                    "status": "not_found",
                    "document_id": document_id,
                }

            if document["ingestion_status"] == "purged":
                return {
                    "status": "purged",
                    "document_id": document_id,
                    "integrity_status": "purged",
                }

            latest_event = connection.execute("""
                SELECT details
                FROM document_ingestion_events
                WHERE document_id = %s
                  AND event_type IN (
                      'document_indexed',
                      'duplicate_content_indexed'
                  )
                ORDER BY id DESC
                LIMIT 1
            """, (
                document_id,
            )).fetchone()

            details = (
                latest_event["details"]
                if latest_event is not None
                else {}
            )

            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except json.JSONDecodeError:
                    details = {}

            if not isinstance(details, dict):
                details = {}

            chunks = connection.execute("""
                SELECT
                    chunk_index,
                    content,
                    page_number,
                    section,
                    embedding,
                    embedding_model,
                    embedding_model_version,
                    embedding_dimensions,
                    embedding_format_version,
                    embedding_text_hash,
                    metadata
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index ASC
            """, (
                document_id,
            )).fetchall()

            issues: list[dict[str, Any]] = []
            warnings: list[dict[str, Any]] = []

            def add_issue(code: str, **data: Any) -> None:
                issues.append({
                    "code": code,
                    **data,
                })

            if latest_event is None:
                add_issue("missing_successful_ingestion_event")

            expected_count = details.get("chunk_count")

            if expected_count is not None:
                try:
                    expected_count = int(expected_count)
                except (TypeError, ValueError):
                    expected_count = None
                    add_issue("invalid_expected_chunk_count")

                if (
                    expected_count is not None
                    and expected_count != len(chunks)
                ):
                    add_issue(
                        "chunk_count_mismatch",
                        expected=expected_count,
                        actual=len(chunks),
                    )

            expected_model = details.get("embedding_model")
            expected_version = details.get(
                "embedding_model_version"
            )
            expected_dimensions = details.get(
                "embedding_dimensions"
            )
            expected_format = details.get(
                "embedding_format_version"
            )

            if not all((
                expected_model,
                expected_version,
                expected_dimensions,
                expected_format,
            )):
                add_issue(
                    "missing_embedding_provenance_in_ingestion_event"
                )

            try:
                expected_dimensions = (
                    int(expected_dimensions)
                    if expected_dimensions is not None
                    else None
                )
            except (TypeError, ValueError):
                expected_dimensions = None

            occurrences: dict[str, int] = {}

            for index, row in enumerate(chunks):

                actual_index = int(row["chunk_index"])

                if actual_index != index:
                    add_issue(
                        "chunk_index_sequence_mismatch",
                        expected=index,
                        actual=actual_index,
                    )

                if row["embedding"] is None:
                    add_issue(
                        "missing_embedding",
                        chunk_index=actual_index,
                    )

                if not str(
                    row["embedding_text_hash"] or ""
                ).strip():
                    add_issue(
                        "missing_embedding_input_hash",
                        chunk_index=actual_index,
                    )

                if (
                    expected_model
                    and row["embedding_model"] != expected_model
                ):
                    add_issue(
                        "embedding_model_mismatch",
                        chunk_index=actual_index,
                    )

                if (
                    expected_version
                    and row["embedding_model_version"]
                    != expected_version
                ):
                    add_issue(
                        "embedding_version_mismatch",
                        chunk_index=actual_index,
                    )

                if (
                    expected_dimensions is not None
                    and int(
                        row["embedding_dimensions"] or 0
                    ) != expected_dimensions
                ):
                    add_issue(
                        "embedding_dimensions_mismatch",
                        chunk_index=actual_index,
                    )

                if (
                    expected_format
                    and row["embedding_format_version"]
                    != expected_format
                ):
                    add_issue(
                        "embedding_format_mismatch",
                        chunk_index=actual_index,
                    )

                content = str(row["content"] or "")

                content_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                identity_base = "\x1f".join((
                    source_key,
                    str(row["page_number"] or ""),
                    str(row["section"] or ""),
                    content_hash,
                ))

                occurrence = occurrences.get(
                    identity_base,
                    0,
                )

                occurrences[identity_base] = occurrence + 1

                expected_chunk_id = hashlib.sha256(
                    (
                        f"{identity_base}\x1f{occurrence}"
                    ).encode("utf-8")
                ).hexdigest()

                metadata = row["metadata"] or {}

                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}

                chunking = (
                    metadata.get("chunking", {})
                    if isinstance(metadata, dict)
                    else {}
                )

                actual_chunk_id = (
                    chunking.get("stable_chunk_id")
                    if isinstance(chunking, dict)
                    else None
                )

                if actual_chunk_id != expected_chunk_id:
                    add_issue(
                        "stable_chunk_id_mismatch",
                        chunk_index=actual_index,
                    )

            # The source file is an external copy of the indexed snapshot.
            # Its absence is a warning, not proof of corrupted database data.
            source_path = str(
                document["source_path"] or ""
            )

            source_state = "not_available"

            if source_path:

                path = Path(source_path)

                try:
                    if path.is_file():
                        stat = path.stat()

                        if stat.st_size > MAX_FILE_BYTES:
                            source_state = "size_limit_exceeded"
                        else:
                            current_hash = hashlib.sha256(
                                path.read_bytes()
                            ).hexdigest()

                            source_state = (
                                "unchanged"
                                if current_hash
                                == document["content_sha256"]
                                else "changed"
                            )

                    else:
                        source_state = "missing"

                except OSError:
                    source_state = "unreadable"

            if source_state == "changed":
                warnings.append({
                    "code": "source_file_changed_since_ingestion",
                })

            elif source_state in {
                "missing",
                "unreadable",
                "size_limit_exceeded",
            }:
                warnings.append({
                    "code": "source_file_unavailable",
                    "reason": source_state,
                })

            status = (
                "degraded"
                if issues
                else "healthy"
            )

            connection.execute("""
                UPDATE documents
                SET
                    integrity_status = %s,
                    last_integrity_check_at = NOW()
                WHERE id = %s
            """, (
                status,
                document_id,
            ))

            connection.execute("""
                INSERT INTO document_ingestion_events (
                    document_id,
                    source_key,
                    event_type,
                    actor,
                    details
                )
                VALUES (
                    %s,
                    %s,
                    'document_integrity_checked',
                    %s,
                    %s::jsonb
                )
            """, (
                document_id,
                source_key,
                actor,
                json.dumps({
                    "integrity_status": status,
                    "chunk_count": len(chunks),
                    "issue_codes": [
                        item["code"]
                        for item in issues
                    ],
                    "warning_codes": [
                        item["code"]
                        for item in warnings
                    ],
                    "source_state": source_state,
                }),
            ))

        return {
            "status": status,
            "document_id": document_id,
            "source_version": int(
                document["source_version"]
            ),
            "ingestion_status": document["ingestion_status"],
            "expected_chunks": expected_count,
            "actual_chunks": len(chunks),
            "issues": issues,
            "warnings": warnings,
            "source_state": source_state,
            "checked_at": "recorded_in_database",
        }

    def reindex_document(
        self,
        document_id: int,
        *,
        actor: str = "local_owner",
    ) -> dict[str, Any]:
        """
        Rebuild an unhealthy active document index from its original file.

        The original source file must still match the indexed binary hash.
        A rebuild creates a new immutable document version when necessary.
        """

        document_id = self._validate_document_id(
            document_id
        )

        if document_id is None:
            return {
                "status": "invalid",
                "reason": "invalid_document_id",
            }

        actor = self._normalize_actor(actor)
        self._require_mutation_permission()

        with self.store._connect() as connection:
            document = connection.execute("""
                SELECT
                    id,
                    source_path,
                    file_name,
                    content_sha256,
                    category,
                    language,
                    ingestion_status,
                    source_version
                FROM documents
                WHERE id = %s
            """, (
                document_id,
            )).fetchone()

        if document is None:
            return {
                "status": "not_found",
                "document_id": document_id,
            }

        if document["ingestion_status"] != "ready":
            return {
                "status": "blocked",
                "reason": "only_active_documents_can_be_reindexed",
                "document_id": document_id,
            }

        source_path = str(
            document["source_path"] or ""
        )

        if not source_path:
            return {
                "status": "blocked",
                "reason": "original_source_path_missing",
                "document_id": document_id,
            }

        path = Path(source_path)

        if not path.is_file():
            return {
                "status": "blocked",
                "reason": "original_source_unavailable",
                "document_id": document_id,
            }

        try:
            stat = path.stat()

            if stat.st_size > MAX_FILE_BYTES:
                return {
                    "status": "blocked",
                    "reason": "source_file_exceeds_size_limit",
                    "document_id": document_id,
                }

            current_hash = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

        except OSError:
            return {
                "status": "blocked",
                "reason": "original_source_unreadable",
                "document_id": document_id,
            }

        if current_hash != document["content_sha256"]:
            return {
                "status": "blocked",
                "reason": "source_changed_reingestion_required",
                "document_id": document_id,
                "instruction": (
                    "The original source has changed. "
                    "Use normal ingestion to create a new version."
                ),
            }

        # Audit before attempting recovery.
        integrity = self.verify_document_integrity(
            document_id,
            actor=actor,
        )

        if integrity.get("status") == "healthy":
            return {
                "status": "already_healthy",
                "document_id": document_id,
                "integrity": integrity,
            }

        # Reuse the existing safe ingestion pipeline.
        # It reuses matching cached embeddings and creates an
        # immutable version instead of destructively overwriting
        # the old document.
        result = self.ingest_file(
            path,
            category=str(document["category"] or "general"),
            language=str(document["language"] or "unknown"),
            force=False,
        )

        if result.get("status") == "already_indexed":
            return {
                "status": "recovery_not_confirmed",
                "reason": "ingestion_returned_existing_index",
                "document_id": document_id,
                "integrity": integrity,
            }

        new_document_id = int(
            result["document_id"]
        )

        if new_document_id != document_id:

            with self.store._connect() as connection:

                new_identity = connection.execute("""
                    SELECT source_key
                    FROM documents
                    WHERE id = %s
                """, (
                    new_document_id,
                )).fetchone()

                connection.execute("""
                    INSERT INTO document_ingestion_events (
                        document_id,
                        previous_document_id,
                        source_key,
                        event_type,
                        actor,
                        details
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        'document_reindexed',
                        %s,
                        %s::jsonb
                    )
                """, (
                    new_document_id,
                    document_id,
                    str(
                        new_identity["source_key"] or ""
                    ) if new_identity else "",
                    actor,
                    json.dumps({
                        "previous_version": int(
                            document["source_version"]
                        ),
                        "recovery_status": result.get("status"),
                    }),
                ))

        return {
            "status": "reindexed",
            "previous_document_id": document_id,
            "new_document_id": new_document_id,
            "ingestion_result": result,
        }