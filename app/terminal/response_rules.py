"""Deterministic output-format controls."""
from __future__ import annotations

import re

from app.ai.response_policy import (
    is_personal_fact_question,
    is_profile_question,
)


_CODE_GENERATION_PATTERN = re.compile(
    r"\b(?:write|create|build|implement|generate|make|banao|"
    r"bana\s+do|likho|likh\s+do)\b.{0,100}\b"
    r"(?:code|function|script|class|program|api|endpoint|module|test)\b"
    r"|"
    r"\b(?:code|function|script|class|program|api|endpoint|module|test)"
    r"\b.{0,80}\b(?:write|create|build|implement|generate|make|banao|"
    r"bana\s+do|likho|likh\s+do)\b",
    re.IGNORECASE,
)


def detect_output_constraint(text: str) -> str:
    value = (text or "").lower()

    if re.search(
        r"\b(code\s+only|only\s+code|just\s+the\s+code)\b", value
    ):
        return "code_only"

    if re.search(
        r"\b(one\s+word|single\s+word|sirf\s+ek\s+shabd)\b", value
    ):
        return "one_word"

    if re.search(
        r"\b(one\s+sentence|single\s+sentence|sirf\s+ek\s+sentence|"
        r"ek\s+hi\s+vakya)\b",
        value,
    ):
        return "one_sentence"

    if _CODE_GENERATION_PATTERN.search(value):
        return "code_generation"

    if re.search(
        r"\b(step\s*[- ]?by\s*[- ]?step|steps|numbered\s+steps|stepwise)\b",
        value,
    ):
        return "steps"

    if re.search(r"\b(table|tabular\s+format)\b", value):
        return "table"

    if re.search(
        r"\b(bullet\s+points|in\s+a\s+list|make\s+a\s+list|list\s+format)\b",
        value,
    ):
        return "list"

    if re.search(
        r"\b(detailed|in\s+detail|deeply|in\s+depth|complete\s+explanation)\b",
        value,
    ):
        return "detailed"

    if re.search(
        r"\b(short|brief|concise|in\s+short|chhota\s+answer|short\s+answer)\b",
        value,
    ):
        return "short"
    
    if is_profile_question(value):
        return "profile_summary"

    if is_personal_fact_question(value):
        return "short"

    return "normal"


def get_output_limit(constraint: str) -> int:
    """Return Ollama's num_predict limit for the selected output format."""
    limits = {
        "one_word": 24,
        "one_sentence": 90,
        "short": 160,
        "profile_summary": 240,
        "normal": 320,
        "list": 420,
        "table": 500,
        "steps": 650,
        "detailed": 900,
        "code_only": 1400,
        "code_generation": 1400,
    }
    return limits.get(constraint, limits["normal"])


def build_output_instruction(constraint: str) -> str:
    instructions = {
        "one_word": (
            "Return exactly one word unless the task makes that impossible."
        ),
        "one_sentence": "Answer in one sentence only.",
        "short": (
            "Keep the answer concise and focused; avoid extra background."
        ),
        "list": "Use a clear, compact bullet list.",
        "table": "Use a readable table when the content fits a table.",
        "steps": "Give the steps in a clear, ordered sequence.",
        "detailed": (
            "Give a thorough, structured explanation without repeating points."
        ),
        "code_only": (
            "Return only the requested code. Do not add an introduction "
            "or explanation outside the code."
        ),
        "normal": "Choose a natural answer length based on the request.",
        "profile_summary": (
            "PROFILE SUMMARY — STRICT RULES: Use only the saved profile "
            "facts provided in the current context. If facts are present, "
            "never say that you do not know the user. Give at most 5 short "
            "bullets and no more than 65 words total. Select the most useful "
            "facts about current learning, projects, goals and preferences. "
            "Do not list every stored field. Do not invent facts, contradict "
            "yourself, repeat points, add generic praise, or ask an unnecessary "
            "follow-up question. If no profile facts are supplied, briefly "
            "say that the saved profile is unavailable."
        ),      
        "code_generation": (
            "Provide complete, copyable code for the requested task, with "
            "required imports and surrounding definitions. Avoid unrelated advice."
        ),
    }

    return instructions.get(constraint, instructions["normal"])