"""Reliable language-mode detection for Zara."""
from __future__ import annotations

import re

DEFAULT_LANGUAGE = "hinglish"
SUPPORTED_LANGUAGES = {"hinglish", "hindi", "english"}

_LANGUAGE_TOKEN = (
    r"(?:hinglish|hindi|english|अंग्रेज़ी|अंग्रेजी|हिंदी|हिन्दी)"
)
# Common misspellings are accepted so a typo does not prevent
# an explicit default-language preference from being understood.
_DEFAULT_TOKEN = r"(?:default|deafult|defualt)"


def normalize_language_mode(value: str | None) -> str | None:
    if not value:
        return None

    value = value.strip().casefold()

    aliases = {
        "hinglish": "hinglish",
        "mixed hindi english": "hinglish",
        "hindi": "hindi",
        "हिंदी": "hindi",
        "हिन्दी": "hindi",
        "english": "english",
        "अंग्रेज़ी": "english",
        "अंग्रेजी": "english",
    }

    normalized = aliases.get(value)
    return normalized if normalized in SUPPORTED_LANGUAGES else None


_PERSISTENT_PATTERNS = (
    # "default language Hinglish rakho"
    re.compile(
        rf"\b{_DEFAULT_TOKEN}(?:\s+response)?\s+language\s*"
        rf"(?:(?:is|=|:|should\s+be|to)\s*)?"
        rf"({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "set the default language to Hinglish"
    re.compile(
        rf"\b(?:set|change|switch)\s+(?:the\s+)?"
        rf"(?:default\s+)?(?:response\s+)?language\s+"
        rf"(?:to|as|=)?\s*({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "ab se Hinglish mein baat karna"
    re.compile(
        rf"\b(?:ab\s*se|aaj\s*se|hamesha|always|permanently|"
        rf"from\s+now\s+on|going\s+forward|"
        rf"for\s+all\s+future\s+replies|future\s+me)\b"
        rf"[\s\S]{{0,50}}?\b({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "language Hinglish rakho"
    re.compile(
        rf"\b(?:language|mode)\s+(?:ko\s+)?"
        rf"({_LANGUAGE_TOKEN})\s+(?:kar(?:o|do)|rakho|set\s+karo)\b",
        re.IGNORECASE,
    ),
    # "Hinglish ko default rakho"
    re.compile(
        rf"\b({_LANGUAGE_TOKEN})\s+(?:ko\s+)?"
        rf"(?:default|permanent|hamesha|always)\b",
        re.IGNORECASE,
    ),
)


_CURRENT_TURN_PATTERNS = (
    # "answer in English", "reply English"
    re.compile(
        rf"\b(?:answer|reply|respond|speak|write|explain|tell|translate)"
        rf"\s+(?:to\s+me\s+)?(?:in\s+)?({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "English mein batao", "Hinglish me jawab dena"
    re.compile(
        rf"\b({_LANGUAGE_TOKEN})\s+(?:mein|me)\s+"
        rf"(?:batao|batana|bolo|likho|samjhao|samjhana|"
        rf"reply|answer|respond|speak|write|"
        rf"jawab\s+(?:do|dena|dijiye)|"
        rf"baat\s+(?:karo|karna)|translate\s+karo)\b",
        re.IGNORECASE,
    ),
    # "please use Hinglish"
    re.compile(
        rf"\b(?:please\s+)?use\s+(?:natural\s+)?"
        rf"({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "switch to Hinglish", "change to Hindi"
    re.compile(
        rf"\b(?:switch|change)\s+(?:the\s+)?(?:language\s+)?"
        rf"(?:to\s+)?({_LANGUAGE_TOKEN})\b",
        re.IGNORECASE,
    ),
    # "Hinglish only", "only Hinglish"
    re.compile(
        rf"\b(?:({_LANGUAGE_TOKEN})\s+only|"
        rf"only\s+({_LANGUAGE_TOKEN}))\b",
        re.IGNORECASE,
    ),
)


def _target_from_patterns(
    text: str,
    patterns: tuple[re.Pattern[str], ...],
) -> str | None:
    for pattern in patterns:
        match = pattern.search(text or "")
        if not match:
            continue

        for group in match.groups():
            if group:
                normalized = normalize_language_mode(group)
                if normalized:
                    return normalized

    return None


def detect_persistent_language_switch(text: str) -> str | None:
    """Detect the target of an explicit lasting language preference."""
    if not text:
        return None

    # Read the language attached to the persistent/default directive,
    # not simply the first language mentioned in the whole sentence.
    return _target_from_patterns(text, _PERSISTENT_PATTERNS)


def detect_current_turn_language(text: str) -> str | None:
    """Detect an explicit language request for the current response."""
    if not text:
        return None

    # A direct request for this answer takes priority over a default
    # preference mentioned elsewhere in the same message.
    current_target = _target_from_patterns(text, _CURRENT_TURN_PATTERNS)
    if current_target:
        return current_target

    # A lasting preference also governs this answer if no temporary
    # language request was specified.
    return detect_persistent_language_switch(text)