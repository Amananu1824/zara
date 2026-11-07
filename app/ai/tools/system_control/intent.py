"""Conservative full-message intent detection for System Control."""

from __future__ import annotations

import re
import unicodedata


_MAC_PATTERN = re.compile(
    r"(?i)(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}|"
    r"\b[0-9a-f]{12}\b"
)
_QUOTED_TEXT = re.compile(
    r"""(?:"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|(?<!\w)'[^'\n]*'(?!\w))"""
)

_BLOCKED_CONTEXT = re.compile(
    r"\b(?:"
    r"what if|what would happen if|what happens if|"
    r"suppose|hypothetically|imagine|in theory|"
    r"explain|define|meaning of|what does|"
    r"what do you mean by|what is meant by|"
    r"how to|how do i|how can i|how should i|how would i|"
    r"tell me how|teach me|"
    r"should i|is it safe to|wonder if|not sure if|"
    r"translate|rewrite|proofread|"
    r"as an example|for example|in a story|"
    r"i was thinking|i am thinking|i'm thinking|"
    r"i am considering|i'm considering|considering|"
    r"i was planning|i plan to|i am planning|i'm planning|"
    r"i will|i'll|i might|i may|i could|"
    r"remind me to|set a reminder|schedule|"
    r"said|mentioned|asked|wrote|typed|"
    r"kal karna|baad mein karna|later|tomorrow|"
    r"next week|after \d+"
    r")\b",
    re.IGNORECASE,
)

_NEGATION_BEFORE_ACTION = re.compile(
    r"\b(?:"
    r"do not|don't|dont|never|no need to|"
    r"not asking you to|should not|shouldn't|"
    r"nahi|nahin|nhi|mat"
    r")\b.{0,90}\b(?:"
    r"shut down|shutdown|turn off|restart|reboot|lock|"
    r"set|change|increase|decrease|raise|lower|"
    r"mute|unmute|wake"
    r")\b",
    re.IGNORECASE,
)

_NEGATION_AFTER_ACTION = re.compile(
    r"\b(?:"
    r"shut down|shutdown|turn off|restart|reboot|lock|"
    r"set|change|increase|decrease|raise|lower|"
    r"mute|unmute|wake"
    r")\b.{0,45}\b(?:"
    r"mat|nahi|nahin|nhi|not|avoid"
    r")\b",
    re.IGNORECASE,
)


_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "system_info": (
        re.compile(
            r"\b(?:show|display|get|fetch|check|tell me)\b"
            r".{0,45}\b(?:my|this|the)?\s*"
            r"(?:pc|computer|system)\b"
            r".{0,25}\b(?:info(?:rmation)?|details?|specs?|"
            r"specifications?|configuration)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat (?:are|is) (?:my|this|the) "
            r"(?:pc|computer|system) "
            r"(?:specs?|specifications?|details?|configuration)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:mere|is) (?:pc|computer|system) "
            r"(?:ki|ka) (?:details?|info|specs?) "
            r"(?:batao|dikhao)\b",
            re.IGNORECASE,
        ),
    ),

    "get_current_time": (
        re.compile(
            r"\b(?:what time is it|what is the current time|"
            r"tell me (?:the )?(?:current )?time|"
            r"current time batao|current time kya hai|"
            r"abhi (?:ka )?time kya hai|kitne baje hue hain)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:abhi|current)\s+(?:ka\s+)?time\s+"
            r"(?:batao|bataiye|dikhao)\b",
            re.IGNORECASE,
        ),
    ),

    "get_current_date": (
        re.compile(
            r"\b(?:what is (?:today'?s )?date|"
            r"what date is it|tell me (?:today'?s )?date|"
            r"today'?s date batao|today'?s date kya hai|"
            r"aaj ki date kya hai|aaj ki tareekh kya hai)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:aaj ki date|aaj ki tareekh)\s+"
            r"(?:batao|bataiye|dikhao)\b",
            re.IGNORECASE,
        ),
    ),

    "battery_status": (
        re.compile(
            r"\b(?:how much|what is|check|show|get|tell me)\b"
            r".{0,35}\b(?:my )?battery"
            r"(?:\s+(?:status|level|percentage|percent|charge))?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:battery|laptop ki battery)\b.{0,30}"
            r"\b(?:kitni|kitna|percentage|percent|status|level)\b"
            r".{0,20}\b(?:batao|hai|left|remaining|bacha)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:mere|meri|my)\s+"
            r"(?:laptop|pc|computer)\s+(?:ki|ka)\s+battery\s+"
            r"(?:status|level|percentage)\s+"
            r"(?:batao|bataiye|dikhao|check karo)\b",
            re.IGNORECASE,
        ),
    ),

    "storage_monitor": (
        re.compile(
            r"\b(?:check|show|get|tell me|how much|what is)\b"
            r".{0,45}\b(?:disk|storage|drive)\b"
            r".{0,35}\b(?:space|free|available|left|remaining|capacity)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:disk|storage|drive)\b.{0,40}"
            r"\b(?:space|kitni|kitna|bacha|free)\b",
            re.IGNORECASE,
        ),
    ),

    "process_monitor": (
        re.compile(
            r"\b(?:show|list|display|check|get|tell me)\b"
            r".{0,45}\b(?:running processes|process list|"
            r"active processes|running tasks)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat (?:processes|apps|programs) are "
            r"(?:currently )?running\b",
            re.IGNORECASE,
        ),
    ),

    "volume_control": (
        re.compile(
            r"\b(?:set|change)\b.{0,20}"
            r"\b(?:system |audio )?volume\b.{0,18}"
            r"\b\d{1,3}\s*%?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bvolume\b.{0,20}\b(?:badha|bada|kam|ghata)\b"
            r".{0,18}\b(?:do|karo|kardo)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:increase|raise|lower|decrease) "
            r"(?:the )?(?:system )?volume\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:mute|unmute) "
            r"(?:the )?(?:audio|sound|speakers?|system)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bvolume\s+(?:up|down)\s+"
            r"(?:kar\s+do|karo|please|now)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bvolume\s+\d{1,3}\s*%?\s+"
            r"(?:kar\s+do|kardo)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:awaaz|awaz)\s+(?:badha|bada|kam)\s+"
            r"(?:do|karo|kar\s+do)\b",
            re.IGNORECASE,
        ),
    ),

    "brightness_control": (
        re.compile(
            r"\b(?:set|change)\b.{0,20}"
            r"\b(?:screen )?brightness\b.{0,18}"
            r"\b\d{1,3}\s*%?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbrightness\b.{0,25}"
            r"\b(?:badha|bada|kam|ghata|set)\b"
            r".{0,18}\b(?:do|karo|kardo)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:increase|raise|lower|decrease) "
            r"(?:the )?(?:screen )?brightness\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbrightness\s+\d{1,3}\s*%?\s+"
            r"(?:kar\s+do|kardo)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:screen ki brightness|brightness ko)\b"
            r".{0,25}\b(?:badha|bada|kam|set)\b"
            r".{0,15}\b(?:do|karo|kar\s+do)\b",
            re.IGNORECASE,
        ),
    ),

    "power_off": (
        re.compile(
            r"\b(?:please )?(?:shut down|shutdown|turn off|power off)"
            r"\s+(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system|windows(?: pc)?)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"(?:band|shutdown|shut down|off)\s+"
            r"(?:kar do|kardo|please)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:mera|mere|is|my|the) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"(?:band|shutdown|shut down|off)\s+"
            r"(?:karo|kar do|kardo|kar dijiye)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:can you|could you|would you)\s+"
            r"(?:please\s+)?"
            r"(?:shut down|shutdown|turn off|power off)\s+"
            r"(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\b",
            re.IGNORECASE,
        ),
    ),

    "restart_pc": (
        re.compile(
            r"\b(?:(?:mera|mere|is|my|the) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"(?:restart|reboot)\s+"
            r"(?:karo|kar do|kardo|kar dijiye)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:can you|could you|would you)\s+"
            r"(?:please\s+)?(?:restart|reboot)\s+"
            r"(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:please )?(?:restart|reboot)\s+"
            r"(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"(?:restart|reboot)\s+(?:kar do|kardo|please)\b",
            re.IGNORECASE,
        ),
    ),

    "lock_pc": (
        re.compile(
            r"\b(?:please )?lock\s+"
            r"(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:my|the|this) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"lock\s+(?:kar do|kardo|please)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:(?:mera|mere|is|my|the) )?"
            r"(?:pc|computer|laptop|system)\s+(?:ko )?"
            r"lock\s+(?:karo|kar do|kardo|kar dijiye)\b",
            re.IGNORECASE,
        ),
    ),

    "wake_on_lan": (
        re.compile(
            r"\b(?:send|broadcast)\b.{0,40}"
            r"\b(?:wake[\s-]*on[\s-]*lan|magic packet)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:wake up|wake)\s+"
            r"(?:(?:my|the|this) )?"
            r"(?:pc|computer)\b",
            re.IGNORECASE,
        ),
    ),
}


def detect_system_control_intent(
    message: str,
) -> str | None:
    """
    Return one exact A1 tool name only for a clear current-turn request.

    Ambiguous, negated, hypothetical, quoted/explanatory or multi-action
    messages intentionally return None.
    """

    if not isinstance(message, str):
        return None

    normalized = unicodedata.normalize(
        "NFKC",
        message,
    ).replace("’", "'")

    normalized = " ".join(normalized.split()).casefold()

    # Quoted commands are text to discuss, not an instruction to execute.
    candidate = _QUOTED_TEXT.sub(
        " ",
        normalized,
    )

    candidate = " ".join(candidate.split())

    if len(candidate) < 8:
        return None

    if _BLOCKED_CONTEXT.search(candidate):
        return None

    if _NEGATION_BEFORE_ACTION.search(candidate):
        return None

    if _NEGATION_AFTER_ACTION.search(candidate):
        return None
    
    # A scheduled future action is not an immediate system command.
    if re.search(
        r"\b(?:tomorrow|later|next week|kal|baad mein)\b",
        candidate,
    ):
        return None
    
    matched_tools: list[str] = []

    for tool_name, patterns in _PATTERNS.items():
        if any(
            pattern.search(candidate)
            for pattern in patterns
        ):
            matched_tools.append(tool_name)
    # Wake-on-LAN is never allowed without an explicit target MAC.
    if (
        "wake_on_lan" in matched_tools
        and not _MAC_PATTERN.search(candidate)
    ):
        matched_tools.remove("wake_on_lan")

    # More than one possible action is ambiguous for this single-action path.
    if len(matched_tools) != 1:
        return None

    return matched_tools[0]