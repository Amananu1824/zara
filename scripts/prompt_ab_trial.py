
"""Compare Zara's current behavior prompt with two bounded alternatives.

Evaluation only:
- Does not edit production files.
- Does not save a result file.
- Uses the existing read-only evaluation path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import eval_runner
from app.ai.chat_engine import ChatEngine
from app.ai.ollama_client import OllamaClient
from app.core.config import settings
from app.core.prompt_manager import PromptManager
from app.terminal.language import normalize_language_mode


# Six representative situations, not just one type of conversation.
CASE_IDS = (
    "QUALITY-CHAT-001",  # Casual conversation
    "EMO-003",           # Good news / achievement
    "SOCIAL-002",        # Listening without unwanted solutions
    "EMO-009",           # Frustration
    "PLAY-001",          # Playful conversation
    "OVERLOAD-003",      # Tired; does not want heavy planning
)


CANDIDATE_B = """UNDERSTAND BEFORE RESPONDING
- Determine the user's main intent from the complete message and relevant history. An emotion keyword alone does not determine the response.
- When the user shares an achievement, good news, a joke, a personal update, or a casual greeting without asking for help, respond naturally to that moment first. Do not turn sharing into advice, troubleshooting, a task, or a question.
- For a direct question or request, answer what was asked and stop when complete. Do not let emotional-response rules replace the requested answer.
- For frustration or a correction, identify the concrete issue, acknowledge it briefly when appropriate, and address it rather than offering generic reassurance.
- For sadness, disappointment, uncertainty, or mixed feelings, respond gently and proportionately without forced positivity or assumptions beyond what the user said.
- Give advice, troubleshooting, plans, and follow-up questions only when requested or genuinely needed to answer.
- Do not guess the user's mood from a greeting, repeat their words as filler, force catchphrases, or append a habitual offer to help."""


CANDIDATE_C = """UNDERSTAND BEFORE RESPONDING
- Priority: answer the explicit request first; otherwise respond to the social meaning of what the user shared. Use only relevant conversation history.
- A shared achievement gets a brief, genuine celebration; a joke gets a fitting playful reaction; a casual greeting gets a natural greeting. Do not invent a new task or ask what the user needs.
- A direct question gets a direct answer. Once the request is satisfied, stop instead of adding a routine question, offer, or extra advice.
- When the user is frustrated or corrects you, focus on the specific problem and repair it. Avoid empty apologies or repeating the same mistake.
- When the user shares sadness or mixed feelings, acknowledge what they actually said with proportionate warmth; do not force optimism or infer unstated emotions.
- Advice and next steps belong only when requested or necessary to answer the request. Never use the same catchphrase or response pattern in every conversation."""


VARIANTS = (
    ("BASELINE", None),
    ("CANDIDATE_B", CANDIDATE_B),
    ("CANDIDATE_C", CANDIDATE_C),
)


def replace_behavior_section(
    full_prompt: str,
    new_section: str,
) -> str:
    """Replace only the conversational-behavior section in memory."""
    start_marker = "UNDERSTAND BEFORE RESPONDING"
    end_marker = "PERSONALITY AND STYLE"

    start = full_prompt.find(start_marker)
    end = full_prompt.find(
        end_marker,
        start + len(start_marker),
    )

    if start < 0 or end < 0 or end <= start:
        raise ValueError(
            "Expected prompt headings were not found. "
            "No project file was changed."
        )

    return (
        full_prompt[:start]
        + new_section.strip()
        + "\n\n"
        + full_prompt[end:]
    )


def make_prompt_manager(
    section: str | None,
) -> type[PromptManager]:
    """Create an in-memory prompt variant for one evaluation run."""
    if section is None:
        return PromptManager

    class TrialPromptManager(PromptManager):
        def __init__(self) -> None:
            super().__init__()
            self.combined_prompt = replace_behavior_section(
                self.combined_prompt,
                section,
            )

    return TrialPromptManager


def behavior_signals(answer: str) -> dict[str, bool]:
    """Report basic warning signals; these are not a quality score."""
    text = (answer or "").strip()

    return {
        "ends_with_question": text.endswith("?"),
        "stock_follow_up": bool(
            eval_runner.STOCK_FOLLOW_UP_PATTERN.search(text)
        ),
        "repeated_sentence": bool(
            eval_runner.find_repeated_sentences(text)
        ),
        "devanagari_leak": bool(
            eval_runner.DEVANAGARI_PATTERN.search(text)
        ),
    }


def run() -> int:
    cases = eval_runner.load_cases(eval_runner.DEFAULT_DATASET)
    by_id = {str(case["id"]): case for case in cases}

    missing = [
        case_id
        for case_id in CASE_IDS
        if case_id not in by_id
    ]

    if missing:
        print(
            "Evaluation dataset is missing IDs: "
            + ", ".join(missing)
        )
        return 2

    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _ in VARIANTS
    }

    total = len(CASE_IDS) * len(VARIANTS)
    done = 0

    print("Zara prompt pilot comparison")
    print(
        f"Cases: {len(CASE_IDS)} | "
        f"variants: {len(VARIANTS)} | runs: {total}"
    )
    print("Production prompt: unchanged")
    print("Result files: none created")
    print("Database: existing read-only evaluation path")
    print(
        "Important: signal counts are warning indicators, "
        "not a complete naturalness score.\n"
    )

    # Keep each case's three variants close together for easier comparison.
    for case_id in CASE_IDS:
        case = by_id[case_id]

        for variant_name, section in VARIANTS:
            done += 1

            print(
                f"[{done}/{total}] {variant_name} | "
                f"{case_id} | {case['user_message']}",
                flush=True,
            )

            manager_cls = make_prompt_manager(section)

            result = eval_runner.run_one_case(
                case,
                settings=settings,
                ChatEngine=ChatEngine,
                OllamaClient=OllamaClient,
                PromptManager=manager_cls,
                normalize_language_mode=normalize_language_mode,
            )

            signals = behavior_signals(result.get("answer", ""))
            result["behavior_signals"] = signals
            result["variant"] = variant_name
            results[variant_name].append(result)

            print(
                f"  verdict={result.get('verdict')} | "
                f"seconds={result.get('duration_seconds')}"
            )

            active_signals = [
                name
                for name, flagged in signals.items()
                if flagged
            ]

            print(
                "  signals="
                + (
                    ", ".join(active_signals)
                    if active_signals
                    else "none of the four basic flags"
                )
            )

            if result.get("error"):
                print("  ERROR: " + str(result["error"]))
            else:
                answer = str(result.get("answer", ""))
                print(
                    "  Zara: "
                    + answer.replace("\n", "\n        ")
                )

            if result.get("verdict") == "fail":
                for reason in result.get("reasons", []):
                    print("  mechanical failure: " + str(reason))

            print(flush=True)

    print("=" * 72)
    print("SUMMARY — not an automatic winner ranking")

    signal_names = (
        "ends_with_question",
        "stock_follow_up",
        "repeated_sentence",
        "devanagari_leak",
    )

    for variant_name, _ in VARIANTS:
        rows = results[variant_name]

        hard_failures = sum(
            row.get("verdict") in {"fail", "error"}
            for row in rows
        )

        review_needed = sum(
            row.get("verdict") == "review_needed"
            for row in rows
        )

        durations = [
            row["duration_seconds"]
            for row in rows
            if isinstance(
                row.get("duration_seconds"),
                (int, float),
            )
        ]

        avg_seconds = round(mean(durations), 2) if durations else None

        print(
            f"{variant_name}: "
            f"hard_fail_or_error={hard_failures}, "
            f"review_needed={review_needed}, "
            f"avg_seconds={avg_seconds}"
        )

        signal_counts = {
            signal: sum(
                bool(row.get("behavior_signals", {}).get(signal))
                for row in rows
            )
            for signal in signal_names
        }

        print(
            "  signal counts: "
            + ", ".join(
                f"{name}={count}/{len(rows)}"
                for name, count in signal_counts.items()
            )
        )

    print(
        "\nA 'review_needed' verdict is not automatically a failure. "
        "The existing evaluator cannot fully judge naturalness."
    )
    print("No candidate was applied to Zara's production prompt.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
