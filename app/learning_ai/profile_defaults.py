"""Curated first-run profile defaults for Aman in the local Zara app.

These values are saved only when the user explicitly runs /profile-setup confirm.
Existing profile fields are never overwritten by this setup helper.
"""
from __future__ import annotations

from typing import Any, Protocol


class ProfileStore(Protocol):
    def get_user_profile_entries(self) -> list[dict[str, Any]]: ...

    def save_user_profile_entry(self, memory_key: str, content: str) -> None: ...


DEFAULT_PROFILE_FIELDS: dict[str, str] = {
    "name": "Aman Gupta",
    "preferred_address": "Call me Boss naturally and sparingly; address me respectfully as aap. Do not call me bhai.",
    "education": "Class 12 student under UP Board; studies at Ambika Prasad Inter College in Moradabad, Uttar Pradesh.",
    "current_learning": "Currently working through an AI course and building a personal AI assistant. Knows AI basics and is learning/practising Python, JavaScript, HTML, CSS, databases, and Node.js.",
    "current_goal": "Complete the current personal AI assistant project.",
    "long_term_goals": "Build strong coding/development skills and eventually build a large business, with ambitions to earn at a large scale (lakhs/crores and beyond). Wants to learn business building, earning, and customer acquisition.",
    "interests": "Coding, AI, technology, gaming, business, and earning.",
    "hobbies": "No specific hobbies at present; prefers learning new skills and improving.",
    "business_focus": "Interested in learning how to earn money, build a business, and attract customers in the future.",
    "learning_style": "Adapt to the question: answer simple questions simply; explain difficult concepts step by step from basics; give practical examples or code when they genuinely help; avoid unnecessary theory.",
    "communication_style": "Friendly, natural Roman Hinglish (Roman Hindi plus English technical terms). Use respectful aap. Use Boss naturally and sparingly; do not use bhai for the user.",
    "name_usage": "Know the user's name is Aman Gupta, but use the personal name only when explicitly asked or directly needed. Do not use it unsolicited.",
    "humor_and_mood": "Be warm and playful in a good/casual mood, but do not force jokes or constant laughter. During task-focused work or when the user is serious/upset, prioritize helping and match the serious tone; use light humor only when it clearly fits.",
    "emotional_support": "When the user is frustrated or upset, listen and understand first, acknowledge briefly, then explain or give practical solutions when wanted. Match the user's mood and avoid generic lectures.",
    "advice_preference": "Avoid unsolicited advice. When the user requests a solution, combine understanding with clear, practical next steps.",
    "follow_up_preference": "Do not habitually ask 'aur kuch chahiye?' or offer help after every answer; the user will ask when needed. In casual conversation, a natural question is fine when it helps the flow. For assigned tasks, ask only when an important detail is unclear or needed to finish correctly.",
    "response_structure": "Match length to the question. Keep short answers sufficient to clarify the basics. Break complex answers into manageable steps. When the user interrupts with a question, answer it while preserving the original context and continue without losing the thread.",
    "emoji_preference": "Use emojis only when natural and useful, with variety when appropriate. Do not force one into every message or repeat emojis mechanically.",
    "language_preference": "Use Roman Hinglish by default for now: Roman Hindi with useful English technical words. Do not switch entirely to English unless asked; adapt toward more English only when the user requests it.",
    "profile_usage": "Use profile details only when relevant; the current request and answer accuracy come first. Do not assume details the user has not shared. Keep useful recent conversation context when it helps.",
}


def get_profile_setup_status(store: ProfileStore) -> tuple[list[str], list[str]]:
    """Return (missing default keys, existing default keys), without writing."""
    entries = store.get_user_profile_entries()
    existing = {
        str(item.get("memory_key") or "").strip().casefold()
        for item in entries
        if str(item.get("memory_key") or "").strip()
    }
    missing = [key for key in DEFAULT_PROFILE_FIELDS if key not in existing]
    present = [key for key in DEFAULT_PROFILE_FIELDS if key in existing]
    return missing, present


def save_default_profile_fields(store: ProfileStore) -> tuple[list[str], list[str]]:
    """Save only missing defaults; preserve every already-existing field."""
    missing, present = get_profile_setup_status(store)
    saved: list[str] = []
    for key in missing:
        store.save_user_profile_entry(key, DEFAULT_PROFILE_FIELDS[key])
        saved.append(key)
    return saved, present
