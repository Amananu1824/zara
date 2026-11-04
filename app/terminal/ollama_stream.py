"""Terminal streaming adapter; model details stay inside ChatEngine/OllamaClient."""
from collections.abc import Iterator

from app.ai.chat_engine import ChatEngine


def stream_chat_response(engine: ChatEngine, user_message: str) -> Iterator[str]:
    yield from engine.stream_reply(user_message)