# """Load the stable Zara prompt once; per-turn controls are added by ChatEngine."""
# from __future__ import annotations

# from pathlib import Path


# PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
# BASE_PROMPT_FILENAME = "00_zara_base.txt"


# class PromptManager:
#     def __init__(self, prompt_directory: Path | None = None) -> None:
#         self.prompt_directory = Path(prompt_directory) if prompt_directory else PROMPT_DIRECTORY
#         self.combined_prompt = self._load_base_prompt()

#     def _load_base_prompt(self) -> str:
#         path = self.prompt_directory / BASE_PROMPT_FILENAME
#         if not path.is_file():
#             raise FileNotFoundError(
#                 f"Zara base prompt is missing: {path}\n"
#                 "Create app/prompts/00_zara_base.txt before starting Zara."
#             )
#         content = path.read_text(encoding="utf-8").strip()
#         if not content:
#             raise ValueError(f"Zara base prompt is empty: {path}")
#         return content

#     def get_system_prompt(self) -> str:
#         return self.combined_prompt

#     def reload(self) -> None:
#         """Reload the base prompt after the text file has been edited."""
#         self.combined_prompt = self._load_base_prompt()


"""Load Zara's stable runtime system prompt.

Behavior examples are intentionally not loaded into the runtime prompt.
They are kept as offline developer/evaluation references only, because
including reply examples in every turn can encourage copying and repetition.
"""
from __future__ import annotations

from pathlib import Path


PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"
BASE_PROMPT_FILENAME = "00_zara_base.txt"


class PromptManager:
    def __init__(self, prompt_directory: Path | None = None) -> None:
        self.prompt_directory = (
            Path(prompt_directory) if prompt_directory else PROMPT_DIRECTORY
        )
        self.combined_prompt = self._load_base_prompt()

    def _load_base_prompt(self) -> str:
        path = self.prompt_directory / BASE_PROMPT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"Zara base prompt is missing: {path}\n"
                "Create app/prompts/00_zara_base.txt before starting Zara."
            )

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Zara base prompt is empty: {path}")

        # Intentionally do not load 01_behavior_examples.txt here.
        # Keep examples outside runtime inference to prevent answer copying.
        return content

    def get_system_prompt(self) -> str:
        return self.combined_prompt

    def reload(self) -> None:
        """Reload the base prompt after its text file has been edited."""
        self.combined_prompt = self._load_base_prompt()
