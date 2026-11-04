"""Start Zara V2 in the terminal."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.terminal.chat_loop import run_chat_loop  # noqa: E402


if __name__ == "__main__":
    run_chat_loop()

    
# SELECT
#     id,
#     topic_key,
#     topic,
#     memory_type,
#     summary,
#     status,
#     active,
#     source_kind,
#     sensitivity,
#     version,
#     supersedes_id,
#     evidence_count,
#     provenance,
#     created_at
# FROM user_topic_memories
# WHERE id BETWEEN 1 AND 4
# ORDER BY id ASC;

# A4 ko activate karne ke liye ek baar terminal mein:
# /lesson-auto-promotion enable


# Then confirmation:
# AUTHORIZE ZARA SYSTEM A PROMOTION

# Example commands, abhi execute nahi karne:
# python scripts/system_a_eval_runner.py


# python scripts/system_a_eval_runner.py --with-model


# python scripts/system_a_eval_runner.py --db-audit


# python scripts/system_a_eval_runner.py --with-model --db-audit


# Purane evaluation se compare:
# python scripts/system_a_eval_runner.py --with-model --db-audit --compare results/system_a_evaluations/previous.json