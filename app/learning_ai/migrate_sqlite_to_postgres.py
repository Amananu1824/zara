"""One-time, non-destructive migration from legacy SQLite to PostgreSQL."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from app.learning_ai.learning_store import LearningStore


LEGACY_DATABASE = (
    Path(__file__).resolve().parents[1] / "data" / "zara_learning.db"
)
MIGRATION_KEY = "legacy_sqlite_learning_store_v1"


def read_table(
    connection: sqlite3.Connection,
    tables: set[str],
    table_name: str,
) -> list[dict]:
    """Read a known legacy table, returning an empty list if it is absent."""
    if table_name not in tables:
        return []

    # Table names here come only from the fixed list below.
    rows = connection.execute(
        f"SELECT * FROM {table_name}"
    ).fetchall()

    return [dict(row) for row in rows]


def reset_sequence(connection, table_name: str) -> None:
    """Advance an imported ID sequence beyond the highest existing ID."""
    if table_name not in {"feedback", "evaluation_cases"}:
        raise ValueError("Unexpected table name for sequence reset.")

    connection.execute(
        f"""
        SELECT setval(
            pg_get_serial_sequence('{table_name}', 'id'),
            GREATEST(COALESCE((SELECT MAX(id) FROM {table_name}), 1), 1),
            EXISTS (SELECT 1 FROM {table_name})
        )
        """
    )


def main() -> None:
    store = LearningStore()

    if not LEGACY_DATABASE.is_file():
        print(f"Legacy SQLite database not found: {LEGACY_DATABASE}")
        print("No legacy data was migrated.")
        return

    with sqlite3.connect(LEGACY_DATABASE) as legacy:
        legacy.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in legacy.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        guides = read_table(legacy, tables, "behavior_guides")
        feedback = read_table(legacy, tables, "feedback")
        evaluations = read_table(legacy, tables, "evaluation_cases")

    with store._connect() as connection:
        already_done = connection.execute(
            "SELECT 1 FROM data_migrations WHERE migration_key = %s",
            (MIGRATION_KEY,),
        ).fetchone()

        if already_done:
            print("Legacy SQLite migration was already recorded.")
            print("No duplicate import was performed.")
            return

        guide_count = 0
        for row in guides:
            connection.execute(
                """
                INSERT INTO behavior_guides
                    (guide_key, category, title, keywords, content,
                     approved, source_type, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, 'sqlite_migrated', NOW())
                ON CONFLICT (guide_key)
                DO UPDATE SET
                    category = EXCLUDED.category,
                    title = EXCLUDED.title,
                    keywords = EXCLUDED.keywords,
                    content = EXCLUDED.content,
                    approved = EXCLUDED.approved,
                    source_type = 'sqlite_migrated',
                    updated_at = NOW()
                WHERE behavior_guides.source_type = 'built_in'
                """,
                (
                    row["guide_key"],
                    row["category"],
                    row["title"],
                    row["keywords"],
                    row["content"],
                    bool(row["approved"]),
                ),
            )
            guide_count += 1

        feedback_count = 0
        for row in feedback:
            connection.execute(
                """
                INSERT INTO feedback
                    (id, created_at, user_message, assistant_response,
                     rating, note, reviewed)
                VALUES (
                    %s,
                    COALESCE(%s::timestamptz, NOW()),
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    row["id"],
                    row.get("created_at"),
                    row["user_message"],
                    row["assistant_response"],
                    int(row["rating"]),
                    row.get("note") or "",
                    bool(row.get("reviewed", 0)),
                ),
            )
            feedback_count += 1

        evaluation_count = 0
        for row in evaluations:
            connection.execute(
                """
                INSERT INTO evaluation_cases
                    (id, name, user_message, criteria, active)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    row["id"],
                    row["name"],
                    row["user_message"],
                    row["criteria"],
                    bool(row.get("active", 1)),
                ),
            )
            evaluation_count += 1

        # Rebuild generalized lessons for previously reviewed bad feedback.
        reviewed_rows = connection.execute(
            """
            SELECT id, user_message, note
            FROM feedback
            WHERE rating = -1
              AND reviewed = TRUE
              AND TRIM(note) <> ''
            """
        ).fetchall()

        for row in reviewed_rows:
            store._upsert_feedback_lessons(
                connection,
                int(row["id"]),
                str(row["user_message"]),
                str(row["note"]),
            )

        reset_sequence(connection, "feedback")
        reset_sequence(connection, "evaluation_cases")

        details = (
            f"Guides processed: {guide_count}; "
            f"feedback records processed: {feedback_count}; "
            f"evaluation cases processed: {evaluation_count}."
        )
        connection.execute(
            """
            INSERT INTO data_migrations (migration_key, details)
            VALUES (%s, %s)
            """,
            (MIGRATION_KEY, details),
        )

    print("SQLite migration completed.")
    print(f"Guides processed: {guide_count}")
    print(f"Feedback records processed: {feedback_count}")
    print(f"Evaluation cases processed: {evaluation_count}")
    print("The original SQLite file has not been changed.")


if __name__ == "__main__":
    main()