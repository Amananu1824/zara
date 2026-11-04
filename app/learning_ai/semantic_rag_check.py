"""Index current behavioral knowledge and benchmark semantic retrieval."""
from __future__ import annotations

from time import perf_counter

from app.learning_ai.learning_store import LearningStore


TEST_QUERIES = [
    "Are, friendly baatein toh kar sakte hain na?",
    "Mera coding project phir fail ho gaya, ab irritate ho raha hoon.",
    "Please correct my grammar and spelling.",
    "How should Zara respond naturally in a conversation?",
]


def main() -> None:
    store = LearningStore()

    print("=== Step 1: Index approved behavioral knowledge ===")
    started = perf_counter()
    stats = store.index_pending_embeddings(batch_size=8)
    elapsed = perf_counter() - started

    print(f"Index statistics: {stats}")
    print(f"Indexing time: {elapsed:.2f} seconds\n")

    print("=== Step 2: Benchmark semantic retrieval ===")

    for query in TEST_QUERIES:
        started = perf_counter()
        results = store.semantic_search(query, limit=4)
        elapsed = perf_counter() - started

        print(f"\nQuery: {query}")
        print(f"Query + retrieval time: {elapsed:.2f} seconds")

        if not results:
            print("No indexed results found.")
            continue

        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. [{item['source']}] "
                f"category={item['category']} "
                f"similarity={float(item['similarity']):.4f}"
            )
            print(f"   {item['title']}")
            print(f"   {item['content'][:260]}\n")


if __name__ == "__main__":
    main()