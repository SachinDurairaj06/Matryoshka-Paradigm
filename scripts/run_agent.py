# scripts/run_agent.py

from src.agent.agent import MemoryAgent


# --- Mock store reused from Split 1 test ---
class MockStore:
    def __init__(self, items):
        self.items = items

    def search_coarse(self, vector, limit):
        return self.items[:limit]


if __name__ == "__main__":
    mock_items = [
        {
            "id": "doc-1",
            "vector_full": [0.1] * 3072,
            "payload": {"text": "Matryoshka embeddings allow truncation-safe retrieval."},
        },
        {
            "id": "doc-2",
            "vector_full": [0.2] * 3072,
            "payload": {"text": "Funnel search uses low-dim recall and high-dim re-ranking."},
        },
    ]

    store = MockStore(mock_items)
    agent = MemoryAgent(store)

    while True:
        q = input("\nAsk agent> ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\nAgent:\n", agent.answer(q))
