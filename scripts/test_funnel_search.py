# scripts/test_funnel_search.py

from src.memory.search import funnel_search


# -------------------- Mock Store --------------------------------------------

class MockStore:
    """
    Minimal in-memory stand-in for a vector DB.
    """

    def __init__(self, items):
        self.items = items

    def search_coarse(self, vector, limit):
        # Ignore vector; just return top-N mock candidates
        return self.items[:limit]


# -------------------- Test --------------------------------------------------

if __name__ == "__main__":
    mock_items = [
        {
            "id": "doc-1",
            "vector_full": [0.1] * 3072,
            "payload": {"text": "first document"},
        },
        {
            "id": "doc-2",
            "vector_full": [0.2] * 3072,
            "payload": {"text": "second document"},
        },
        {
            "id": "doc-3",
            "vector_full": [0.05] * 3072,
            "payload": {"text": "third document"},
        },
    ]

    store = MockStore(mock_items)

    results = funnel_search(
        query="test query",
        store=store,
        coarse_k=3,
        top_k=2,
    )

    print("Results:")
    for r in results:
        print(r)
