# src/memory/search.py

import numpy as np
from typing import List, Dict

from src.memory.embeddings import embed_text


# -------------------- Utils -------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


# -------------------- Funnel Search ----------------------------------------

def funnel_search(
    query: str,
    store,
    *,
    coarse_k: int = 50,
    top_k: int = 5,
) -> List[Dict]:
    """
    Matryoshka Funnel Search.

    Steps:
    1. Embed query at full resolution (3072)
    2. Use prefix (128) for coarse recall
    3. Re-rank candidates using full vectors
    """

    # 1. Embed query
    query_vec_full = embed_text(query)
    query_vec_coarse = query_vec_full[:128]

    # 2. Coarse retrieval (backend-agnostic)
    candidates = store.search_coarse(
        vector=query_vec_coarse,
        limit=coarse_k,
    )

    if not candidates:
        return []

    # 3. Fine re-ranking (exact cosine on full vectors)
    scored = []
    for item in candidates:
        score = _cosine_similarity(
            query_vec_full,
            item["vector_full"],
        )
        scored.append({
            "id": item["id"],
            "score": score,
            "payload": item.get("payload"),
        })

    # 4. Sort and select
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
