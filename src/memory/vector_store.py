import os
import numpy as np
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

# -------------------- Client -----------------------------------------------

_qdrant = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant = QdrantClient(url=url)
    return _qdrant


# -------------------- Utils -------------------------------------------------

def _l2_normalize(vec: List[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        raise ValueError("Zero-norm vector encountered")
    return (arr / norm).tolist()


# -------------------- Public API --------------------------------------------

def init_vector_store(collection_name: str):
    """
    Create the Matryoshka vector collection if it does not exist.
    """

    client = _get_qdrant()

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "mrl_coarse": VectorParams(
                size=128,
                distance=Distance.COSINE,
            ),
            "mrl_full": VectorParams(
                size=3072,
                distance=Distance.COSINE,
            ),
        },
    )


def upsert_chunk(
    collection_name: str,
    chunk_id: str,
    embedding: List[float],
    payload: dict,
):
    """
    Store one chunk with Matryoshka vectors.
    """

    if len(embedding) != 3072:
        raise ValueError("Expected full 3072-dim embedding")

    # --- Matryoshka slicing (THIS IS THE CORE IDEA) ---
    coarse = embedding[:128]
    full = embedding

    # --- Independent normalization (NON-NEGOTIABLE) ---
    coarse = _l2_normalize(coarse)
    full = _l2_normalize(full)

    point = PointStruct(
        id=chunk_id,
        vector={
            "mrl_coarse": coarse,
            "mrl_full": full,
        },
        payload=payload,
    )

    client = _get_qdrant()
    client.upsert(
        collection_name=collection_name,
        points=[point],
    )
