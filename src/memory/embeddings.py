
# src/memory/embeddings.py

import os
from typing import List

from google import genai
from google.genai import types


# ---- Gemini Client (Singleton) ----------------------------------------------

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


# ---- Public API --------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """
    Generate a full-resolution Matryoshka embedding.

    IMPORTANT:
    - Always returns the full vector (3072 dims)
    - Truncation is handled elsewhere
    - No normalization here
    """

    if not text or not text.strip():
        raise ValueError("embed_text received empty text")

    client = _get_client()

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[text],  # explicit list
        config=types.EmbedContentConfig(
            output_dimensionality=3072
        )
    )

    if not response.embeddings or len(response.embeddings) != 1:
        raise RuntimeError("Unexpected embeddings response shape")

    vector = response.embeddings[0].values

    if len(vector) != 3072:
        raise RuntimeError(
            f"Expected 3072-dim embedding, got {len(vector)}"
        )

    return vector
