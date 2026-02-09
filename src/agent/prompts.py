# src/agent/prompts.py

SYSTEM_PROMPT = """
You are a long-lived autonomous agent.
You answer questions using retrieved memory.
If memory is insufficient, say so explicitly.
"""

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
{SYSTEM_PROMPT}

=== Retrieved Memory ===
{context}

=== User Question ===
{query}

Answer concisely and accurately.
"""
