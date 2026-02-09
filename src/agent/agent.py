# src/agent/agent.py
import os
from dotenv import load_dotenv
from google import genai

from src.memory.search import funnel_search
from src.agent.prompts import build_prompt

load_dotenv()


class MemoryAgent:
    def __init__(self, store, model: str = "gemini-1.0-pro"):
        self.store = store
        self.client = genai.Client()
        self.model = model

    def answer(self, query: str) -> str:
    results = funnel_search(
        query=query,
        store=self.store,
        coarse_k=50,
        top_k=5,
    )

    if not results:
        return "No relevant memory found."

    context = "\n".join(
        r["payload"]["text"] for r in results if r.get("payload")
    )

    prompt = build_prompt(query, context)

    try:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    except Exception as e:
        return (
            "Text generation is currently unavailable.\n\n"
            "Reason: Gemini API access requires billing to be enabled.\n"
            "Memory retrieval and agent orchestration are active.\n"
            "Once billing is enabled, this will automatically activate."
        )

