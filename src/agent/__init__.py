class MemoryAgent:
    def __init__(self, store, model: str = "models/gemini-1.0-pro"):
        self.store = store
        self.model = model
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
