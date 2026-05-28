import json
import os


class MemoryStore:
    """Persistent conversation memory for the e-commerce support agent."""

    def __init__(self, filepath: str = "data/memory_store.json", max_history: int = 10):
        self.filepath = filepath
        self.max_history = max_history
        self.history = self._load_memory()

    def _load_memory(self) -> list[dict[str, str]]:
        if not os.path.exists(self.filepath):
            return []

        try:
            with open(self.filepath, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("history", []) if isinstance(data, dict) else []
        except Exception:
            return []

    def _save_memory(self) -> None:
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"history": self.history}, f, indent=2, ensure_ascii=False)

    def append(self, role: str, content: str) -> None:
        entry = {"role": role, "content": content}
        self.history.append(entry)
        self.history = self.history[-self.max_history :]
        self._save_memory()

    def add_user(self, message: str) -> None:
        self.append("user", message)

    def add_agent(self, message: str) -> None:
        self.append("assistant", message)

    def get_recent_messages(self) -> list[dict[str, str]]:
        return list(self.history)

    def clear(self) -> None:
        self.history = []
        self._save_memory()


if __name__ == "__main__":
    store = MemoryStore(max_history=8)
    store.add_user("Hello, I need help with order 1001.")
    store.add_agent("I can help with that. What would you like to know?")
    print(store.get_recent_messages())
