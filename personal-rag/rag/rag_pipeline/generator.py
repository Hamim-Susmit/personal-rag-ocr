from __future__ import annotations

import requests


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("response", "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc
