"""Small Ollama chat client used by the character design workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    timeout: int = 180
    keep_alive: str = "0s"


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    @property
    def chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/chat"

    @property
    def tags_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/api/tags"

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = requests.get(self.tags_url, timeout=8)
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Ollama is not reachable at {self.config.base_url}: {exc}"

        models = response.json().get("models", [])
        names = {model.get("name") for model in models}
        if self.config.model not in names:
            available = ", ".join(sorted(name for name in names if name)) or "no models found"
            return False, f"Model {self.config.model!r} was not found. Available: {available}"
        return True, f"Ollama ready with {self.config.model}."

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "top_p": top_p,
            },
            # Release Gemma from VRAM after prompt generation so SDXL ControlNet
            # can use the RTX 4080's memory for image synthesis.
            "keep_alive": self.config.keep_alive,
        }
        response = requests.post(self.chat_url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Ollama returned an unexpected response: {data}")
        return str(content).strip()
