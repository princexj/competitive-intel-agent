"""Minimal Groq client used by the graph nodes."""

from __future__ import annotations

import json
import re
from typing import Any

import requests


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 90,
    ) -> None:
        if not api_key:
            raise ValueError("A Groq API key is required.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        content = self.complete(system, user, json_mode=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("The model did not return valid JSON.")
            return json.loads(match.group(0))
