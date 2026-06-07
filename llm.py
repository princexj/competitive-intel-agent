"""Minimal Groq client used by the graph nodes."""

from __future__ import annotations

import json
import re
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any

import requests


class GroqRateLimitError(RuntimeError):
    """Raised when Groq remains rate-limited after all retries."""


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 90,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("A Groq API key is required.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, min(float(retry_after), 60.0))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    now = datetime.now(timezone.utc)
                    return max(1.0, min((retry_at - now).total_seconds(), 60.0))
                except (TypeError, ValueError):
                    pass
        return min(2**attempt, 30)

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

        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()

            if attempt == self.max_retries:
                break
            time.sleep(self._retry_delay(response, attempt))

        detail = ""
        if response is not None:
            try:
                detail = response.json().get("error", {}).get("message", "")
            except (AttributeError, ValueError):
                detail = response.text[:300]
        message = "Groq rate limit remained active after automatic retries."
        if detail:
            message += f" {detail}"
        raise GroqRateLimitError(message)

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        content = self.complete(system, user, json_mode=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("The model did not return valid JSON.")
            return json.loads(match.group(0))
