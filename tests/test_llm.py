from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from llm import GroqClient, GroqRateLimitError


def response(status: int, *, content: str = "ok", retry_after: str = "0") -> Mock:
    result = Mock()
    result.status_code = status
    result.headers = {"Retry-After": retry_after}
    result.text = ""
    if status == 200:
        result.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
    else:
        result.json.return_value = {"error": {"message": "quota exceeded"}}
    return result


class GroqClientTests(unittest.TestCase):
    @patch("llm.time.sleep")
    @patch("llm.requests.post")
    def test_retries_rate_limit_then_succeeds(self, post, sleep):
        post.side_effect = [response(429), response(200, content="done")]
        client = GroqClient("key", max_retries=2)

        self.assertEqual(client.complete("system", "user"), "done")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    @patch("llm.time.sleep")
    @patch("llm.requests.post")
    def test_raises_clear_error_after_retries(self, post, _sleep):
        post.return_value = response(429)
        client = GroqClient("key", max_retries=1)

        with self.assertRaisesRegex(GroqRateLimitError, "automatic retries"):
            client.complete("system", "user")


if __name__ == "__main__":
    unittest.main()
