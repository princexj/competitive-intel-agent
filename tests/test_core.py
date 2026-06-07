from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from graph import build_intel_graph
from storage import get_run, list_runs, save_run
from tools import deduplicate_sources


def source(title: str, url: str, category: str = "news") -> dict:
    return {
        "title": title,
        "url": url,
        "snippet": f"Evidence about {title}",
        "source": "Test",
        "published": "2026-06-01",
        "citations": None,
        "category": category,
    }


class FakeGroqClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def complete_json(self, system: str, _user: str) -> dict:
        if "three concise search queries" in system:
            return {
                "news": "target news",
                "technical": "target research",
                "community": "target developers",
            }
        return {
            "summary": ["A", "B", "C"],
            "market_position": "Strong",
            "technical_direction": "Focused",
            "community_sentiment": "Positive",
            "opportunities": ["Expansion"],
            "risks": ["Competition"],
            "signals": [],
            "scores": {
                "momentum": 80,
                "technical_strength": 75,
                "community_interest": 70,
                "opportunity": 85,
            },
        }

    def complete(self, *_args, **_kwargs) -> str:
        return "# Test report\n\nEvidence-backed output."


class CoreTests(unittest.TestCase):
    def test_deduplicate_sources_prefers_richer_record(self):
        short = source("One", "https://example.com/item")
        rich = source("One updated", "https://example.com/item/")
        rich["snippet"] = "A much longer and more useful evidence snippet."
        self.assertEqual(deduplicate_sources([short, rich]), [rich])

    def test_storage_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "history.db"
            results = [
                {
                    "topic": "Acme",
                    "analysis": {"scores": {}},
                    "report": "# Acme",
                    "sources": [],
                }
            ]
            run_id = save_run(results, "", db_path)
            loaded = get_run(run_id, db_path)
            self.assertEqual(loaded["results"], results)
            self.assertEqual(list_runs(db_path=db_path)[0]["id"], run_id)

    @patch("graph.fetch_web_results")
    @patch("graph.fetch_semantic_scholar")
    @patch("graph.fetch_arxiv_papers")
    @patch("graph.GroqClient", FakeGroqClient)
    def test_graph_executes_parallel_research_and_report(
        self, arxiv_mock, scholar_mock, web_mock
    ):
        arxiv_mock.return_value = [source("Paper", "https://arxiv.org/1", "research")]
        scholar_mock.return_value = []
        web_mock.side_effect = [
            [source("News", "https://example.com/news")],
            [source("Community", "https://example.com/community", "community")],
        ]

        graph = build_intel_graph("test-groq", "test-tavily")
        updates = list(
            graph.stream(
                {"topic": "Acme", "errors": []},
                stream_mode="updates",
            )
        )
        nodes = {next(iter(update)) for update in updates}
        self.assertEqual(
            nodes,
            {
                "planner",
                "news_researcher",
                "paper_researcher",
                "community_researcher",
                "analyst",
                "reporter",
            },
        )
        report_update = next(update["reporter"] for update in updates if "reporter" in update)
        self.assertIn("Test report", report_update["report"])


if __name__ == "__main__":
    unittest.main()
