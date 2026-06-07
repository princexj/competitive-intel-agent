"""LangGraph workflow for evidence-backed competitive intelligence."""

from __future__ import annotations

import json
from datetime import date
from typing import Iterator

from langgraph.graph import END, START, StateGraph

from llm import GroqClient
from state import IntelState, Source, TargetResult
from tools import (
    deduplicate_sources,
    fetch_arxiv_papers,
    fetch_semantic_scholar,
    fetch_web_results,
)

PLANNER_PROMPT = """You are a competitive intelligence research lead.
Return JSON with exactly three concise search queries: news, technical, community.
The news query should prioritize recent market activity. The technical query should
find substantive research. The community query should capture practitioner opinion."""

ANALYST_PROMPT = """You are a skeptical senior competitive intelligence analyst.
Use only the supplied evidence. Distinguish facts from inference and never invent a
source. Return JSON with these keys:
summary (three short strings), market_position (string), technical_direction
(string), community_sentiment (string), opportunities (array of strings),
risks (array of strings), signals (array of objects with label, evidence, confidence
where confidence is high/medium/low), and scores (object with momentum,
technical_strength, community_interest, opportunity; each integer 0-100).
Mention uncertainty and evidence gaps directly."""

REPORTER_PROMPT = """You are an exacting strategy writer. Create a polished Markdown
competitive intelligence brief using only the supplied analysis and evidence.
Include: Executive Summary, Scorecard, Market Position, Technical Landscape,
Community Signal, Opportunities, Risks & Watch Items, Confidence & Gaps, and Sources.
Every material claim must use numbered source citations like [1]. Include a numbered
source list with title, publisher, date when available, and URL. Do not add facts."""

COMPARISON_PROMPT = """You are a strategy consultant comparing competitive targets.
Create a concise Markdown comparison with: Executive Verdict, side-by-side score
table, Where Each Target Wins, Risks, and Strategic Recommendation. Use only the
provided target analyses and make uncertainty explicit."""


def _source_context(sources: list[Source]) -> str:
    return "\n".join(
        f"[{index}] {source['title']} | {source['source']} | "
        f"{source['published'] or 'date unavailable'} | {source['url']}\n"
        f"{source['snippet']}"
        for index, source in enumerate(sources, start=1)
    )


def build_intel_graph(
    groq_api_key: str, tavily_api_key: str, model: str = "llama-3.3-70b-versatile"
):
    llm = GroqClient(groq_api_key, model=model)

    def plan(state: IntelState) -> dict:
        topic = state["topic"]
        try:
            queries = llm.complete_json(
                PLANNER_PROMPT,
                f"Target: {topic}\nToday: {date.today().isoformat()}",
            )
            required = {"news", "technical", "community"}
            if not required.issubset(queries):
                raise ValueError("Planner response is missing required queries.")
            return {"queries": {key: str(queries[key]) for key in required}}
        except Exception as exc:
            return {
                "queries": {
                    "news": f"{topic} latest news funding launches strategy",
                    "technical": f"{topic} research architecture benchmark",
                    "community": f"{topic} developer experience reviews",
                },
                "errors": [f"Planner fallback used: {exc}"],
            }

    def research_news(state: IntelState) -> dict:
        try:
            return {
                "news_sources": fetch_web_results(
                    state["queries"]["news"], tavily_api_key, category="news"
                )
            }
        except Exception as exc:
            return {"news_sources": [], "errors": [f"News search failed: {exc}"]}

    def research_papers(state: IntelState) -> dict:
        sources: list[Source] = []
        errors: list[str] = []
        try:
            sources.extend(fetch_arxiv_papers(state["queries"]["technical"]))
        except Exception as exc:
            errors.append(f"arXiv search failed: {exc}")
        try:
            sources.extend(fetch_semantic_scholar(state["queries"]["technical"]))
        except Exception as exc:
            errors.append(f"Semantic Scholar search failed: {exc}")
        update: dict = {"research_sources": deduplicate_sources(sources)}
        if errors:
            update["errors"] = errors
        return update

    def research_community(state: IntelState) -> dict:
        try:
            return {
                "community_sources": fetch_web_results(
                    state["queries"]["community"],
                    tavily_api_key,
                    category="community",
                )
            }
        except Exception as exc:
            return {
                "community_sources": [],
                "errors": [f"Community search failed: {exc}"],
            }

    def analyze(state: IntelState) -> dict:
        sources = deduplicate_sources(
            state.get("news_sources", [])
            + state.get("research_sources", [])
            + state.get("community_sources", [])
        )
        if not sources:
            raise RuntimeError("No research evidence was collected.")
        analysis = llm.complete_json(
            ANALYST_PROMPT,
            f"Target: {state['topic']}\n\nEvidence:\n{_source_context(sources)}",
        )
        return {"analysis": analysis}

    def report(state: IntelState) -> dict:
        sources = deduplicate_sources(
            state.get("news_sources", [])
            + state.get("research_sources", [])
            + state.get("community_sources", [])
        )
        content = llm.complete(
            REPORTER_PROMPT,
            f"Target: {state['topic']}\nGenerated: {date.today().isoformat()}\n\n"
            f"Analysis:\n{json.dumps(state['analysis'], indent=2)}\n\n"
            f"Evidence:\n{_source_context(sources)}",
            temperature=0.1,
        )
        return {"report": content}

    workflow = StateGraph(IntelState)
    workflow.add_node("planner", plan)
    workflow.add_node("news_researcher", research_news)
    workflow.add_node("paper_researcher", research_papers)
    workflow.add_node("community_researcher", research_community)
    workflow.add_node("analyst", analyze)
    workflow.add_node("reporter", report)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "news_researcher")
    workflow.add_edge("planner", "paper_researcher")
    workflow.add_edge("planner", "community_researcher")
    workflow.add_edge(
        ["news_researcher", "paper_researcher", "community_researcher"], "analyst"
    )
    workflow.add_edge("analyst", "reporter")
    workflow.add_edge("reporter", END)
    return workflow.compile()


def stream_target(
    topic: str,
    groq_api_key: str,
    tavily_api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> Iterator[tuple[str, dict]]:
    graph = build_intel_graph(groq_api_key, tavily_api_key, model)
    for update in graph.stream(
        {"topic": topic, "errors": []}, stream_mode="updates"
    ):
        node, values = next(iter(update.items()))
        yield node, values


def result_from_state(topic: str, state: IntelState) -> TargetResult:
    sources = deduplicate_sources(
        state.get("news_sources", [])
        + state.get("research_sources", [])
        + state.get("community_sources", [])
    )
    return {
        "topic": topic,
        "analysis": state.get("analysis", {}),
        "report": state.get("report", ""),
        "sources": sources,
    }


def compare_targets(
    results: list[TargetResult],
    groq_api_key: str,
    model: str = "llama-3.3-70b-versatile",
) -> str:
    if len(results) < 2:
        return ""
    payload = [
        {"topic": result["topic"], "analysis": result["analysis"]}
        for result in results
    ]
    return GroqClient(groq_api_key, model=model).complete(
        COMPARISON_PROMPT, json.dumps(payload, indent=2), temperature=0.1
    )
