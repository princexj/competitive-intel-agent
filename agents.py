"""Compatibility facade for the v2 typed LangGraph workflow."""

from graph import build_intel_graph


def build_agent_graph(
    groq_api_key: str,
    tavily_api_key: str,
    model: str = "llama-3.3-70b-versatile",
):
    """Return the compiled v2 graph.

    Kept for callers that imported the original function name.
    """
    return build_intel_graph(groq_api_key, tavily_api_key, model=model)
