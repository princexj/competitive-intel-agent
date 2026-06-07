"""Typed contracts shared by the competitive intelligence workflow."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Source(TypedDict):
    title: str
    url: str
    snippet: str
    source: str
    published: str
    citations: int | None
    category: str


class ResearchQueries(TypedDict):
    news: str
    technical: str
    community: str


class IntelState(TypedDict, total=False):
    topic: str
    queries: ResearchQueries
    news_sources: list[Source]
    research_sources: list[Source]
    community_sources: list[Source]
    analysis: dict
    report: str
    errors: Annotated[list[str], operator.add]


class TargetResult(TypedDict):
    topic: str
    analysis: dict
    report: str
    sources: list[Source]
