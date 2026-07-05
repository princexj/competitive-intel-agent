# Competitive Intelligence Agent

An autonomous multi-agent research pipeline that generates structured competitive intelligence briefs on any company, technology, or topic — in minutes.

## What it does

Give it a company name or technology (e.g. "Anthropic", "RAG pipelines", "Mistral AI") and it spins up a team of 6 specialized AI agents that:

1. Break your topic into 3 research angles (news, technical, community)
2. Query 3 data sources simultaneously (web, arXiv, Semantic Scholar)
3. Score and filter results for relevance (critic layer)
4. Cross-reference findings across sources
5. Output a structured markdown brief with TL;DR, technical landscape, strategic implications, and research gaps

## Agent Architecture

```
User Input
    │
    ▼
┌─────────┐
│ Planner │  ← Breaks topic into NEWS / TECH / COMMUNITY queries
└────┬────┘
     │
     ├──────────────────────┐
     ▼                      ▼
┌──────────────┐    ┌────────────────┐
│ News         │    │ Paper          │
│ Researcher   │    │ Researcher     │
│ (Tavily Web) │    │ (arXiv +       │
│              │    │  Semantic      │
│              │    │  Scholar)      │
└──────┬───────┘    └───────┬────────┘
       │                    │
       └──────────┬─────────┘
                  ▼
          ┌──────────────┐
          │   Critic     │  ← Scores each result 1-10, drops < 6
          │              │  ← Deduplicates across sources
          └──────┬───────┘
                 ▼
         ┌─────────────┐
         │ Synthesizer │  ← Cross-references news vs research
         └──────┬──────┘
                ▼
          ┌──────────┐
          │ Reporter │  ← Final structured markdown brief
          └──────────┘
```

The agent graph is controlled by a custom `state_transition` function — a state machine where each agent is a node and transitions are conditional on message content, not position. This is more robust than round-robin selection because it handles tool execution flows correctly.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | Microsoft AutoGen (GroupChat) |
| LLM | Llama 3.3 70B via Groq API |
| Web Search | Tavily Search API |
| Academic Search | arXiv API + Semantic Scholar API |
| UI | Streamlit |
| Caching | JSON-based 24h cache layer |

## Setup

```bash
git clone https://github.com/yourusername/competitive-intel-agent
cd competitive-intel-agent
pip install -r requirements.txt
```

## Running

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

You'll be prompted for:
- **Groq API key** — free at [console.groq.com](https://console.groq.com)
- **Tavily API key** — free at [tavily.com](https://tavily.com) (1000 searches/month)

Keys are stored in session memory only — never saved to disk (BYOK pattern).

## Design Decisions

**Why a custom state_transition instead of round-robin?**
Round-robin doesn't account for tool execution flows. When an agent calls a tool, the user proxy needs to execute it before the next agent runs. The state_transition function handles this conditional branching explicitly — it's essentially a finite state machine over agent states.

**Why two academic sources?**
arXiv and Semantic Scholar have different coverage. arXiv is stronger for preprints and ML papers; Semantic Scholar adds citation counts which the Critic agent uses for relevance scoring. The Critic deduplicates overlapping results.

**Why a caching layer?**
Repeated searches on the same topic hit the same APIs. The cache stores results for 24 hours, keyed by `md5(source:query)`. This avoids redundant API calls and makes the pipeline faster on follow-up queries.

**Why BYOK (Bring Your Own Key)?**
No server-side key storage, no security risk, no rate limit sharing across users. Each user authenticates with their own keys for their own session.

## Example Output

```
# Competitive Intelligence Brief: Anthropic

## TL;DR
- Released Claude 3.5 Sonnet with major coding improvements in June 2024
- Constitutional AI research showing measurable reduction in harmful outputs
- Developer community praising API reliability but flagging context window costs

## What's Happening Now
...

## Technical Landscape
...

## Strategic Implications
...

## Research Gaps & Opportunities
...
```

## v2 — LangGraph Upgrade

After building this AutoGen version, I rebuilt the pipeline using **LangGraph** to address two limitations:

1. **Text-based routing is fragile** — the `state_transition` function works, but it parses message content for tags like `[NEWS DATA]`. A malformed LLM response can break the flow. LangGraph uses typed Python `TypedDict` state and compile-time edge validation, making routing fully deterministic.
2. **True parallelism** — in this v1, news and paper research run sequentially. LangGraph runs them as genuine parallel graph branches with a synchronisation barrier before the analyst.

The v2 branch (`v2-langgraph`) replaces AutoGen GroupChat with a LangGraph `StateGraph`, the MD5 string cache with SHA-256 structured source dicts, and the Critic+Synthesizer pair with a single typed analyst node that outputs scored JSON directly.

Both versions run the same 3 APIs (Tavily, arXiv, Semantic Scholar) and the same Streamlit BYOK UI.

**→ [v2-langgraph branch](../../tree/v2-langgraph)**

## License

MIT
