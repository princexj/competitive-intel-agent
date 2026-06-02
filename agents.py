import autogen
from config import get_llm_config
from tools import fetch_arxiv_papers, fetch_semantic_scholar, fetch_web_news


def build_agent_graph(groq_api_key: str, tavily_api_key: str):
    """
    Builds and returns the full multi-agent pipeline.
    
    Agent Graph (Directed):
    
    User → Planner → [News Researcher, Paper Researcher] → Critic → Synthesis → Report → User
    
    The state_transition function is the edge logic of this graph.
    Each agent is a node; transitions are conditional on message content.
    """

    llm_config = get_llm_config(groq_api_key)

    # ── NODE 1: Planner ─────────────────────────────────────────────────────
    # Breaks user input into 3 research angles.
    # Decides what to search and how.

    planner = autogen.AssistantAgent(
        name="Planner",
        system_message="""You are the lead research strategist for a competitive intelligence system.

Your job: Take the user's input (a company name, technology, or topic) and break it into exactly 3 targeted search queries:
1. A NEWS query — recent developments, announcements, funding, market moves (e.g. "Anthropic 2025 news funding")
2. A TECHNICAL query — for academic/research papers (e.g. "constitutional AI alignment methods")  
3. A COMMUNITY query — what engineers and builders are discussing (e.g. "Anthropic Claude API developer experience")

Output format (strictly follow this):
NEWS_QUERY: <your query>
TECH_QUERY: <your query>
COMMUNITY_QUERY: <your query>

Do not add explanations. Just the three queries.""",
        llm_config=llm_config,
    )

    # ── NODE 2: News Researcher ──────────────────────────────────────────────
    # Uses web search (Tavily) for market/news angle.

    news_researcher = autogen.AssistantAgent(
        name="News_Researcher",
        system_message="""You are a market intelligence analyst.

You will receive a NEWS_QUERY from the Planner.
Use the fetch_web_news tool with that query.
Return the raw results with the prefix: [NEWS DATA]

Do not summarize. Return raw tool output only.""",
        llm_config=llm_config,
    )

    # ── NODE 3: Paper Researcher ─────────────────────────────────────────────
    # Uses arXiv + Semantic Scholar for technical/research angle.

    paper_researcher = autogen.AssistantAgent(
        name="Paper_Researcher",
        system_message="""You are an academic research analyst.

You will receive a TECH_QUERY from the Planner.
Use BOTH fetch_arxiv_papers AND fetch_semantic_scholar tools with that query.
Return all raw results combined with the prefix: [RESEARCH DATA]

Do not summarize. Return raw tool output only.""",
        llm_config=llm_config,
    )

    # ── NODE 4: Critic ───────────────────────────────────────────────────────
    # Scores and filters results. The "evaluation layer".
    # This is the resume flex — guardrails against low-quality data.

    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="""You are a rigorous quality reviewer for a competitive intelligence system.

You will receive raw data from both researchers (news + research papers).

Your job:
1. Score each item 1-10 on relevance to the original topic
2. DROP any item scoring below 6
3. For research papers, prefer items with higher citation counts
4. Remove duplicate papers that appear in both arXiv and Semantic Scholar (keep the one with more info)
5. Output the filtered, cleaned data under two sections:

[FILTERED NEWS]
<approved news items>

[FILTERED RESEARCH]  
<approved papers with their scores>

Be strict. Quality over quantity.""",
        llm_config=llm_config,
    )

    # ── NODE 5: Synthesis Agent ──────────────────────────────────────────────
    # Combines all 3 angles into structured insights.

    synthesizer = autogen.AssistantAgent(
        name="Synthesizer",
        system_message="""You are a senior analyst at a competitive intelligence firm.

You will receive filtered news and research data from the Critic.

Extract and connect insights across both sources:
- What's the company/technology doing RIGHT NOW (news)?
- What does the research say about where this is headed technically?
- What are practitioners saying vs what researchers are saying?
- Where are the gaps or opportunities?

Output structured bullet points under these headings:
## Current Market Position
## Technical Direction  
## Practitioner Sentiment
## Emerging Opportunities

Be specific and cite sources. No vague statements.""",
        llm_config=llm_config,
    )

    # ── NODE 6: Report Agent ─────────────────────────────────────────────────
    # Final output — clean, formatted, downloadable report.

    reporter = autogen.AssistantAgent(
        name="Reporter",
        system_message="""You are a technical writer producing a competitive intelligence brief.

Take the structured insights from the Synthesizer and compile a final report in this exact format:

---
# Competitive Intelligence Brief: [TOPIC]
**Generated:** [today's date]

## TL;DR
- [3 bullet points, max 15 words each]

## What's Happening Now
[News-based insights, 2-3 paragraphs]

## Technical Landscape
[Research-based insights, 2-3 paragraphs]  

## Strategic Implications
[So what? What does this mean for someone watching this space?]

## Research Gaps & Opportunities
[2-3 areas not yet well-covered]

## Sources
[List all referenced URLs]
---

End your message with the word TERMINATE.""",
        llm_config=llm_config,
    )

    # ── USER PROXY ───────────────────────────────────────────────────────────
    # Bridge between system and tool execution.
    # human_input_mode="NEVER" — fully autonomous pipeline.

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or "").upper()
    )

    # ── TOOL REGISTRATION ────────────────────────────────────────────────────
    # Bind tools to specific agents — only they can call them.

    autogen.agentchat.register_function(
        fetch_arxiv_papers,
        caller=paper_researcher,
        executor=user_proxy,
        name="fetch_arxiv_papers",
        description="Search arXiv for academic papers on a topic."
    )

    autogen.agentchat.register_function(
        fetch_semantic_scholar,
        caller=paper_researcher,
        executor=user_proxy,
        name="fetch_semantic_scholar",
        description="Search Semantic Scholar for academic papers with citation counts."
    )

    # Wrap fetch_web_news to inject the tavily key (closure pattern)
    def _fetch_web_news(query: str) -> str:
        return fetch_web_news(query, tavily_api_key)

    autogen.agentchat.register_function(
        _fetch_web_news,
        caller=news_researcher,
        executor=user_proxy,
        name="fetch_web_news",
        description="Search the live web for recent news and market intelligence."
    )

    # ── STATE TRANSITION FUNCTION ────────────────────────────────────────────
    # This is the directed graph edge logic.
    # Think of it as: given current node (last_speaker) + edge condition (message),
    # return the next node to activate.
    #
    # Graph:
    # Admin → Planner → News_Researcher → (tool call) → Admin → Paper_Researcher
    # → (tool calls) → Admin → Critic → Synthesizer → Reporter → Admin (TERMINATE)

    def state_transition(last_speaker, groupchat):
        messages = groupchat.messages
        if not messages:
            return planner

        last_message = messages[-1]
        content = last_message.get("content") or ""

        # If any agent requests a tool call, hand to user_proxy to execute
        if last_message.get("tool_calls") or last_message.get("function_call"):
            return user_proxy

        # After tool execution, return to the agent that called it
        if last_speaker is user_proxy:
            # Check who called the tool by looking back through messages
            for msg in reversed(messages[:-1]):
                if msg.get("tool_calls") or msg.get("function_call"):
                    caller_name = msg.get("name", "")
                    if caller_name == "News_Researcher":
                        return paper_researcher  # news done → now do papers
                    if caller_name == "Paper_Researcher":
                        return critic  # both researchers done → critic filters
                    break
            return planner  # fallback

        # Standard pipeline flow
        if last_speaker is planner:
            return news_researcher

        if last_speaker is news_researcher:
            # If news researcher just returned data (not a tool call), go to paper researcher
            if "[NEWS DATA]" in content:
                return paper_researcher
            return news_researcher  # still working

        if last_speaker is paper_researcher:
            if "[RESEARCH DATA]" in content:
                return critic
            return paper_researcher  # still working

        if last_speaker is critic:
            return synthesizer

        if last_speaker is synthesizer:
            return reporter

        if last_speaker is reporter:
            return user_proxy  # TERMINATE check happens here

        return planner  # fallback

    # ── GROUP CHAT ───────────────────────────────────────────────────────────

    groupchat = autogen.GroupChat(
        agents=[user_proxy, planner, news_researcher, paper_researcher, critic, synthesizer, reporter],
        messages=[],
        max_round=25,
        speaker_selection_method=state_transition
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or "").upper()
    )

    return user_proxy, manager, groupchat
