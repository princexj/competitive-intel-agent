"""
main.py — CLI entry point for the Competitive Intelligence Agent.

Usage:
    python main.py

You will be prompted for:
    - Your Groq API key (get free at console.groq.com)
    - Your Tavily API key (get free at tavily.com — 1000 searches/month)
    - The company/technology/topic to research

The final report is saved as a .md file in the current directory.
"""

import os
from agents import build_agent_graph


def main():
    print("\n" + "="*60)
    print("   COMPETITIVE INTELLIGENCE AGENT")
    print("   Multi-Agent Research Pipeline")
    print("="*60 + "\n")

    # ── API Keys (BYOK pattern — never stored) ───────────────────
    groq_key = input("Enter your Groq API key: ").strip()
    if not groq_key:
        print("Error: Groq API key is required.")
        return

    tavily_key = input("Enter your Tavily API key (or press Enter to skip web search): ").strip()
    if not tavily_key:
        print("Warning: No Tavily key — web/news search will be disabled.\n")

    # ── Research Topic ───────────────────────────────────────────
    topic = input("\nWhat company or technology do you want to research?\n> ").strip()
    if not topic:
        print("Error: Topic cannot be empty.")
        return

    print(f"\n[Starting pipeline for: '{topic}']\n")
    print("Agent graph: Planner → News Researcher → Paper Researcher → Critic → Synthesizer → Reporter\n")
    print("-"*60 + "\n")

    # ── Build and run the agent graph ────────────────────────────
    user_proxy, manager, groupchat = build_agent_graph(groq_key, tavily_key)

    user_proxy.initiate_chat(
        manager,
        message=f"Research topic: {topic}"
    )

    # ── Save report ───────────────────────────────────────────────
    safe_name = "".join(c if c.isalnum() else "_" for c in topic)[:30]
    filename = f"{safe_name}_intel_report.md"

    with open(filename, "w", encoding="utf-8") as f:
        for msg in groupchat.messages:
            agent_name = msg.get("name", "")
            content = msg.get("content") or ""
            if agent_name == "Reporter" and content:
                clean = content.replace("TERMINATE", "").strip()
                f.write(clean)

    print(f"\n[✓] Report saved to: {filename}")


if __name__ == "__main__":
    main()
