"""
app.py — Streamlit UI for the Competitive Intelligence Agent.

Run with:
    streamlit run app.py

BYOK (Bring Your Own Key) pattern:
- User enters their Groq + Tavily API keys in the sidebar
- Keys stored in st.session_state only — never saved to disk
- Session ends → keys gone
"""

import streamlit as st
import time
import os
from agents import build_agent_graph

# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Intel Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

    :root {
        --bg: #0a0a0f;
        --surface: #12121a;
        --border: #1e1e2e;
        --accent: #7c3aed;
        --accent-light: #a78bfa;
        --green: #10b981;
        --text: #e2e8f0;
        --muted: #64748b;
    }

    .stApp {
        background: var(--bg);
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }

    .stSidebar {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    /* Header */
    .intel-header {
        text-align: center;
        padding: 2rem 0 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }

    .intel-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: var(--accent-light);
        letter-spacing: -0.02em;
        margin: 0;
    }

    .intel-header p {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Agent pipeline visual */
    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
        padding: 1rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }

    .agent-node {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        border: 1px solid var(--border);
        color: var(--muted);
        background: var(--bg);
    }

    .agent-node.active {
        border-color: var(--accent);
        color: var(--accent-light);
        background: rgba(124, 58, 237, 0.1);
    }

    .arrow {
        color: var(--muted);
        font-size: 0.7rem;
    }

    /* Status log */
    .log-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        max-height: 200px;
        overflow-y: auto;
        margin-bottom: 1.5rem;
    }

    .log-entry {
        padding: 0.2rem 0;
        color: var(--muted);
        border-bottom: 1px solid rgba(30, 30, 46, 0.5);
    }

    .log-entry.success { color: var(--green); }
    .log-entry.active { color: var(--accent-light); }

    /* Report output */
    .report-container {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 2rem;
    }

    /* Sidebar styling */
    .key-section {
        background: var(--bg);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    .key-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--muted);
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Streamlit overrides */
    .stTextInput > div > div > input {
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
        border-radius: 6px !important;
    }

    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        border-radius: 6px !important;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #6d28d9 !important;
        transform: translateY(-1px);
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--accent-light);
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Download button */
    .stDownloadButton > button {
        background: transparent !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent-light) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        width: auto !important;
    }

    div[data-testid="stSidebarContent"] {
        padding: 1.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────────────

if "report" not in st.session_state:
    st.session_state.report = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "running" not in st.session_state:
    st.session_state.running = False

# ── Sidebar — API Keys ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔑 API Keys")
    st.caption("Keys are stored in session memory only — never saved to disk.")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com"
    )

    tavily_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        help="Free at tavily.com — 1000 searches/month"
    )

    st.divider()
    st.markdown("### 📡 Agent Pipeline")
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; line-height: 1.8;">
    1. Planner<br>
    ↓ breaks topic into 3 angles<br>
    2. News Researcher<br>
    ↓ Tavily web search<br>
    3. Paper Researcher<br>
    ↓ arXiv + Semantic Scholar<br>
    4. Critic<br>
    ↓ scores & filters (≥6/10)<br>
    5. Synthesizer<br>
    ↓ cross-source insights<br>
    6. Reporter<br>
    ↓ final brief
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("Built with AutoGen · Groq · arXiv · Semantic Scholar · Tavily")

# ── Main UI ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="intel-header">
    <h1>// INTEL AGENT</h1>
    <p>competitive intelligence · multi-agent pipeline · real-time research</p>
</div>
""", unsafe_allow_html=True)

# Topic input
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input(
        "Research Target",
        placeholder="e.g. Anthropic, RAG vs Fine-tuning, Mistral AI, LangChain...",
        label_visibility="collapsed"
    )
with col2:
    run_btn = st.button("▶ RUN", disabled=st.session_state.running)

# Pipeline visualization
st.markdown("""
<div class="pipeline">
    <span class="agent-node">Planner</span>
    <span class="arrow">→</span>
    <span class="agent-node">News Researcher</span>
    <span class="arrow">→</span>
    <span class="agent-node">Paper Researcher</span>
    <span class="arrow">→</span>
    <span class="agent-node">Critic</span>
    <span class="arrow">→</span>
    <span class="agent-node">Synthesizer</span>
    <span class="arrow">→</span>
    <span class="agent-node">Reporter</span>
</div>
""", unsafe_allow_html=True)

# ── Run Pipeline ─────────────────────────────────────────────────────────────

if run_btn and topic:
    if not groq_key:
        st.error("Groq API key is required. Add it in the sidebar.")
    else:
        st.session_state.running = True
        st.session_state.report = None
        st.session_state.logs = []

        log_placeholder = st.empty()
        status_placeholder = st.empty()

        def add_log(msg, level="active"):
            st.session_state.logs.append((msg, level))

        with st.spinner(f"Running intelligence pipeline for: **{topic}**"):
            try:
                add_log(f"[00:00] Pipeline initialized for: {topic}")
                add_log(f"[00:01] Building agent graph...")

                user_proxy, manager, groupchat = build_agent_graph(groq_key, tavily_key or "")

                add_log(f"[00:02] Planner generating research angles...")
                add_log(f"[00:03] News Researcher querying web...")
                add_log(f"[00:04] Paper Researcher querying arXiv + Semantic Scholar...")
                add_log(f"[00:05] Critic filtering and scoring results...")
                add_log(f"[00:06] Synthesizer cross-referencing sources...")
                add_log(f"[00:07] Reporter compiling final brief...")

                user_proxy.initiate_chat(
                    manager,
                    message=f"Research topic: {topic}"
                )

                # Extract report from groupchat messages
                report_content = ""
                for msg in groupchat.messages:
                    if msg.get("name") == "Reporter" and msg.get("content"):
                        report_content = msg["content"].replace("TERMINATE", "").strip()
                        break

                if report_content:
                    st.session_state.report = report_content
                    add_log(f"[✓] Pipeline complete. Report generated.", "success")
                else:
                    add_log(f"[!] Pipeline finished but no report was generated.", "active")

            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                add_log(f"[✗] Error: {str(e)}", "active")

        st.session_state.running = False
        st.rerun()

# ── Display Logs ─────────────────────────────────────────────────────────────

if st.session_state.logs:
    log_html = '<div class="log-container">'
    for entry, level in st.session_state.logs:
        log_html += f'<div class="log-entry {level}">{entry}</div>'
    log_html += '</div>'
    st.markdown(log_html, unsafe_allow_html=True)

# ── Display Report ───────────────────────────────────────────────────────────

if st.session_state.report:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("### Intelligence Brief")
    with col2:
        st.download_button(
            "⬇ Download .md",
            data=st.session_state.report,
            file_name=f"intel_{topic.replace(' ', '_')[:20]}.md",
            mime="text/markdown"
        )

    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)

elif not st.session_state.running:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
        Enter a company, technology, or topic above and hit RUN.<br><br>
        <span style="font-size: 0.75rem; opacity: 0.6;">
        Examples: "Anthropic" · "RAG pipelines" · "Mistral vs Llama" · "LangChain" · "OpenAI Sora"
        </span>
    </div>
    """, unsafe_allow_html=True)
