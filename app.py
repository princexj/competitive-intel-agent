"""Streamlit product UI for the competitive intelligence workflow."""

from __future__ import annotations

import json
import os
import re
from html import escape

import streamlit as st

from graph import compare_targets, result_from_state, stream_target
from storage import get_run, list_runs, save_run

st.set_page_config(
    page_title="SignalForge Intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #e8edf5;
        --muted: #8490a6;
        --panel: #111827;
        --line: #243047;
        --accent: #5eead4;
        --hot: #fb7185;
    }
    .stApp { background: #080d17; color: var(--ink); }
    [data-testid="stSidebar"] { background: #0d1422; border-right: 1px solid var(--line); }
    .hero {
        padding: 2rem 0 1.4rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: var(--accent);
        font: 600 .72rem ui-monospace, monospace;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        color: var(--ink);
        font-size: clamp(2.2rem, 5vw, 4.5rem);
        letter-spacing: -.055em;
        line-height: .95;
        margin: .5rem 0 .8rem;
    }
    .hero p { color: var(--muted); max-width: 700px; font-size: 1.05rem; }
    .metric-card {
        background: linear-gradient(145deg, #121b2c, #0d1422);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        min-height: 112px;
    }
    .metric-label {
        color: var(--muted);
        font: 600 .68rem ui-monospace, monospace;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .metric-value { color: var(--accent); font-size: 1.7rem; font-weight: 700; }
    .source-row {
        border-left: 2px solid var(--line);
        padding: .25rem 0 .25rem .8rem;
        margin: .55rem 0;
        color: var(--muted);
    }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; }
    div[data-testid="stStatusWidget"] { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
STAGE_LABELS = {
    "planner": "Research plan created",
    "news_researcher": "Market and news signals collected",
    "paper_researcher": "Academic evidence collected",
    "community_researcher": "Practitioner signals collected",
    "analyst": "Evidence scored and synthesized",
    "reporter": "Intelligence brief written",
}


def initialize_state() -> None:
    defaults = {
        "results": [],
        "comparison": "",
        "run_id": "",
        "loaded_targets": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def parse_targets(raw: str) -> list[str]:
    targets = [part.strip() for part in re.split(r"[,\n]", raw) if part.strip()]
    return list(dict.fromkeys(targets))


def markdown_export(results: list[dict], comparison: str) -> str:
    sections = []
    if comparison:
        sections.append("# Competitive Landscape\n\n" + comparison)
    for result in results:
        sections.append(result["report"])
    return "\n\n---\n\n".join(sections)


def load_saved_run(run_id: str) -> None:
    saved = get_run(run_id)
    if saved:
        st.session_state.results = saved["results"]
        st.session_state.comparison = saved["comparison"]
        st.session_state.run_id = saved["id"]
        st.session_state.loaded_targets = ", ".join(saved["targets"])


def score_cards(analysis: dict) -> None:
    scores = analysis.get("scores", {})
    labels = [
        ("Momentum", scores.get("momentum", "N/A")),
        ("Technical strength", scores.get("technical_strength", "N/A")),
        ("Community interest", scores.get("community_interest", "N/A")),
        ("Opportunity", scores.get("opportunity", "N/A")),
    ]
    columns = st.columns(4)
    for column, (label, value) in zip(columns, labels):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{escape(str(value))}</div></div>',
                unsafe_allow_html=True,
            )


initialize_state()

with st.sidebar:
    st.markdown("## SignalForge")
    st.caption("Evidence-backed competitive intelligence")
    groq_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Used only for this session.",
    )
    tavily_key = st.text_input(
        "Tavily API key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Required for live market and community research.",
    )
    model = st.selectbox("Model", [DEFAULT_MODEL], index=0)

    st.divider()
    st.markdown("### Workflow")
    st.caption(
        "Plan -> parallel market, research, and community collection -> "
        "evidence analysis -> cited report"
    )

    history = list_runs()
    if history:
        st.divider()
        st.markdown("### Recent runs")
        for item in history[:8]:
            label = " vs ".join(item["targets"])[:34]
            if st.button(label, key=f"history_{item['id']}", use_container_width=True):
                load_saved_run(item["id"])
                st.rerun()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Competitive Intelligence Workbench</div>
      <h1>Know the field.<br>Choose the move.</h1>
      <p>Research one company or compare up to four. SignalForge triangulates live
      market activity, technical research, and practitioner sentiment into a cited
      strategic brief.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

targets_raw = st.text_area(
    "Research targets",
    value=st.session_state.loaded_targets,
    placeholder="Anthropic, OpenAI, Mistral AI",
    help="Separate targets with commas or new lines. Maximum four.",
    height=92,
)
depth_col, action_col = st.columns([3, 1])
with depth_col:
    st.caption("One target creates a deep brief. Two or more also create a comparison.")
with action_col:
    run_clicked = st.button(
        "Run intelligence scan", type="primary", use_container_width=True
    )

if run_clicked:
    targets = parse_targets(targets_raw)
    if not groq_key:
        st.error("Add a Groq API key in the sidebar.")
    elif not tavily_key:
        st.error("Add a Tavily API key for live market and community evidence.")
    elif not targets:
        st.error("Enter at least one research target.")
    elif len(targets) > 4:
        st.error("Choose at most four targets per comparison.")
    else:
        st.session_state.results = []
        st.session_state.comparison = ""
        st.session_state.run_id = ""
        st.session_state.loaded_targets = ", ".join(targets)

        overall = st.progress(0, text="Starting intelligence workflow...")
        run_failed = False
        for target_index, target in enumerate(targets):
            aggregate = {"topic": target, "errors": []}
            with st.status(f"Researching {target}", expanded=True) as status:
                try:
                    for node, update in stream_target(
                        target, groq_key, tavily_key, model=model
                    ):
                        for key, value in update.items():
                            if key == "errors":
                                aggregate["errors"].extend(value)
                            else:
                                aggregate[key] = value
                        st.write(f"Done: {STAGE_LABELS.get(node, node)}")
                        progress = (
                            target_index + (list(STAGE_LABELS).index(node) + 1) / 6
                        ) / len(targets)
                        overall.progress(
                            min(progress, 1.0),
                            text=f"{target}: {STAGE_LABELS.get(node, node)}",
                        )
                    status.update(label=f"{target} complete", state="complete")
                except Exception as exc:
                    run_failed = True
                    status.update(label=f"{target} failed", state="error")
                    st.error(f"{target}: {exc}")
                    break
            st.session_state.results.append(result_from_state(target, aggregate))
            if aggregate["errors"]:
                st.warning(" ".join(aggregate["errors"]))

        if not run_failed and len(st.session_state.results) > 1:
            overall.progress(0.95, text="Building cross-target comparison...")
            try:
                st.session_state.comparison = compare_targets(
                    st.session_state.results, groq_key, model=model
                )
            except Exception as exc:
                run_failed = True
                st.error(f"Comparison failed: {exc}")
        if not run_failed:
            st.session_state.run_id = save_run(
                st.session_state.results, st.session_state.comparison
            )
            overall.progress(1.0, text="Intelligence scan complete")
            st.rerun()

results = st.session_state.results
comparison = st.session_state.comparison

if results:
    header_col, md_col, json_col = st.columns([4, 1, 1])
    with header_col:
        st.markdown(
            f"### Intelligence output "
            f"`{st.session_state.run_id or 'unsaved'}`"
        )
    export_markdown = markdown_export(results, comparison)
    with md_col:
        st.download_button(
            "Markdown",
            export_markdown,
            file_name="competitive_intelligence.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with json_col:
        st.download_button(
            "JSON",
            json.dumps(
                {"comparison": comparison, "results": results}, indent=2
            ),
            file_name="competitive_intelligence.json",
            mime="application/json",
            use_container_width=True,
        )

    tabs = st.tabs(
        (["Comparison"] if comparison else [])
        + [result["topic"] for result in results]
    )
    tab_index = 0
    if comparison:
        with tabs[0]:
            st.markdown(comparison)
        tab_index = 1

    for result, tab in zip(results, tabs[tab_index:]):
        with tab:
            score_cards(result["analysis"])
            st.markdown(result["report"])
            with st.expander(f"Evidence ledger ({len(result['sources'])} sources)"):
                for source in result["sources"]:
                    st.markdown(
                        f"**[{source['category'].upper()}] "
                        f"[{source['title']}]({source['url']})**  \n"
                        f"{source['source']} | {source['published'] or 'Date unavailable'}"
                    )
                    st.caption(source["snippet"])
else:
    st.info(
        "Start with a company, product, or technology. For example: "
        "`Anthropic, OpenAI, Mistral AI`."
    )
