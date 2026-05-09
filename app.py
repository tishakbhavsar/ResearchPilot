"""
Streamlit UI for ResearchPilot.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import threading
import time
from contextlib import redirect_stdout

import streamlit as st

from main import BANNER
from steps.step1_parse import parse_query
from steps.step2_fetch import fetch_and_rank
from steps.step3_relevance import assess_relevance
from steps.step4_deepread import deep_read
from steps.step5_synthesise import synthesise_critique
from steps.step6_report import write_lit_review
from utils.writer import save_report


def _stream_step(step_fn, state: dict, label: str, placeholder) -> tuple[dict, str]:
    """Run a pipeline step in a background thread and stream stdout to Streamlit.

    Returns the updated state and the captured logs.
    """
    buffer = io.StringIO()
    result = {}

    def target():
        nonlocal result
        with redirect_stdout(buffer):
            result = step_fn(state)

    thread = threading.Thread(target=target)
    thread.start()

    last = ""
    # Poll buffer and update placeholder while thread is running
    while thread.is_alive():
        time.sleep(0.15)
        text = buffer.getvalue()
        if text != last:
            placeholder.code(f"[{label}]\n" + text, language="text")
            last = text

    thread.join()
    full = buffer.getvalue()
    placeholder.code(f"[{label}]\n" + full, language="text")
    return result, full


def run_pipeline(query: str, provider: str, focus_note: str) -> tuple[dict, str, str]:
    """Run the full six-step pipeline and return the final state, path, and logs."""
    os.environ["LLM_PROVIDER"] = provider

    state: dict = {
        "raw_query": query,
        "topic": None,
        "keywords": [],
        "subtopics": [],
        "field": None,
        "time_range": None,
        "search_strings": [],
        "papers": [],
        "authors": {},
        "top_papers": [],
        "notable_researchers": [],
        "deep_reads": [],
        "synthesis": {
            "agreements": [],
            "contradictions": [],
            "gaps": [],
            "clusters": [],
        },
        "user_note": focus_note.strip(),
        "report_markdown": "",
    }

    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        state = parse_query(state)
        state = fetch_and_rank(state)
        state = assess_relevance(state)
        state = deep_read(state)
        state = synthesise_critique(state)
        state = write_lit_review(state)

    output_path = save_report(state["report_markdown"])
    logs = stdout_buffer.getvalue()
    return state, output_path, logs


def main() -> None:
    st.set_page_config(page_title="ResearchPilot", page_icon="📚", layout="wide")

    st.title("ResearchPilot")
    st.caption("Automated literature review agent with step-by-step progress and a single-query workflow.")
    st.code(BANNER, language="text")

    with st.sidebar:
        st.header("Run Settings")
        provider = st.selectbox("LLM provider", ["mock", "gemini", "grok"], index=0)
        focus_note = st.text_area(
            "Optional focus note",
            placeholder="e.g. focus on efficiency benchmarks only",
            height=100,
        )

    query = st.text_area(
        "Research topic",
        value="retrieval augmented generation for question answering",
        height=110,
        help="Describe the research topic in plain English.",
    )

    run_clicked = st.button("Run Agent", type="primary")

    if run_clicked:
        if not query.strip():
            st.error("Please enter a research topic before running the agent.")
            return

        progress = st.progress(0)
        status = st.empty()
        step_log = st.empty()
        status.info("Starting pipeline...")

        try:
            os.environ["LLM_PROVIDER"] = provider

            state: dict = {
                "raw_query": query,
                "topic": None,
                "keywords": [],
                "subtopics": [],
                "field": None,
                "time_range": None,
                "search_strings": [],
                "papers": [],
                "authors": {},
                "top_papers": [],
                "notable_researchers": [],
                "deep_reads": [],
                "synthesis": {
                    "agreements": [],
                    "contradictions": [],
                    "gaps": [],
                    "clusters": [],
                },
                "user_note": focus_note.strip(),
                "report_markdown": "",
            }

            log_chunks: list[str] = []

            status.info("Step 1/6: parsing query")
            step_log.info("Running Step 1...")
            state, logs = _stream_step(parse_query, state, "Step 1", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 1 finished.", language="text")
            progress.progress(15)

            status.info("Step 2/6: fetching and scoring papers")
            state, logs = _stream_step(fetch_and_rank, state, "Step 2", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 2 finished.", language="text")
            progress.progress(35)

            status.info("Step 3/6: selecting top papers")
            state, logs = _stream_step(assess_relevance, state, "Step 3", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 3 finished.", language="text")
            progress.progress(50)

            status.info("Step 4/6: deep reading papers")
            state, logs = _stream_step(deep_read, state, "Step 4", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 4 finished.", language="text")
            progress.progress(70)

            status.info("Step 5/6: synthesising themes")
            state, logs = _stream_step(synthesise_critique, state, "Step 5", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 5 finished.", language="text")
            progress.progress(85)

            status.info("Step 6/6: writing report")
            state, logs = _stream_step(write_lit_review, state, "Step 6", step_log)
            log_chunks.append(logs)
            step_log.code("".join(log_chunks) or "Step 6 finished.", language="text")

            output_path = save_report(state["report_markdown"])
            progress.progress(100)
            status.success("Pipeline complete")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Summary")
                st.write(f"**Topic:** {state.get('topic', query)}")
                st.write(f"**Field:** {state.get('field', 'Unknown')}")
                st.write(f"**Papers reviewed:** {len(state.get('top_papers', []))}")
                st.write(f"**Themes:** {', '.join(state.get('synthesis', {}).get('clusters', []))}")
                st.write(f"**Saved to:** {output_path}")

            with col2:
                st.subheader("Report Preview")
                st.markdown(state["report_markdown"])

            with st.expander("Run log"):
                st.code("".join(log_chunks) or "No logs captured.", language="text")

            st.download_button(
                "Download Markdown report",
                data=state["report_markdown"],
                file_name="lit_review.md",
                mime="text/markdown",
            )

        except Exception as exc:
            progress.progress(100)
            status.error("Pipeline failed")
            st.exception(exc)


if __name__ == "__main__":
    main()