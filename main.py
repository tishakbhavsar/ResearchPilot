"""
ResearchPilot — Automated Literature Review Agent
main.py: Orchestrates the full 6-step chain + confirmation gate.

Usage:
    python main.py
    python main.py "transformer architectures for vision tasks"
    python main.py --provider mock "retrieval augmented generation for question answering"

Chain:
    Step 1 (LLM)      → parse_query()        : extract structured query info
    Step 2 (Tool)     → fetch_and_rank()      : Semantic Scholar API + scoring
    Step 3 (LLM)      → assess_relevance()    : select top 5, profile researchers
    Step 4 (Tool+LLM) → deep_read()           : ArXiv PDF fetch + critical reading
    Step 5 (LLM)      → synthesise_critique() : agreements, contradictions, gaps
    [User confirmation gate]
    Step 6 (LLM)      → write_lit_review()    : final cited markdown report
"""

import argparse
import os
import sys
import time

from steps.step1_parse import parse_query
from steps.step2_fetch import fetch_and_rank
from steps.step3_relevance import assess_relevance
from steps.step4_deepread import deep_read
from steps.step5_synthesise import synthesise_critique
from steps.step6_report import write_lit_review
from utils.writer import save_report

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        ResearchPilot  •  Automated Literature Review Agent   ║
║   Step 1→2→3→4→5 → [You] → Step 6 → output/lit_review.md   ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_step_header(n: int, label: str, kind: str):
    icons = {"LLM": "🤖", "TOOL": "🔧", "TOOL+LLM": "🔧🤖"}
    icon = icons.get(kind, "▶")
    print(f"\n{'─' * 60}")
    print(f"  {icon}  Step {n} [{kind}] — {label}")
    print(f"{'─' * 60}")


def _fmt(v):
    """Format a value for display."""
    if isinstance(v, list):
        if len(v) == 0:
            return "0 items"
        if isinstance(v[0], dict):
            return f"{len(v)} items"
        # list of strings
        sample = ", ".join(str(x) for x in v[:3])
        return f"[{sample}{'…' if len(v) > 3 else ''}]"
    if isinstance(v, str) and len(v) > 110:
        return v[:110] + "…"
    if isinstance(v, dict):
        return f"{len(v)} entries"
    return str(v)


def print_step_result(key_values: dict):
    for k, v in key_values.items():
        print(f"  {k:<28}: {_fmt(v)}")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("query", nargs="*", help="Research topic in plain English")
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "grok"],
        help="Override the LLM provider without editing config.py",
    )
    parser.add_argument(
        "--use-openalex",
        action="store_true",
        help="Use OpenAlex instead of Semantic Scholar for Step 2 paper search",
    )
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        print(f"  Provider override: {args.provider}")

    if args.use_openalex:
        os.environ["S2_BACKEND"] = "openalex"
        print("  Step 2 backend override: OpenAlex")
    else:
        os.environ.setdefault("S2_BACKEND", "semantic_scholar")

    # ── Collect user query ─────────────────────────────────────────────────────
    if args.query:
        user_query = " ".join(args.query)
        print(f"  Query (from args): {user_query}\n")
    else:
        print("  Enter your research topic (specific or broad):")
        user_query = input("  > ").strip()

    if not user_query:
        print("[ERROR] No query provided. Exiting.")
        sys.exit(1)

    # ── Initialise shared state ────────────────────────────────────────────────
    # This dictionary is the backbone of the pipeline.
    # Every step reads from it and writes its outputs back into it.
    # No step receives arguments other than this dict — full traceability.
    state: dict = {
        # ── Raw input ──────────────────────────────────────────────────────────
        "raw_query": user_query,

        # ── Step 1 outputs ─────────────────────────────────────────────────────
        "topic": None,           # cleaned, canonical topic string
        "keywords": [],          # primary keyword list (5–8 terms)
        "subtopics": [],         # sub-areas the LLM identifies
        "field": None,           # e.g. "Computer Science / NLP"
        "time_range": None,      # e.g. "2019–2024"
        "search_strings": [],    # ready-to-use API query strings → fed to Step 2

        # ── Step 2 outputs ─────────────────────────────────────────────────────
        "papers": [],            # list of paper dicts (up to 50), each scored
        "authors": {},           # author_id → {name, h_index, paper_count, url}

        # ── Step 3 outputs ─────────────────────────────────────────────────────
        "top_papers": [],        # top 5 selected by LLM (subset of state["papers"])
        "notable_researchers": [],  # [{name, h_index, contribution, affiliation}]

        # ── Step 4 outputs ─────────────────────────────────────────────────────
        "deep_reads": [],        # one dict per top paper:
                                 #   {title, pdf_used, core_argument, methodology,
                                 #    key_findings, limitations, inner_citations}

        # ── Step 5 outputs ─────────────────────────────────────────────────────
        "synthesis": {
            "agreements":     [],   # themes all papers agree on
            "contradictions": [],   # paper-vs-paper disagreements
            "gaps":           [],   # unresolved questions / future work
            "clusters":       [],   # thematic groupings for Step 6 sections
        },

        # ── Confirmation gate ──────────────────────────────────────────────────
        "user_note": "",            # additional focus the user types in

        # ── Step 6 output ──────────────────────────────────────────────────────
        "report_markdown": "",      # full lit review as markdown
    }

    pipeline_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — LLM: Parse query into structured research parameters
    # Input:  state["raw_query"]
    # Output: state["topic"], ["keywords"], ["subtopics"], ["field"],
    #         ["time_range"], ["search_strings"]
    # Why separate: downstream steps need machine-readable fields, not raw text.
    #   Step 2 cannot call the API without ["search_strings"].
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(1, "Parsing research query", "LLM")
    print("  Extracting: keywords, subtopics, field, time_range, search_strings…")
    state = parse_query(state)
    print_step_result({
        "topic":          state["topic"],
        "field":          state["field"],
        "time_range":     state["time_range"],
        "keywords":       state["keywords"],
        "search_strings": state["search_strings"],
    })

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — TOOL: Fetch papers from Semantic Scholar + score them
    # Input:  state["search_strings"], state["field"]
    # Output: state["papers"] (sorted by suitability score),
    #         state["authors"] (h-index profiles)
    # Why separate: this is a real API call — the LLM cannot produce real papers.
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(2, "Fetching papers from Semantic Scholar", "TOOL")
    print(f"  Using {len(state['search_strings'])} search strings → target 30–50 papers…")
    state = fetch_and_rank(state)
    print_step_result({
        "papers fetched & scored": state["papers"],
        "author profiles":         state["authors"],
    })
    if state["papers"]:
        top3 = state["papers"][:3]
        print("  Top 3 by score:")
        for p in top3:
            print(f"    [{p.get('suitability_score', 0):.2f}] {p.get('title', '?')[:65]}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — TOOL: Select top 10, profile researchers
    # Input:  state["papers"], state["authors"], state["topic"],
    #         state["keywords"], state["subtopics"]
    # Output: state["top_papers"] (10 papers), state["notable_researchers"]
    # Why separate: the LLM applies semantic judgement that the numeric scorer
    #   cannot — it identifies seminal vs tangential papers, checks subtopic
    #   coverage, and filters specialist vs general works.
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(3, "Selecting top 10 papers + profiling researchers", "TOOL")
    print("  Using suitability score ranking + author profiles for Top 10…")
    state = assess_relevance(state)
    print_step_result({
        "top_papers": [p.get("title", "?")[:60] for p in state["top_papers"]],
        "notable_researchers": [r.get("name", "?") for r in state["notable_researchers"]],
    })

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — TOOL: Abstract-only reading for each selected paper
    # Input:  state["top_papers"]
    # Output: state["deep_reads"] — one abstract-derived analysis dict per paper
    # Why separate: keeps runtime predictable and robust for non-ArXiv fields.
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(4, "Reading abstracts for Top 10 papers", "TOOL")
    print("  Abstract-only processing (no PDF dependency)…")
    state = deep_read(state)
    print_step_result({
        "deep_reads completed": len(state["deep_reads"]),
    })
    for dr in state["deep_reads"]:
        src = "PDF" if dr.get("pdf_used") else "abstract-only"
        print(f"    • [{src}] {dr.get('title', '?')[:62]}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — LLM: Synthesise across all deep reads
    # Input:  state["deep_reads"], state["topic"], state["subtopics"]
    # Output: state["synthesis"] — agreements, contradictions, gaps, clusters
    # Why separate: synthesis requires seeing ALL papers simultaneously to make
    #   cross-paper comparisons. Step 4 cannot do this — it reads one at a time.
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(5, "Synthesising agreements, contradictions & gaps", "LLM")
    print("  Comparing deep-reads cross-paper…")
    state = synthesise_critique(state)
    s = state["synthesis"]
    print_step_result({
        "agreements":     s.get("agreements", []),
        "contradictions": s.get("contradictions", []),
        "gaps":           s.get("gaps", []),
        "clusters":       s.get("clusters", []),
    })

    # ══════════════════════════════════════════════════════════════════════════
    # USER CONFIRMATION GATE
    # Shows the user what the agent found before writing the full report.
    # User can say "yes" to proceed, or add a focus note to narrow the report.
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 60}")
    print("  ⏸   USER CONFIRMATION GATE")
    print(f"{'═' * 60}")
    print(f"\n  Topic      : {state['topic']}")
    print(f"  Papers     : {len(state['top_papers'])} selected for review")
    clusters = s.get("clusters", [])
    cluster_text = []
    for item in clusters:
        if isinstance(item, str):
            cluster_text.append(item)
        elif isinstance(item, dict):
            maybe = item.get("name") or item.get("theme") or item.get("cluster")
            if isinstance(maybe, str):
                cluster_text.append(maybe)
    print(f"  Themes     : {', '.join(cluster_text)}")
    print()
    agreements = s.get("agreements", [])
    contradictions = s.get("contradictions", [])
    gaps = s.get("gaps", [])
    if agreements:
        print(f"  📗 Agreements ({len(agreements)}):")
        for a in agreements[:2]:
            print(f"     - {a[:90]}")
    if contradictions:
        print(f"  🔴 Contradictions ({len(contradictions)}):")
        for c in contradictions[:2]:
            print(f"     - {c[:90]}")
    if gaps:
        print(f"  🔵 Gaps ({len(gaps)}):")
        for g in gaps[:2]:
            print(f"     - {g[:90]}")
    print()
    print("  Auto-proceeding to report generation.")
    state["user_note"] = ""

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — TOOL: Write final research brief
    # Input:  ALL of state — top_papers, deep_reads, synthesis, notable_researchers,
    #         user_note, topic, field, time_range
    # Output: state["report_markdown"] — full .md document
    # Why separate: this step requires the complete accumulated state from every
    #   prior step. It cannot run until synthesis (Step 5) is complete.
    # ══════════════════════════════════════════════════════════════════════════
    print_step_header(6, "Writing final research brief", "TOOL")
    print("  Generating ranked summary, themes, navigation, and critical gaps…")
    state = write_lit_review(state)

    # ── Save to disk ───────────────────────────────────────────────────────────
    output_path = save_report(state["report_markdown"])

    elapsed = time.time() - pipeline_start
    print(f"\n{'═' * 60}")
    print(f"  ✅  Complete in {elapsed:.1f}s")
    print(f"  📄  Saved → {output_path}")
    print(f"{'═' * 60}\n")

    # Preview first 800 chars
    preview = state["report_markdown"][:800]
    print("── Report Preview ────────────────────────────────────────")
    print(preview)
    if len(state["report_markdown"]) > 800:
        print(f"\n  … ({len(state['report_markdown']) - 800} more characters in file)")
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()