"""
steps/step2_fetch.py
Step 2 — TOOL: Fetch papers from Semantic Scholar + score them.

Input  (from state):
    state["search_strings"]  — API query strings from Step 1
    state["field"]           — academic field (used for score weighting)

Output (written to state):
    state["papers"]   — list of paper dicts, sorted descending by suitability_score
                        Each paper dict:
                        {
                          paperId, title, year, citationCount,
                          abstract, authors (list of {authorId, name}),
                          externalIds (contains ArXiv id if available),
                          venue, openAccessPdf (url or None),
                          suitability_score
                        }
    state["authors"]  — dict: author_id → {name, h_index, paperCount, url}

Why this step exists:
    The LLM cannot produce real, up-to-date papers with accurate citation counts
    and author h-indices. This step retrieves real data from Semantic Scholar's
    free public API and enriches it with a transparent numeric suitability score.
    The score is what Step 3 uses as a starting ranking before applying semantic
    judgement.

Tool used:
    Semantic Scholar Academic Graph API (no API key required for basic use)
    Docs: https://api.semanticscholar.org/graph/v1
"""

import time
import os

from utils.scorer import compute_suitability_score


# ── Configuration ──────────────────────────────────────────────────────────────
MAX_RESULTS_PER_QUERY = 12   # keep normal mode fast while preserving breadth
MAX_TOTAL_PAPERS      = 40   # sufficient pool for stable top-10 selection
MAX_AUTHOR_LOOKUPS    = 8    # enough for key researcher section without slowdown


def _get_backend_module():
    backend = os.environ.get("S2_BACKEND", "semantic_scholar").strip().lower()
    if backend == "openalex":
        from utils import semantic_scholar_open_alex as backend_module
    else:
        from utils import semantic_scholar as backend_module
    return backend_module


# ── Main function ──────────────────────────────────────────────────────────────

def fetch_and_rank(state: dict) -> dict:
    """
    Tool Step 2: Fetch real papers from Semantic Scholar, score them,
    and profile key authors.

    Args:
        state: shared pipeline state dict

    Returns:
        state: updated with state["papers"] and state["authors"]
    """
    search_strings = state["search_strings"]
    field = state.get("field", "")
    backend_module = _get_backend_module()
    backend_name = os.environ.get("S2_BACKEND", "semantic_scholar").strip().lower()
    max_results_per_query = MAX_RESULTS_PER_QUERY
    max_total_papers = MAX_TOTAL_PAPERS
    max_author_lookups = MAX_AUTHOR_LOOKUPS

    # ── 1. Fetch papers for each search string ─────────────────────────────────
    seen_ids: set = set()
    all_papers: list = []

    print(f"  Using Step 2 backend: {backend_name}")

    for i, query in enumerate(search_strings):
        print(f"  [{i+1}/{len(search_strings)}] Querying: '{query}'")
        try:
            results = backend_module.search_papers(query, limit=max_results_per_query)
            # Abstract-focused fallback for niche domains when S2 is sparse.
            if backend_name == "semantic_scholar" and len(results) < 3:
                try:
                    from utils import semantic_scholar_open_alex as oa_backend
                    oa_results = oa_backend.search_papers(query, limit=max_results_per_query)
                    results.extend(oa_results)
                    print(f"    [FALLBACK] OpenAlex returned {len(oa_results)} results for '{query}'")
                except Exception as fallback_error:
                    print(f"    [WARN] OpenAlex fallback failed for '{query}': {fallback_error}")
        except Exception as e:
            print(f"    [WARN] Search failed for '{query}': {e}")
            results = []

        for paper in results:
            pid = paper.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(paper)

        # Respect API rate limits while keeping runtime bounded.
        if i < len(search_strings) - 1:
            time.sleep(0.2)

        if len(all_papers) >= max_total_papers:
            break

    print(f"  Fetched {len(all_papers)} unique papers before scoring.")

    if not all_papers:
        print("  [WARN] No papers found. State['papers'] will be empty.")
        state["papers"] = []
        state["authors"] = {}
        return state

    # ── 2. Score every paper ───────────────────────────────────────────────────
    keywords = state.get("keywords", [])
    for paper in all_papers:
        paper["suitability_score"] = compute_suitability_score(
            paper=paper,
            keywords=keywords,
            field=field,
        )

    # Sort descending by score
    all_papers.sort(key=lambda p: p["suitability_score"], reverse=True)
    state["papers"] = all_papers

    # ── 3. Profile top authors ─────────────────────────────────────────────────
    # Collect unique author IDs from the top-scored papers
    author_ids_ordered: list = []
    seen_author_ids: set = set()

    if max_author_lookups <= 0:
        state["authors"] = {}
        print(f"  Scored {len(all_papers)} papers. Profiled 0 authors.")
        return state

    paper_slice = all_papers[:15]
    for paper in paper_slice:
        for author in paper.get("authors", []):
            aid = author.get("authorId")
            if aid and aid not in seen_author_ids:
                seen_author_ids.add(aid)
                author_ids_ordered.append(aid)

    # Fetch author details (up to MAX_AUTHOR_LOOKUPS)
    authors: dict = {}
    for j, author_id in enumerate(author_ids_ordered[:max_author_lookups]):
        try:
            profile = backend_module.get_author_details(author_id)
            if profile:
                authors[author_id] = profile
        except Exception as e:
            print(f"    [WARN] Author lookup failed for {author_id}: {e}")

        # Rate limiting
        if j < min(len(author_ids_ordered), max_author_lookups) - 1:
            time.sleep(0.15)

    state["authors"] = authors
    print(f"  Scored {len(all_papers)} papers. Profiled {len(authors)} authors.")

    return state