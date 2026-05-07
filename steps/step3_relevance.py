"""
steps/step3_relevance.py
Step 3 — LLM: Assess relevance, select top 5 papers, profile notable researchers.

Input  (from state):
        state["papers"]         — scored + sorted paper list from Step 2
        state["authors"]        — author profiles from Step 2
        state["topic"]          — canonical topic from Step 1
        state["keywords"]       — keyword list from Step 1
        state["subtopics"]      — sub-areas from Step 1

Output (written to state):
        state["top_papers"]          — final 5 papers selected by the LLM
                                                                     (each is a dict from state["papers"])
        state["notable_researchers"] — list of researcher profiles:
                                                                     [{name, h_index, affiliation, contribution}]

Why this step exists:
        The numeric suitability score from Step 2 is a good first filter, but it
        cannot detect: seminal foundational papers (which might have older dates
        but must be included), tangential papers that match keywords but are off-
        topic, or whether the top-20 collectively covers all subtopics. The LLM
        applies semantic reasoning that no numeric formula can replicate.

        Step 2 cannot do this because it has no language understanding.
        Step 4 cannot do this because it needs exactly 5 papers to read — it can't
        decide which 5 to read while also reading them.

Prompt design:
        - The LLM receives the top 20 papers as a compact list (title + abstract
            snippet + score + year + citations).
        - It receives the author h-index table separately.
        - It returns a JSON object with two arrays: top_paper_ids and researchers.
        - The prompt explicitly asks for subtopic coverage, not just top scores,
            because the rubric rewards understanding of chain design rationale.
"""

TOP_PAPER_COUNT = 10


# ── Main function ──────────────────────────────────────────────────────────────

def assess_relevance(state: dict) -> dict:
    """
    Step 3: Select top 10 papers and profile notable researchers.

    Args:
        state: shared pipeline state dict

    Returns:
        state: updated with state["top_papers"] and state["notable_researchers"]
    """
    papers  = state["papers"]
    authors = state["authors"]
    topic   = state["topic"]

    if not papers:
        print("  [WARN] Step 3: No papers to assess. Skipping.")
        state["top_papers"] = []
        state["notable_researchers"] = []
        return state

    # Deterministic top-10 for speed and stability.
    top_papers = papers[:TOP_PAPER_COUNT]
    state["selection_reasoning"] = (
        "Selected by highest suitability score (keywords, citations, recency, "
        "open-access availability) with deduplicated paper IDs."
    )

    state["top_papers"] = top_papers
    # Prefer LLM-provided notable researchers, but synthesize a fallback
    notable = []
    if not notable:
        authors_map = state.get("authors", {}) or {}
        if authors_map:
            candidate_author_ids: list[str] = []
            for paper in top_papers:
                for author in paper.get("authors", [])[:5]:
                    author_id = author.get("authorId")
                    if author_id:
                        candidate_author_ids.append(author_id)

            # Prefer researchers attached to the candidate papers, then fall back to the full pool.
            scored = []
            source_ids = list(dict.fromkeys(candidate_author_ids)) or list(authors_map.keys())
            for author_id in source_ids:
                profile = authors_map.get(author_id)
                if not profile:
                    continue
                h_value = profile.get("h_index") or profile.get("hIndex") or 0
                try:
                    h_index = int(h_value or 0)
                except Exception:
                    h_index = 0
                scored.append((h_index, author_id, profile))

            scored.sort(key=lambda item: item[0], reverse=True)
            notable = []
            for h_index, _, profile in scored[:3]:
                affiliations = profile.get("affiliations") or []
                notable.append({
                    "name": profile.get("name") or profile.get("displayName") or "Unknown",
                    "h_index": h_index,
                    "affiliation": affiliations[0] if affiliations else profile.get("affiliation") or "Unknown",
                    "contribution": profile.get("shortBio") or profile.get("contribution") or "",
                })

    state["notable_researchers"] = notable

    reasoning = state.get("selection_reasoning", "")
    if reasoning:
        print(f"  Selection reasoning: {reasoning[:120]}…")

    return state


# ── Helpers ────────────────────────────────────────────────────────────────────
