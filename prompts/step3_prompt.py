"""
prompts/step3_prompt.py
System and user prompt for Step 3 — assess_relevance (LLM).

PURPOSE OF THIS STEP:
    Step 2 returns up to 50 papers ranked by a numeric formula (citation count,
    keyword match, recency, open-access availability). That formula is fast and
    transparent but cannot detect:
      - Seminal papers that are older but foundational to the field
      - Papers that match keywords but are about a tangential topic
      - Whether the top-5 collectively covers all subtopics (not just the top one)
      - Author credibility relative to the specific subfield

    The LLM applies semantic reasoning on top of the numeric ranking to pick the
    best 5 papers for deep reading in Step 4.

    Step 3 cannot be merged with Step 4 because Step 4 reads each paper
    individually with full attention — it can't also decide which papers to read
    while reading them.

PROMPT DESIGN RATIONALE:
    1. Selection criteria are ordered by priority — the LLM must resolve conflicts
       (e.g. a highly-cited paper that doesn't cover a key subtopic) explicitly.
    2. The prompt asks for top_paper_ids (not titles) because the parser uses
       IDs to look up the full paper dict in state["papers"]. Titles can have
       subtle differences between what S2 returns and what the LLM echoes back.
    3. "Exactly 5 paper IDs" is stated explicitly — without this, the LLM
       sometimes returns 4 or 6.
    4. notable_researchers are extracted here (not Step 4) because Step 3
       already has the author h-index data. Re-fetching in Step 4 would be
       redundant.

PROMPT ITERATION (v1 → v2):
    v1 prompt:
        Asked for top papers by title, not by ID.

    v1 problem:
        The LLM would return a slightly different title than what S2 stored
        (e.g. dropped "A" at the start, different capitalisation). The lookup
        in step3_relevance.py failed silently → top_papers was empty →
        Step 4 had nothing to read.

    v2 change (current):
        Changed to top_paper_ids (exact IDs from the candidate list).
        Added "top_paper_ids must contain exactly 5 paper IDs from the
        candidate list" with enforcement in the fallback code.
"""

SYSTEM_PROMPT = """You are a researcher selecting 5 key papers from a candidate list.

Rules:
- Return ONLY valid JSON. No prose.
- top_paper_ids: list of exactly 5 IDs copied from candidates (no modification).
- selection_reasoning: 1–2 sentences only.
- notable_researchers: optional short list of 3 names with h_index, affiliation, contribution.

Schema:
{"top_paper_ids": ["id1", "id2", "id3", "id4", "id5"], "selection_reasoning": "reason", "notable_researchers": []}"""

USER_PROMPT_TEMPLATE = """Topic: {topic}
Subtopics: {subtopics}

--- PAPERS (pick 5 by ID) ---
{papers_block}

--- AUTHORS (optional context) ---
{authors_block}

Return JSON: {{"top_paper_ids": [...5 IDs...], "selection_reasoning": "...brief reason...", "notable_researchers": [...]}}"""