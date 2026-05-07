"""
prompts/step6_prompt.py
System and user prompt for Step 6 — gaps_and_navigation (LLM).

PURPOSE OF THIS STEP:
    Step 6 makes ONE small LLM call to produce two things that require
    cross-paper reasoning and cannot be computed deterministically:
      1. Research gaps — open questions emerging from reading all papers together
      2. Navigation guide — which papers to start with for each subtopic

    Everything else in Step 6 (ranked table, summaries, themes, agreements,
    references) is assembled deterministically from state without an LLM.

WHY ONLY THESE TWO OUTPUTS:
    Gaps require synthesising what is absent across all papers — the LLM must
    reason about what none of the papers address. No formula can produce this.

    The navigation guide requires understanding each paper's method and scope
    well enough to recommend which ones address a given subtopic. This needs
    language understanding, not arithmetic.

    Everything else in the brief is already in state from Steps 1-5:
    - Ranked table: scores from Step 2, computed by scorer.py
    - Summaries: core_argument from Step 4
    - Themes: clusters from Step 5
    - Agreements/contradictions: from Step 5
    - References: metadata from Step 2

DESIGN RATIONALE:
    1. max_tokens=512 — the output is small JSON. No truncation risk.
    2. paper_numbers must be integers (1-indexed) — the assembly code in
       step6_report.py uses these to render "Paper 1, Paper 3" references.
    3. gaps must be formulated as open research questions, not restatements
       of paper limitations. The prompt enforces this with the example.
    4. navigation subtopics come from the clusters produced in Step 5 —
       the prompt injects them so the LLM uses consistent terminology.

PROMPT ITERATION (v1 → v2):
    v1: Asked LLM to write a full academic literature review (1200-2500 words).
        System prompt was 400+ words. User prompt was 600+ words.
        max_tokens=5120-8192.

    v1 problems:
        - finish_reason=2 (MAX_TOKENS truncation) on every run in fast mode.
        - 60+ seconds for the LLM call alone.
        - Output was verbose prose that didn't match the brief format we wanted.
        - LLM ignored the "thematic not paper-by-paper" instruction ~30% of time.

    v2 change (current):
        - Entire brief is now assembled deterministically from state.
        - LLM call reduced to gaps + navigation only.
        - Prompt is 80% shorter. Output is ~100-150 tokens of JSON.
        - Step 6 total runtime: ~5s (down from 60s+).
        - Zero truncation issues.
"""

SYSTEM_PROMPT = """You are a research navigator. Given a list of papers and their themes,
return ONLY valid JSON with two fields:

{
  "gaps": ["gap a as an open research question", "gap b", "gap c", "gap d"],
  "navigation": [
    {"subtopic": "subtopic name from clusters", "paper_numbers": [1, 3]},
    {"subtopic": "another subtopic", "paper_numbers": [2, 4, 5]}
  ]
}

Rules:
- gaps: 3-5 items. Each must be a specific open research question, NOT a restatement
  of a paper limitation. Bad: "Paper 3 only tests English." 
  Good: "Cross-lingual generalisation of this method remains untested."
- navigation: one entry per cluster/subtopic (use the clusters provided).
  paper_numbers are 1-indexed integers matching the paper list.
- Return ONLY the JSON object. No prose, no markdown fences."""

USER_PROMPT_TEMPLATE = """Topic: {topic}
User focus (if any): {user_note}
Themes/clusters: {clusters}

Papers (number, title, method, claim):
{titles_block}

Return the gaps + navigation JSON now."""