"""
steps/step1_parse.py
Step 1 — LLM: Parse the raw user query into structured research parameters.

Input  (from state):
    state["raw_query"]      — the user's plain-English topic string

Output (written to state):
    state["topic"]          — canonical topic title
    state["keywords"]       — 5–8 primary search keywords
    state["subtopics"]      — 3–5 sub-areas within the topic
    state["field"]          — academic field / discipline
    state["time_range"]     — relevant publication years, e.g. "2018–2024"
    state["search_strings"] — 3 ready-to-use Semantic Scholar query strings

Why this step exists:
    The raw query ("tell me about attention in transformers") is ambiguous and
    unstructured. Step 2 needs exact API query strings. Step 3 needs subtopics
    to check coverage. Step 6 needs a clean topic title for the report header.
    A single downstream prompt cannot produce all of these reliably — structured
    extraction must happen first and explicitly.

Prompt iteration note (for report appendix):
    v1 asked for JSON with no format enforcement → LLM sometimes returned
    markdown-fenced JSON or added explanatory prose before it.
    v2 (current) adds "Return ONLY the JSON object, no prose, no backticks"
    and provides a full example object so the schema is unambiguous.
"""

import json
import re

from prompts.step1_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.llm_client import call_llm


# ── Main function ──────────────────────────────────────────────────────────────

def parse_query(state: dict) -> dict:
    """
    LLM Step 1: Extract structured research parameters from raw user query.

    Args:
        state: shared pipeline state dict (reads "raw_query")

    Returns:
        state: updated with topic, keywords, subtopics, field,
               time_range, search_strings
    """
    raw_query = state["raw_query"]

    user_prompt = USER_PROMPT_TEMPLATE.format(raw_query=raw_query)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    raw_response = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        step_label="Step 1 — parse_query",
    )

    # ── Parse JSON response ────────────────────────────────────────────────────
    parsed = _safe_parse_json(raw_response, raw_query)

    # ── Write to state ─────────────────────────────────────────────────────────
    state["topic"]          = parsed.get("topic", raw_query)
    state["field"]          = parsed.get("field", "General Research")
    state["time_range"]     = parsed.get("time_range", "2018–2024")
    state["keywords"]       = parsed.get("keywords", [raw_query])
    state["subtopics"]      = parsed.get("subtopics", [])
    state["search_strings"] = parsed.get("search_strings", [raw_query])

    # Guarantee at least one search string so Step 2 never receives an empty list
    if not state["search_strings"]:
        state["search_strings"] = [raw_query]

    return state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_parse_json(text: str, fallback_query: str) -> dict:
    """
    Robustly extract a JSON object from the LLM response.
    Handles cases where the model wraps JSON in markdown fences or adds prose.
    """
    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object anywhere in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: return minimal structure so pipeline doesn't break
    print("  [WARN] Step 1: Could not parse LLM JSON — using fallback structure.")
    return {
        "topic": fallback_query,
        "field": "General Research",
        "time_range": "2018–2024",
        "keywords": fallback_query.split()[:5],
        "subtopics": [],
        "search_strings": [fallback_query],
    }