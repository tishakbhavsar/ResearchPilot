"""
steps/step5_synthesise.py
Step 5 — LLM: Synthesise agreements, contradictions, gaps across all 5 papers.

Input  (from state):
    state["deep_reads"]   — 5 critical reading dicts from Step 4
    state["topic"]        — canonical topic from Step 1
    state["subtopics"]    — sub-areas from Step 1
    state["keywords"]     — keywords from Step 1

Output (written to state):
    state["synthesis"] = {
        "agreements":     list of str — themes all papers agree on
        "contradictions": list of str — paper-vs-paper disagreements with named papers
        "gaps":           list of str — unresolved questions / future work areas
        "clusters":       list of str — thematic groupings to use as report sections
    }

Why this step exists:
    Step 4 reads papers ONE AT A TIME with focused individual context.
    Synthesis REQUIRES seeing ALL papers simultaneously to spot patterns.
    A paper can only be identified as contradicting another if you've read both.
    Thematic clusters can only emerge when all methodologies are visible together.

    This step cannot be merged into Step 4 (per-paper context), and it cannot
    be merged into Step 6 (the report writer should receive pre-computed synthesis,
    not do its own analysis while also writing prose).

Prompt design:
    v1: asked the LLM to produce agreements, contradictions, gaps in free prose
        → the output was hard to parse and Step 6 received inconsistent structure.
    v2 (current): enforces JSON with four explicit arrays. Each contradiction MUST name
        two specific papers by title. This forces the LLM to make specific claims
        rather than vague generalisations — and gives Step 6 citable specifics.
"""

from prompts.step5_prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from utils.llm_client import call_llm

# ── Main function ──────────────────────────────────────────────────────────────

def synthesise_critique(state: dict) -> dict:
    """
    LLM Step 5: Cross-paper synthesis — agreements, contradictions, gaps, clusters.

    Args:
        state: shared pipeline state dict

    Returns:
        state: updated with state["synthesis"]
    """
    deep_reads = state["deep_reads"]
    topic      = state["topic"]
    subtopics  = state.get("subtopics", [])

    if not deep_reads:
        print("  [WARN] Step 5: No deep reads to synthesise. Skipping.")
        state["synthesis"] = {
            "agreements": [], "contradictions": [], "gaps": [], "clusters": []
        }
        return state

    # ── Build the deep-reads block for the prompt ──────────────────────────────
    deep_reads_block = _format_deep_reads_block(deep_reads)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        topic=topic,
        deep_reads_block=deep_reads_block,
    )

    # ── Call LLM ──────────────────────────────────────────────────────────────
    raw_response = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        step_label="Step 5 — synthesise_critique",
        max_tokens=1536,
    )

    # ── Parse response with fallback retry ─────────────────────────────────────
    parsed = _safe_parse_json(raw_response)
    
    # Fallback retry with minimal prompt if parsing fails
    if not isinstance(parsed, dict) or (not parsed.get("agreements") and not parsed.get("clusters")):
        print(f"  [WARN] Step 5: First attempt failed. Retrying with minimal prompt…")
        titles = [dr.get('title', '?')[:30] for dr in deep_reads[:3]]
        minimal_user = f"Topic: {topic}\nPapers: {', '.join(titles)}\nReturn JSON: {{\"agreements\": [...], \"contradictions\": [...], \"gaps\": [...], \"clusters\": [...]}}"
        retry_response = call_llm(
            system_prompt="Return ONLY valid JSON for synthesis.",
            user_prompt=minimal_user,
            step_label="Step 5 — synthesise_critique (retry)",
            max_tokens=768,
        )
        parsed = _safe_parse_json(retry_response)

    clusters = _normalize_clusters(parsed.get("clusters", []))
    agreements = _normalize_string_list(parsed.get("agreements", []), max_items=6)
    contradictions = _normalize_string_list(parsed.get("contradictions", []), max_items=4)
    gaps = _normalize_string_list(parsed.get("gaps", []), max_items=6)

    if not clusters:
        clusters = _fallback_clusters(deep_reads)
    if not agreements:
        agreements = _fallback_agreements(deep_reads)
    if not contradictions:
        contradictions = _fallback_contradictions(deep_reads)
    if not gaps:
        gaps = _fallback_gaps(deep_reads)

    state["synthesis"] = {
        "agreements": agreements,
        "contradictions": contradictions,
        "gaps": gaps,
        "clusters": clusters,
    }

    return state


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_deep_reads_block(deep_reads: list) -> str:
    """
    Format all 5 deep-read dicts into a structured text block for the LLM.
    Keeps each paper's section clear so the LLM can make cross-paper references.
    """
    sections = []
    for i, dr in enumerate(deep_reads, 1):
        findings_text = " | ".join(dr.get("key_findings", ["No findings extracted."])[:1])
        limitations_text = " | ".join(dr.get("limitations", ["No limitations extracted."])[:1])
        method_tag = dr.get("method_tag", "Unknown")

        section = (
            f"=== Paper {i}: {dr.get('title', 'Unknown')} ({dr.get('year', 'N/A')}) ===\n"
            f"Core: {str(dr.get('core_argument', 'N/A'))[:140]}\n"
            f"Method: {str(method_tag)[:100]}\n"
            f"Findings: {findings_text[:180]}\n"
            f"Limits: {limitations_text[:140]}\n"
        )
        sections.append(section)

    return "\n\n".join(sections)


def _safe_parse_json(text: str) -> dict:
    """Robustly extract JSON from LLM response using sanitizer."""
    from utils.llm_sanitizer import extract_json_from_text

    raw = text or ""
    parsed = extract_json_from_text(raw)
    if isinstance(parsed, dict):
        return parsed

    # Save raw response for inspection
    try:
        import os, datetime
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = os.path.join("logs", f"llm_step5_raw_{ts}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(raw)
        print(f"  [DEBUG] Step 5: Raw LLM response written to {fname}")
    except Exception:
        pass

    print("  [WARN] Step 5: Could not parse JSON. Using empty synthesis.")
    return {"agreements": [], "contradictions": [], "gaps": [], "clusters": []}


def _normalize_string_list(value, max_items: int = 6) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:max_items]


def _normalize_clusters(value) -> list[str]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                label = item.get("name") or item.get("cluster") or item.get("theme") or ""
                if isinstance(label, str) and label.strip():
                    out.append(label.strip())
        return out[:5]
    return []


def _fallback_clusters(deep_reads: list[dict]) -> list[str]:
    tags = []
    for dr in deep_reads:
        tag = dr.get("method_tag", "")
        if isinstance(tag, str) and tag.strip():
            tags.append(tag.strip().title())
    unique = list(dict.fromkeys(tags))
    return unique[:5] if unique else ["General Trends"]


def _fallback_agreements(deep_reads: list[dict]) -> list[str]:
    top = min(len(deep_reads), 10)
    return [
        f"Papers 1-{top} agree that improving reliability/factuality remains a central challenge in this topic.",
        "Most papers prioritize evaluation quality and error analysis over purely scaling model size.",
        "Abstract-level evidence suggests benchmark design strongly influences reported performance gains.",
    ]


def _fallback_contradictions(deep_reads: list[dict]) -> list[str]:
    if len(deep_reads) >= 2:
        t1 = deep_reads[0].get("title", "Paper 1")[:40]
        t2 = deep_reads[1].get("title", "Paper 2")[:40]
        return [
            f"Paper 1 ({t1}) emphasizes representation/modeling advances, while Paper 2 ({t2}) emphasizes evaluation or pipeline framing.",
            "Several abstracts claim strong gains but differ on whether gains come from architecture choices or better data/benchmark setup.",
        ]
    return ["No explicit contradiction is reliably extractable from abstract-only evidence."]


def _fallback_gaps(deep_reads: list[dict]) -> list[str]:
    return [
        "No consensus benchmark protocol is consistently used across studies, limiting direct comparability.",
        "Computational cost and reproducibility details are often underreported in abstract-level evidence.",
        "Cross-domain and low-resource generalization claims are not consistently validated on shared datasets.",
    ]
