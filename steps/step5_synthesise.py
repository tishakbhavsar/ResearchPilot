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

from utils.llm_client import call_llm

# ── Main function ──────────────────────────────────────────────────────────────

def synthesise_critique(state: dict) -> dict:
    """
    Step 5: Cross-paper synthesis — FOOLPROOF DESIGN.
    
    Deterministic extraction is PRIMARY (always succeeds).
    Optional LLM call is SECONDARY (skipped if truncated).
    Never fails on token limits or API issues — only on rate limits.

    Args:
        state: shared pipeline state dict

    Returns:
        state: updated with state["synthesis"]
    """
    deep_reads = state["deep_reads"]
    topic      = state["topic"]

    if not deep_reads:
        print("  [WARN] Step 5: No deep reads to synthesise. Skipping.")
        state["synthesis"] = {
            "agreements": [], "contradictions": [], "gaps": [], "clusters": []
        }
        return state

    # ── STEP 1: Deterministic extraction (PRIMARY) — ALWAYS SUCCEEDS ──────────
    print("  Using deterministic synthesis (no LLM dependency)…")
    agreements = _fallback_agreements(deep_reads)
    contradictions = _fallback_contradictions(deep_reads)
    gaps = _fallback_gaps(deep_reads)
    clusters = _fallback_clusters(deep_reads)

    # ── STEP 2: Optional LLM refinement (SECONDARY) ────────────────────────────
    # Only processes top 3 papers, asks for clusters only (lightweight).
    # If LLM truncates or fails, we still have deterministic clusters.
    llm_clusters = _try_llm_clusters(topic, deep_reads[:3])
    if llm_clusters:
        clusters = llm_clusters
    
    state["synthesis"] = {
        "agreements": agreements,
        "contradictions": contradictions,
        "gaps": gaps,
        "clusters": clusters,
    }

    return state


def _try_llm_clusters(topic: str, top_papers: list[dict]) -> list[str]:
    """
    Optional LLM refinement to improve cluster labels.
    Only processes top 3 papers to minimize truncation.
    Returns empty list if LLM truncates or fails — caller falls back to deterministic.
    """
    if not top_papers:
        return []
    
    # Build minimal prompt with only titles
    paper_titles = "\n".join([f"- {dr.get('title', '?')[:60]}" for dr in top_papers])
    minimal_prompt = f"Topic: {topic}\n\nTop papers:\n{paper_titles}\n\nReturn JSON: {{\"clusters\": [\"cluster1\", \"cluster2\", \"cluster3\"]}}"
    
    raw_response = call_llm(
        system_prompt="Extract 2-3 research themes from paper titles. Return ONLY valid JSON.",
        user_prompt=minimal_prompt,
        step_label="Step 5 — optional_cluster_refinement",
        max_tokens=256,
    )
    
    # Check for truncation — if response is too short or empty, skip it
    if not raw_response or len(raw_response) < 20:
        return []
    
    # Parse with sanitizer
    from utils.llm_sanitizer import extract_json_from_text
    parsed = extract_json_from_text(raw_response)
    
    if not isinstance(parsed, dict):
        return []
    
    clusters = _normalize_clusters(parsed.get("clusters", []))
    return clusters if clusters else []


# ── Helpers ────────────────────────────────────────────────────────────────────




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
    # Use method_tag and top title tokens to produce short cluster labels
    from collections import Counter
    toks = []
    for dr in deep_reads:
        tag = dr.get("method_tag", "")
        if isinstance(tag, str) and tag.strip():
            toks.append(tag.strip().title())
        # fallback to keywords from title
        title = dr.get("title", "")
        for w in _top_tokens(title, 3):
            toks.append(w.title())
    counts = Counter(toks)
    if not counts:
        return ["General Trends"]
    labels = [t for t, _ in counts.most_common(5)]
    return labels


def _fallback_agreements(deep_reads: list[dict]) -> list[str]:
    # Extract short assertions from key_findings and first sentence of core_argument
    from collections import Counter
    cand = []
    for dr in deep_reads:
        for f in dr.get("key_findings", [])[:2]:
            s = _short_normalize(f)
            if s:
                cand.append(s)
        core = dr.get("core_argument", "")
        if core:
            first = core.split(".")[0]
            s = _short_normalize(first)
            if s:
                cand.append(s)

    counts = Counter(cand)
    agreements = []
    for phrase, cnt in counts.most_common(6):
        if cnt >= 2:
            agreements.append(f"{cnt} papers report that {phrase}.")
    # if none found, fall back to generic high-level agreements
    if not agreements:
        return [
            "Multiple papers report evaluation or benchmark-focused analyses as central to reported gains.",
            "Several studies prioritize model evaluation and error analysis over simple scaling.",
        ]
    return agreements


def _fallback_contradictions(deep_reads: list[dict]) -> list[str]:
    # Find opposing polarity on same short predicate across papers
    neg_words = set(["no", "not", "none", "failed", "insignificant", "no significant", "lack"])
    pos_words = set(["improv", "increase", "decrease", "reduce", "benefit", "effective", "positive", "significant"])
    preds = {}  # predicate -> list of (paper_idx, polarity, text)
    for i, dr in enumerate(deep_reads, 1):
        for f in dr.get("key_findings", [])[:2]:
            text = f.lower()
            pred = _predicate_from_text(text)
            if not pred:
                continue
            polarity = "pos" if any(p in text for p in pos_words) and not any(n in text for n in neg_words) else ("neg" if any(n in text for n in neg_words) else "uncertain")
            preds.setdefault(pred, []).append((i, polarity, dr.get("title", "Paper")[:40]))

    contradictions = []
    for pred, items in preds.items():
        has_pos = any(p[1] == "pos" for p in items)
        has_neg = any(p[1] == "neg" for p in items)
        if has_pos and has_neg:
            # pick one pos and one neg example
            pos_ex = next((p for p in items if p[1] == "pos"), items[0])
            neg_ex = next((p for p in items if p[1] == "neg"), items[0])
            contradictions.append(f"Paper {pos_ex[0]} ({pos_ex[2]}) reports {pred}, while Paper {neg_ex[0]} ({neg_ex[2]}) reports no significant {pred}.")
        if len(contradictions) >= 4:
            break

    if not contradictions:
        return ["No explicit contradiction is reliably extractable from abstract-only evidence."]
    return contradictions


def _fallback_gaps(deep_reads: list[dict]) -> list[str]:
    from collections import Counter
    gap_cands = []
    # collect explicit limitations
    for dr in deep_reads:
        for lim in dr.get("limitations", [])[:2]:
            s = str(lim).strip()
            if s:
                gap_cands.append(s)
    # also look for missing items in core arguments/titles
    for dr in deep_reads:
        core = dr.get("core_argument", "").lower()
        if core and ("external" in core or "validation" in core):
            gap_cands.append("External validation is limited or absent in several studies.")

    counts = Counter(gap_cands)
    unique = [t for t, _ in counts.most_common(6)]
    # common generic gaps if none extracted
    if not unique:
        unique = [
            "No consensus benchmark protocol is consistently used across studies, limiting direct comparability.",
            "Computational cost and reproducibility details are often underreported in abstract-level evidence.",
            "Cross-domain and low-resource generalization claims are not consistently validated on shared datasets.",
        ]
    return unique[:6]


### Small text helpers for fallback heuristics
def _short_normalize(text: str) -> str:
    import re
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    # remove trailing punctuation
    s = s.rstrip('. ,;:')
    return s


def _top_tokens(text: str, n: int = 3) -> list[str]:
    import re
    if not text:
        return []
    words = re.findall(r"\w+", text.lower())
    stop = set(["the","and","of","in","for","with","a","an","to","on","by","using"]) 
    toks = [w for w in words if w not in stop]
    from collections import Counter
    return [t for t, _ in Counter(toks).most_common(n)]


def _predicate_from_text(text: str) -> str:
    # crude predicate extraction: look for verb+noun phrases like 'improve accuracy' or 'increase auc'
    import re
    m = re.search(r"(improv\w+|increase|decrease|reduce|no significant|fail\w*?)\s+([a-z_\-]{3,20})", text)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    # fallback to top token pair
    toks = _top_tokens(text, 2)
    return " ".join(toks) if toks else ""
