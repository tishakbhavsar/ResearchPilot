"""
steps/step6_report.py
Step 6 — LLM + Deterministic: Assemble the final research brief in Markdown.

Input  (from state — reads EVERYTHING accumulated across all prior steps):
    state["topic"]               — Step 1
    state["field"]               — Step 1
    state["time_range"]          — Step 1
    state["keywords"]            — Step 1
    state["papers"]              — Step 2 (full scored list for ranking table)
    state["top_papers"]          — Step 3
    state["notable_researchers"] — Step 3
    state["deep_reads"]          — Step 4
    state["synthesis"]           — Step 5
    state["user_note"]           — Confirmation gate (may be empty)

Output (written to state):
    state["report_markdown"]  — research brief as a Markdown string

Why this step exists:
    Step 6 is the only step with access to ALL accumulated state simultaneously.
    The navigation guide and gap analysis (from the LLM call here) require the
    cross-paper synthesis from Step 5 — they cannot be computed earlier.
    The final brief cannot be assembled before Step 4's per-paper analyses exist.
    Writing the brief at any earlier step would produce a shallow output because
    the synthesis (Step 5) would not yet exist.

Design change from v1:
    v1: One large LLM call asked to write a full academic prose literature review
        (~1200-2500 words). This caused finish_reason=2 (MAX_TOKENS truncation)
        even at 5120-8192 tokens, and produced a slow, verbose output that didn't
        match what a researcher actually needs.

    v2 (current): The LLM makes ONE small focused call (~512 tokens) to produce
        only the gaps and navigation guide — the two outputs that genuinely require
        language reasoning across all papers. Everything else (ranked table,
        summaries, themes, agreements, references) is assembled deterministically
        from state. This eliminates truncation entirely and reduces Step 6 runtime
        from ~60s to ~5s.

Output structure:
    1. Header (topic, field, date)
    2. Ranked table (top papers by suitability score)
    3. Key researchers (from Step 3)
    4. Paper summaries (one line each, from Step 4 core_argument)
    5. Broad themes (clusters from Step 5)
    6. Agreements & contradictions (from Step 5)
    7. Navigation guide (LLM: which papers to read for which subtopic)
    8. Critical gaps (LLM: open research questions)
    9. References
"""

from datetime import date


# ── Main function ──────────────────────────────────────────────────────────────

def write_lit_review(state: dict) -> dict:
    """
    Step 6: Assemble the final research brief.

    LLM call: one small focused call for gaps + navigation guide only.
    Everything else: deterministic assembly from state.

    Args:
        state: shared pipeline state dict

    Returns:
        state: updated with state["report_markdown"]
    """
    report = _build_brief(state)

    state["report_markdown"] = report
    print(f"  Brief generated: {len(report):,} chars / ~{len(report.split())} words")
    return state


# ── Deterministic brief assembly ──────────────────────────────────────────────

def _build_brief(state: dict) -> str:
    """
    Assemble the full research brief from state.
    No LLM calls here — all data is already in state from Steps 1-5.
    This function cannot fail due to LLM issues.
    """
    topic       = state.get("topic", "Research Topic")
    field       = state.get("field", "General Research")
    time_range  = state.get("time_range", "Recent")
    today       = date.today().strftime("%B %Y")
    deep_reads  = state.get("deep_reads", [])[:10]
    researchers = state.get("notable_researchers", [])
    synthesis   = state.get("synthesis", {})
    top_papers  = state.get("top_papers", [])[:10]

    lines = []

    # ── 1. Header ──────────────────────────────────────────────────────────────
    lines += [f"# Research Brief: {topic}"]
    lines += [f"Field: {field} | Period: {time_range} | Generated: {today}", "", "---", ""]

    # ── 2. Ranked table ────────────────────────────────────────────────────────
    lines += ["## Top 10 Papers Ranked by Suitability Score", ""]
    lines.append("| Rank | Score | Paper | Authors | Year | Venue |")
    lines.append("|------|-------|-------|---------|------|-------|")

    for i, dr in enumerate(deep_reads, 1):
        title_short = dr.get("title", "?")[:52]
        if len(dr.get("title", "")) > 52:
            title_short += "…"
        authors_list = dr.get("authors", [])
        authors_short = ", ".join(authors_list[:2])
        if len(authors_list) > 2:
            authors_short += " et al."
        lines.append(
            f"| {i} | {dr.get('score', 0):.2f} | {title_short} "
            f"| {authors_short} | {dr.get('year','?')} "
            f"| {dr.get('venue','?')} |"
        )
    lines.append("")
    lines += ["---", ""]

    # ── 3. Key researchers ─────────────────────────────────────────────────────
    if researchers:
        lines += ["## Key Researchers in This Field", ""]
        for r in researchers[:5]:
            name        = r.get("name", "Unknown")
            affiliation = r.get("affiliation", "Unknown")
            h_index     = r.get("h_index", "N/A")
            contribution = r.get("contribution", "")
            lines.append(
                f"- **{name}** ({affiliation}, h-index: {h_index})"
                + (f" — {contribution}" if contribution else "")
            )
        lines.append("")

    # ── 4. Paper summaries ─────────────────────────────────────────────────────
    lines += ["## Paper Summaries (from abstracts)", ""]
    for i, dr in enumerate(deep_reads, 1):
        authors_str  = ", ".join(dr.get("authors", [])[:2])
        core         = dr.get("core_argument", "No summary available.")
        method_tag   = dr.get("method_tag", "")
        findings     = dr.get("key_findings", [])
        finding_str  = findings[0] if findings else ""

        lead_author = authors_str.split(",")[0] if authors_str else "Unknown"
        lines.append(f"{i}. **{lead_author} et al. ({dr.get('year','?')})** — {core}")
        if finding_str:
            lines.append(f"   {finding_str}")
        lines.append("")

    # ── 5. Broad themes ────────────────────────────────────────────────────────
    clusters = _normalize_clusters(synthesis.get("clusters", []))
    if clusters:
        lines += ["## Broad Themes", ""]
        for c in clusters[:5]:
            refs = _paper_refs_for_cluster(c, deep_reads)
            lines.append(f"- {c} ({refs})")
        lines.append("")

    # ── 6. Agreements & contradictions ────────────────────────────────────────
    agreements     = synthesis.get("agreements", [])
    contradictions = synthesis.get("contradictions", [])

    if agreements or contradictions:
        lines += ["## Agreements & Contradictions", ""]
        for a in agreements[:4]:
            lines.append(f"- {a}")
        for c in contradictions[:4]:
            lines.append(f"- {c}")
        lines.append("")

    # ── Mini Discussion: concise agreements + contradictions ───────────────
    lines += ["## Mini Discussion", ""]
    if agreements:
        lines.append("- Agreements (summary):")
        for a in agreements[:3]:
            lines.append(f"  - {a}")
    else:
        # fallback: infer a short discussion from clusters
        if clusters:
            lines.append(f"- Discussion: Papers cluster around {', '.join(clusters[:3])}.")
        else:
            lines.append("- Discussion: No clear agreement emerged from the abstracts.")

    if contradictions:
        lines.append("- Contradictions (summary):")
        for c in contradictions[:3]:
            lines.append(f"  - {c}")
    else:
        lines.append("- Contradictions: No explicit contradictions were evident across reviewed abstracts.")

    lines.append("")

    # ── 7. Navigation guide ────────────────────────────────────────────────────
    lines += ["## Navigation Guide", ""]
    for c in clusters[:5]:
        refs = _paper_refs_for_cluster(c, deep_reads)
        lines.append(f"- For **{c.lower()}**: start with {refs}")
    lines.append("")

    # ── 8. Critical gaps ──────────────────────────────────────────────────────
    gaps = synthesis.get("gaps", [])
    if gaps:
        lines += ["## Critical Gaps to Explore", ""]
        for j, g in enumerate(gaps[:6], ord("a")):
            lines.append(f"{chr(j)}) {g}")
        lines.append("")

    # ── Mini Critical Gaps (Actionable) — short, solvable items ─────────────
    lines += ["## Mini Critical Gaps (Actionable)", ""]
    actionable = gaps[:3] if gaps else []
    if not actionable:
        # derive from per-paper limitations as a fallback
        seen = set()
        for dr in deep_reads:
            for lim in (dr.get("limitations") or [])[:1]:
                txt = str(lim).strip()
                if txt and txt not in seen:
                    actionable.append(txt)
                    seen.add(txt)
                if len(actionable) >= 3:
                    break
            if len(actionable) >= 3:
                break

    if actionable:
        for i, a in enumerate(actionable, 1):
            lines.append(f"{i}. {a} — Suggested next step: run focused experiments or targeted benchmarking to resolve this gap.")
    else:
        lines.append("No specific actionable gaps identified from the abstracts. Consider targeted benchmarking and direct method comparisons to surface practical gaps.")

    lines.append("")

    return "\n".join(lines)


def _normalize_clusters(clusters) -> list[str]:
    out = []
    if not isinstance(clusters, list):
        return out
    for c in clusters:
        if isinstance(c, str) and c.strip():
            out.append(c.strip())
        elif isinstance(c, dict):
            label = c.get("name") or c.get("theme") or c.get("cluster")
            if isinstance(label, str) and label.strip():
                out.append(label.strip())
    return out[:5]


def _paper_refs_for_cluster(cluster: str, deep_reads: list[dict]) -> str:
    if not cluster:
        return "Papers 1, 2"
    c = cluster.lower()
    matches = []
    for i, dr in enumerate(deep_reads, 1):
        text = f"{dr.get('method_tag','')} {dr.get('core_argument','')}".lower()
        if any(token in text for token in c.split()[:3]):
            matches.append(i)
    if not matches:
        matches = list(range(1, min(3, len(deep_reads)) + 1))
    refs = ", ".join(f"Papers {n}" if idx == 0 else str(n) for idx, n in enumerate(matches[:3]))
    return refs