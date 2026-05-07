"""
utils/scorer.py
Suitability score formula for ranking papers fetched from Semantic Scholar.

This is pure Python — no LLM involved. The score is transparent and deterministic.
It is used by Step 2 to rank papers before Step 3's LLM makes semantic judgements.

Score formula (total max = 1.0):
    Component               Weight    Rationale
    ─────────────────────────────────────────────────────────────────────
    Keyword match (title)   0.30      Title match → high topical relevance
    Keyword match (abstract)0.15      Abstract match → confirmed relevance
    Citation score          0.25      High citations → community endorsement
    Recency score           0.20      Newer papers → current state of the art
    Open access             0.10      PDF available → Step 4 can read it
    ─────────────────────────────────────────────────────────────────────
    Total                   1.00

Citation scoring uses a log-scaled formula to avoid a handful of mega-cited
papers dominating the ranking.

Recency scoring rewards papers from the last 5 years but doesn't penalise
older foundational works too harshly (floor of 0.0, not negative).
"""

import math
import re
from datetime import date


# ── Current year for recency calculation ───────────────────────────────────────
CURRENT_YEAR = date.today().year


def compute_suitability_score(paper: dict, keywords: list, field: str = "") -> float:
    """
    Compute a suitability score in [0.0, 1.0] for a paper.

    Args:
        paper:    paper dict from Semantic Scholar (must have title, abstract,
                  year, citationCount, openAccessPdf)
        keywords: list of topic keywords from Step 1
        field:    academic field string (reserved for future field-specific weighting)

    Returns:
        float in [0.0, 1.0]
    """
    score = 0.0

    title    = (paper.get("title")    or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    year     = paper.get("year") or 0
    citations = paper.get("citationCount") or 0
    has_pdf  = bool(paper.get("openAccessPdf"))

    # ── Component 1: Keyword match in title (weight 0.30) ─────────────────────
    title_hits = sum(1 for kw in keywords if kw.lower() in title)
    title_ratio = title_hits / max(len(keywords), 1)
    score += 0.30 * min(title_ratio * 2, 1.0)  # *2 so 50% hit rate = full score

    # ── Component 2: Keyword match in abstract (weight 0.15) ──────────────────
    abstract_hits = sum(1 for kw in keywords if kw.lower() in abstract)
    abstract_ratio = abstract_hits / max(len(keywords), 1)
    score += 0.15 * min(abstract_ratio * 1.5, 1.0)

    # ── Component 3: Citation score — log-scaled (weight 0.25) ────────────────
    # log10(citations + 1) / log10(1001) maps [0, 1000+] → [0, 1]
    # This means: 0 cites → 0.0, 10 cites → 0.33, 100 cites → 0.67, 1000+ → ~1.0
    if citations > 0:
        citation_score = math.log10(citations + 1) / math.log10(1001)
        score += 0.25 * min(citation_score, 1.0)

    # ── Component 4: Recency score (weight 0.20) ──────────────────────────────
    # Papers from last 2 years: full score. Papers 2–7 years: linear decay.
    # Papers older than 7 years: 0.0 (unless very highly cited — handled by Comp 3)
    if year and year > 0:
        age = CURRENT_YEAR - year
        if age <= 2:
            recency_score = 1.0
        elif age <= 7:
            recency_score = 1.0 - ((age - 2) / 5.0)
        else:
            recency_score = 0.0
        score += 0.20 * recency_score

    # ── Component 5: Open access PDF available (weight 0.10) ──────────────────
    # Directly affects Step 4's ability to deep-read the paper
    if has_pdf:
        score += 0.10

    return round(score, 4)