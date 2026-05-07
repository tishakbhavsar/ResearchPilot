"""
utils/arxiv_client.py
ArXiv PDF fetching and text extraction using PyMuPDF (fitz).

Strategy:
    1. Fetch the PDF from arxiv.org/pdf/{arxiv_id}
    2. Extract text page by page using PyMuPDF
    3. Return concatenated text of intro + methods + results sections
       (prioritise these over references/appendices)

Error handling:
    All errors return None gracefully — the caller (step4_deepread.py)
    falls back to the abstract when this returns None.
"""

import io
import re
import requests

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

HEADERS = {
    "User-Agent": "ResearchPilot/1.0 (academic literature review tool)"
}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_arxiv_text(arxiv_id: str, max_chars: int = 12000) -> str | None:
    """
    Fetch and extract text from an ArXiv paper PDF.

    Args:
        arxiv_id:  ArXiv ID, e.g. "1706.03762"
        max_chars: maximum characters to return (to fit in LLM context)

    Returns:
        Extracted text string, or None if fetch/parse fails.
    """
    arxiv_id = arxiv_id.strip()

    # ── 1. Download PDF bytes ──────────────────────────────────────────────────
    pdf_url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    return fetch_pdf_text_from_url(pdf_url, max_chars=max_chars, source_label=f"arXiv:{arxiv_id}")


def fetch_pdf_text_from_url(pdf_url: str, max_chars: int = 12000, source_label: str = "PDF") -> str | None:
    """
    Fetch and extract text from any PDF URL.

    Args:
        pdf_url: direct URL to a PDF file
        max_chars: maximum characters to return
        source_label: label used in warnings

    Returns:
        Extracted text string, or None if fetch/parse fails.
    """
    try:
        headers = {**HEADERS, **BROWSER_HEADERS}
        resp = requests.get(pdf_url, headers=headers, timeout=30)
        resp.raise_for_status()
        pdf_bytes = resp.content
    except requests.exceptions.RequestException as e:
        print(f"    [WARN] PDF download failed for {source_label}: {e}")
        return None

    if not pdf_bytes or len(pdf_bytes) < 1000:
        print(f"    [WARN] PDF too small for {source_label} ({len(pdf_bytes)} bytes)")
        return None

    # ── 2. Extract text with PyMuPDF ───────────────────────────────────────────
    try:
        import fitz   # PyMuPDF
    except ImportError:
        print("    [WARN] PyMuPDF not installed. Run: pip install pymupdf")
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"    [WARN] Could not open PDF for {source_label}: {e}")
        return None

    # Extract all page text
    all_text_pages = []
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")
            all_text_pages.append(text)
        except Exception:
            continue
    doc.close()

    full_text = "\n".join(all_text_pages)

    if len(full_text) < 500:
        print(f"    [WARN] Extracted text too short for {source_label}: {len(full_text)} chars")
        return None

    # ── 3. Extract the most useful sections ───────────────────────────────────
    # For a critical review we care about:
    #   Abstract → Introduction → Method/Approach → Results/Experiments → Conclusion
    # We skip References, Acknowledgements, Appendix (these bloat the context)
    extracted = _extract_key_sections(full_text)

    # Cap to max_chars
    if len(extracted) > max_chars:
        extracted = extracted[:max_chars]

    return extracted


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_key_sections(text: str) -> str:
    """
    Heuristically extract the most analytically valuable sections from a paper.
    Sections targeted: Abstract, Introduction, Method*, Result*, Experiment*, Conclusion.
    Sections avoided: References, Bibliography, Acknowledgement, Appendix.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # Section header patterns common in academic papers
    STOP_SECTIONS = re.compile(
        r"\n\s*(references|bibliography|acknowledgements?|appendix|about the authors?)\s*\n",
        re.IGNORECASE,
    )

    # Cut at references section to remove bibliography noise
    stop_match = STOP_SECTIONS.search(text)
    if stop_match:
        text = text[:stop_match.start()]

    return text.strip()


def fetch_pdf_text_from_landing(landing_url: str, max_chars: int = 12000, source_label: str = "landing") -> str | None:
    """
    Attempt to find a PDF from a publisher landing page or DOI URL.

    Strategy:
      - GET the landing page with a browser-like User-Agent
      - If the response is a PDF (content-type), treat like direct PDF
      - Otherwise, parse HTML for obvious PDF links (.pdf) or meta tags
      - Try the first few candidate PDF links
    """
    try:
        headers = {**HEADERS, **BROWSER_HEADERS}
        resp = requests.get(landing_url, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [WARN] Landing page fetch failed for {source_label}: {e}")
        return None

    ctype = resp.headers.get("content-type", "")
    if "application/pdf" in ctype:
        # We got a PDF directly
        try:
            return fetch_pdf_text_from_url(landing_url, max_chars=max_chars, source_label=source_label)
        except Exception:
            return None

    html = resp.text or ""
    # Find candidate PDF links in HTML
    candidates = []
    # common patterns: href="...pdf" or data-file="...pdf"
    for m in re.finditer(r'href\s*=\s*"([^"]+\.pdf)"', html, flags=re.IGNORECASE):
        candidates.append(m.group(1))
    for m in re.finditer(r"href\s*=\s*'([^']+\.pdf)'", html, flags=re.IGNORECASE):
        candidates.append(m.group(1))

    # meta tags with pdf url
    for m in re.finditer(r'<meta[^>]+content\s*=\s*"([^"]+\.pdf)"', html, flags=re.IGNORECASE):
        candidates.append(m.group(1))

    # Try absolute or relative links
    tried = set()
    from urllib.parse import urljoin
    for href in candidates:
        full = urljoin(landing_url, href)
        if full in tried:
            continue
        tried.add(full)
        txt = fetch_pdf_text_from_url(full, max_chars=max_chars, source_label=full)
        if txt:
            return txt

    return None