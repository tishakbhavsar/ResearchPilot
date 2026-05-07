"""
utils/semantic_scholar_open_alex.py
Alternative implementation of the Semantic Scholar helpers using OpenAlex.

This file mirrors the public functions from utils/semantic_scholar.py so
you can experiment with OpenAlex as a drop-in replacement. Function names
and signatures are intentionally the same: `search_papers`,
`get_author_details`, `get_paper_by_id`.

Notes on fields mapping:
- paperId: OpenAlex work id (e.g. https://openalex.org/Wxxxx)
- title, abstract (best-effort), year, citationCount (cited_by_count)
- authors: list of {authorId, name}
- externalIds: may contain doi, arxiv (when available via ids)

Rate limiting: implements exponential backoff and returns empty/None on
persistent failures to avoid crashing the pipeline.
"""

import time
import requests

BASE_URL = "https://api.openalex.org"
WEB_BASE_URL = "https://openalex.org"

# Default fields we attempt to map from OpenAlex works
MAX_RETRIES = 3
INITIAL_WAIT = 5
BACKOFF_MULT = 2

HEADERS = {
    "User-Agent": "ResearchPilot-OpenAlex/1.0"
}


def _safe_get(url: str, params: dict | None = None, timeout: int = 15) -> dict | None:
    wait = INITIAL_WAIT
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                print(f"    [WARN] OpenAlex rate limit — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})…")
                time.sleep(wait)
                wait *= BACKOFF_MULT
                continue
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "json" not in content_type:
                snippet = resp.text[:160].replace("\n", " ")
                raise ValueError(f"non-JSON response (status {resp.status_code}): {snippet}")
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    [WARN] OpenAlex request error: {e}")
            time.sleep(wait)
            wait *= BACKOFF_MULT
        except ValueError as e:
            print(f"    [WARN] OpenAlex response error: {e}")
            time.sleep(wait)
            wait *= BACKOFF_MULT
    print("    [WARN] OpenAlex gave up after retries for URL: {}".format(url))
    return None


def _extract_abstract(work: dict) -> str:
    # OpenAlex sometimes provides abstract_inverted_index; reconstruct if present
    inv = work.get("abstract_inverted_index")
    if isinstance(inv, dict) and inv:
        # reconstruct by placing words by index (best-effort)
        try:
            words = []
            # inv maps token -> list of positions
            positions = {}
            for token, idxs in inv.items():
                for i in idxs:
                    positions[i] = token
            # build by position order
            words = [positions[i] for i in sorted(positions.keys())]
            return " ".join(words)
        except Exception:
            pass

    # Fall back to 'abstract' field if present
    abst = work.get("abstract")
    if isinstance(abst, str):
        return abst

    return ""


def search_papers(query: str, limit: int = 20) -> list:
    """
    Search OpenAlex works for papers matching a query string.

    Returns a list of paper-like dicts compatible with the rest of the pipeline.
    """
    per_page = min(limit, 200)
    url = f"{BASE_URL}/works"
    params = {"search": query, "per_page": per_page}

    data = _safe_get(url, params=params)
    if not data:
        return []

    results = data.get("results") or data.get("data") or []
    papers = []
    for w in results:
        try:
            paper = {
                "paperId": w.get("id"),
                "title": w.get("display_name") or w.get("title"),
                "abstract": _extract_abstract(w),
                "year": w.get("publication_year") or w.get("publication_date", "N/A")[:4],
                "citationCount": w.get("cited_by_count", 0),
                "authors": [],
                "externalIds": {},
                "venue": (w.get("host_venue") or {}).get("display_name", ""),
                "openAccessPdf": {},
            }

            # authors
            authors = []
            for auth in w.get("authorships", [])[:10]:
                a = auth.get("author") or {}
                authors.append({
                    "authorId": a.get("id"),
                    "name": a.get("display_name")
                })
            paper["authors"] = authors

            # ids: DOI, arXiv etc.
            ids = w.get("ids") or {}
            if ids.get("openalex"):
                paper["externalIds"]["OpenAlex"] = ids.get("openalex")
            if ids.get("doi"):
                paper["externalIds"]["DOI"] = ids.get("doi")
            if ids.get("arxiv"):
                paper["externalIds"]["ArXiv"] = ids.get("arxiv")

            # primary location for OA PDF
            primary = w.get("primary_location") or {}
            if primary and primary.get("landing_page_url"):
                paper["openAccessPdf"] = {"url": primary.get("landing_page_url")}

            papers.append(paper)
        except Exception:
            continue

    return papers


def get_author_details(author_id: str) -> dict | None:
    """
    Fetch author details from OpenAlex. Accepts either full OpenAlex id (https://openalex.org/A...) or plain id.
    Returns a dict similar to Semantic Scholar's author profile, or None on failure.
    """
    if not author_id:
        return None

    # Normalize id
    raw_id = str(author_id).strip()
    if raw_id.startswith(WEB_BASE_URL + "/"):
        raw_id = raw_id.rsplit("/", 1)[-1]
    if raw_id.startswith("A"):
        aid = f"{BASE_URL}/authors/{raw_id}"
    else:
        aid = f"{BASE_URL}/authors/{raw_id}"

    data = _safe_get(aid)
    if not data:
        return None

    try:
        institution = data.get("last_known_institution") or {}
        if isinstance(institution, dict):
            affiliations = [institution.get("display_name")] if institution.get("display_name") else []
        else:
            affiliations = []

        summary_stats = data.get("summary_stats") or {}
        profile = {
            "authorId": data.get("id"),
            "name": data.get("display_name"),
            "hIndex": data.get("h_index") or summary_stats.get("h_index") or None,
            "paperCount": data.get("works_count") or None,
            "affiliations": affiliations,
            "url": data.get("id")
        }
        return profile
    except Exception:
        return None


def get_paper_by_id(paper_id: str) -> dict | None:
    """
    Fetch a single paper/work by OpenAlex id (or full URL) and return a paper-like dict.
    """
    if not paper_id:
        return None

    raw_id = str(paper_id).strip()
    if raw_id.startswith(WEB_BASE_URL + "/"):
        raw_id = raw_id.rsplit("/", 1)[-1]
    url = f"{BASE_URL}/works/{raw_id}"

    data = _safe_get(url)
    if not data:
        return None

    try:
        w = data
        paper = {
            "paperId": w.get("id"),
            "title": w.get("display_name") or w.get("title"),
            "abstract": _extract_abstract(w),
            "year": w.get("publication_year") or w.get("publication_date", "N/A")[:4],
            "citationCount": w.get("cited_by_count", 0),
            "authors": [],
            "externalIds": {},
            "venue": (w.get("host_venue") or {}).get("display_name", ""),
            "openAccessPdf": {},
        }

        authors = []
        for auth in w.get("authorships", [])[:10]:
            a = auth.get("author") or {}
            authors.append({"authorId": a.get("id"), "name": a.get("display_name")})
        paper["authors"] = authors

        ids = w.get("ids") or {}
        if ids.get("doi"):
            paper["externalIds"]["DOI"] = ids.get("doi")
        if ids.get("arxiv"):
            paper["externalIds"]["ArXiv"] = ids.get("arxiv")

        primary = w.get("primary_location") or {}
        if primary and primary.get("landing_page_url"):
            paper["openAccessPdf"] = {"url": primary.get("landing_page_url")}

        return paper
    except Exception:
        return None
