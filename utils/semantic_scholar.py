"""
utils/semantic_scholar.py
All Semantic Scholar Academic Graph API calls — isolated in one place.

Rate limiting strategy:
    On 429, uses exponential backoff: 10s → 20s → 40s → give up after 3 retries.
    If all retries fail, returns an empty list rather than hanging the pipeline.
    This is much better than the original infinite-retry loop.
"""

import time
import requests
import os
import json
import hashlib
from dotenv import load_dotenv

# Load environment variables from .env so SEMANTIC_SCHOLAR_API_KEY is available
load_dotenv()

BASE_URL = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = ",".join([
    "paperId", "title", "abstract", "year", "citationCount",
    "authors", "externalIds", "venue", "openAccessPdf",
    "referenceCount", "influentialCitationCount",
])

AUTHOR_FIELDS = ",".join([
    "authorId", "name", "hIndex", "paperCount", "affiliations", "url",
])

HEADERS = {
    "User-Agent": "ResearchPilot/1.0 (academic literature review tool)"
}

# Simple on-disk cache to reduce repeated S2 queries during demos
CACHE_DIR = ".s2_cache"

def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()

def _load_cache(query: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(query) + ".json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_cache(query: str, data: list):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(query) + ".json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

# If the user provided an API key in the environment, add it to headers.
# DO NOT read any .ev files as requested by the user.
api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
if api_key:
    HEADERS["x-api-key"] = str(api_key)

# Client-side throttle: ensure at most 1 request per second to respect
# Semantic Scholar's guidance (and the user's account rate limit).
_LAST_REQUEST_TS: float | None = None
_MIN_INTERVAL = 1.0

def _throttled_get(url: str, params: dict, timeout: int = 15):
    global _LAST_REQUEST_TS
    now = time.time()
    if _LAST_REQUEST_TS is not None:
        elapsed = now - _LAST_REQUEST_TS
        if elapsed < _MIN_INTERVAL:
            to_sleep = _MIN_INTERVAL - elapsed
            time.sleep(to_sleep)
    resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    _LAST_REQUEST_TS = time.time()
    return resp


MAX_RETRIES   = 3    # attempts before giving up on a single query
INITIAL_WAIT  = 10   # seconds for first retry
BACKOFF_MULT  = 2    # multiply wait time each retry: 10 → 20 → 40


def search_papers(query: str, limit: int = 20) -> list:
    """
    Search Semantic Scholar for papers matching a query string.
    Returns empty list on persistent failure — does not crash the pipeline.

    Args:
        query: search string (e.g. "attention mechanism transformer")
        limit: max results to return

    Returns:
        list of paper dicts, or [] on failure
    """
    url    = f"{BASE_URL}/paper/search"
    params = {"query": query, "limit": min(limit, 100), "fields": PAPER_FIELDS}
    wait   = INITIAL_WAIT

    # Check cache first (helps demo speed and avoids rate limits)
    cached = _load_cache(query)
    if cached is not None:
        print(f"    [CACHE] Returning cached results for: '{query}'")
        return cached

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _throttled_get(url, params, timeout=15)

            if resp.status_code == 429:
                print(f"    [WARN] Semantic Scholar rate limit — waiting {wait}s (attempt {attempt}/{MAX_RETRIES})…")
                time.sleep(wait)
                wait *= BACKOFF_MULT
                continue   # retry

            resp.raise_for_status()
            data   = resp.json()
            papers = data.get("data", [])
            papers = [p for p in papers if p.get("title")]
            # Save to cache for subsequent runs
            try:
                _save_cache(query, papers)
            except Exception:
                pass
            return papers

        except requests.exceptions.Timeout:
            print(f"    [WARN] Semantic Scholar timeout on attempt {attempt}/{MAX_RETRIES}")
            time.sleep(wait)
            wait *= BACKOFF_MULT

        except requests.exceptions.RequestException as e:
            print(f"    [WARN] Semantic Scholar request error: {e}")
            return []

    print(f"    [WARN] Semantic Scholar gave up after {MAX_RETRIES} retries for query: '{query}'")
    return []


def get_author_details(author_id: str) -> dict | None:
    """
    Fetch detailed profile for a single author. Returns None on failure.
    """
    if not author_id:
        return None

    url    = f"{BASE_URL}/author/{author_id}"
    params = {"fields": AUTHOR_FIELDS}
    wait   = INITIAL_WAIT

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _throttled_get(url, params, timeout=10)

            if resp.status_code == 404:
                return None

            if resp.status_code == 429:
                print(f"    [WARN] S2 author rate limit — waiting {wait}s…")
                time.sleep(wait)
                wait *= BACKOFF_MULT
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.RequestException as e:
            print(f"    [WARN] Author fetch error for {author_id}: {e}")
            return None

    return None


def get_paper_by_id(paper_id: str) -> dict | None:
    """Fetch a single paper by Semantic Scholar paper ID."""
    url    = f"{BASE_URL}/paper/{paper_id}"
    params = {"fields": PAPER_FIELDS}

    try:
        resp = _throttled_get(url, params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"    [WARN] Paper fetch failed for {paper_id}: {e}")
        return None
