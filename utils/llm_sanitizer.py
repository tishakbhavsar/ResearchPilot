"""
utils/llm_sanitizer.py
Utilities to extract and repair JSON-like objects from noisy LLM text.

Functions:
  - extract_json_from_text(text) -> dict | list | None

This module performs local, deterministic sanitisation (no extra LLM calls)
so it is safe to run even when provider quotas are tight.
"""
import json
import re


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _find_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced substring starting at the first open_ch, or None."""
    start = s.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i+1]

    return None


def extract_json_from_text(text: str):
    """Attempt to extract a JSON value (object or array) from `text`.

    Returns the parsed Python object on success, otherwise returns None.
    The function is conservative and does not call any external services.
    """
    if not text:
        return None

    raw = text
    s = _strip_code_fence(raw)

    # 1) Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Try to extract a balanced object {} first
    for (open_ch, close_ch) in [('{','}'), ('[',']')]:
        candidate = _find_balanced(s, open_ch, close_ch)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                # continue to next attempt
                pass

    # 3) Heuristic: remove leading non-brace text and try again
    first_brace = min([idx for idx in (s.find('{'), s.find('[')) if idx != -1] or [None])
    if first_brace is not None:
        cand = s[first_brace:]
        # try trimming trailing incomplete lines
        # attempt to find last closing brace and parse up to it
        for end in range(len(cand), max(len(cand)-300, 0), -1):
            try:
                return json.loads(cand[:end])
            except Exception:
                continue

    return None
