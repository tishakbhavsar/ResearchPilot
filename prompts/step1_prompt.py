"""
prompts/step1_prompt.py
System and user prompt for Step 1 — parse_query (LLM).

PURPOSE OF THIS STEP:
    Convert a plain-English research topic into structured JSON fields that
    drive every subsequent step. Step 2 cannot call the Semantic Scholar API
    without search_strings. Step 3 cannot check subtopic coverage without
    subtopics. Step 6 cannot write the report header without topic and field.

PROMPT DESIGN RATIONALE:
    The system prompt enforces three things:
    1. Output format — "Return ONLY a valid JSON object. No prose, no fences."
       Without this, the LLM often wraps JSON in ```json...``` blocks or adds
       an explanation sentence before the JSON, breaking json.loads().
    2. Constraint on search_strings — "3–6 words each, no boolean operators."
       Semantic Scholar's API performs poorly on long or boolean queries.
       Short keyword phrases return better results.
    3. time_range heuristic — "default to last 5–7 years for fast-moving fields."
       Without guidance, the LLM sometimes returns "all years" which is useless
       for filtering recency scores in Step 2.

PROMPT ITERATION (v1 → v2):
    v1 prompt:
        "Parse this research topic and return JSON with: topic, keywords,
         subtopics, field, time_range, search_strings."

    v1 problem:
        The LLM returned markdown-fenced JSON about 40% of the time:
            ```json
            { "topic": "...", ... }
            ```
        json.loads() failed on this every time. We had to add a regex strip
        in the parser just to handle v1's output.

    v2 change (current):
        Added "Return ONLY a valid JSON object. No prose, no markdown fences,
        no explanation before or after." to the system prompt.
        Added a full example output object so the schema is unambiguous.

    v2 result:
        Clean JSON on first attempt ~95% of the time. The regex strip in
        step1_parse.py still exists as a safety net for the remaining 5%.
"""

SYSTEM_PROMPT = """You are a research librarian and academic search specialist.
Your job is to parse a user's research topic into a structured JSON object that
will drive an automated literature review pipeline.

Rules:
- Return ONLY a valid JSON object. No prose, no markdown fences, no explanation.
- All string values must be in English.
- search_strings must be short enough to use directly as Semantic Scholar API
  queries (3–6 words each, no boolean operators, no quotes, no AND/OR).
- keywords must be single words or short noun phrases (no full sentences).
- time_range should be a string like "2018–2024". Pick based on field velocity:
  fast-moving (deep learning, NLP) → last 5–6 years.
  slower (theory, linguistics) → last 8–10 years.

Output schema — return exactly this structure:
{
  "topic": "<canonical one-line topic title>",
  "field": "<primary academic discipline, e.g. 'Computer Science / NLP'>",
  "time_range": "<e.g. 2019–2024>",
  "keywords": ["kw1", "kw2", "kw3", "kw4", "kw5"],
  "subtopics": ["subtopic1", "subtopic2", "subtopic3"],
  "search_strings": [
    "<query string 1>",
    "<query string 2>",
    "<query string 3>"
  ]
}"""

USER_PROMPT_TEMPLATE = """Parse the following research topic for a literature review pipeline.

Research topic: "{raw_query}"

Return the JSON object only."""