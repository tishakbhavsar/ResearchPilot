"""
prompts/step4_prompt.py
System and user prompt for Step 4 — abstract_read (LLM).

PURPOSE OF THIS STEP:
    Extract structured signals from each paper's abstract, one paper at a time.
    The output feeds Step 5 (cross-paper synthesis) and Step 6 (brief assembly).

PROMPT DESIGN RATIONALE:
    1. method_tag is new in v2 — a 3-5 word label for the method used.
       This enables Step 6 to write "Papers using [method_tag] include..."
       which produces cleaner thematic groupings than free-text methodology.

    2. key_findings capped at 2 (was 3-5 in v1).
       For abstract-only reading, extracting more than 2 findings is
       speculative — abstracts rarely report more than 2 concrete results.
       Fewer fields = smaller output = no truncation.

    3. inner_citations removed (was in v1).
       Abstracts almost never mention specific citations. This field was
       always empty or hallucinated. Removed entirely.

    4. limitations capped at 1.
       Abstracts rarely acknowledge limitations. One is sufficient signal
       for Step 5's gap analysis.

    5. max_tokens=512 (set in step4_deepread.py).
       This schema produces ~100-150 tokens of valid JSON. 512 gives
       comfortable headroom without risking truncation.

PROMPT ITERATION (v1 → v2):
    v1 prompt: asked for core_argument (3 sentences), methodology (paragraph),
               key_findings (3-5), limitations (2-3), inner_citations (list).
               max_tokens=1536-2048.

    v1 problem: finish_reason=2 (MAX_TOKENS hit) on 2/3 papers in fast mode.
               Truncated JSON caused parse failures and repair-pass LLM calls,
               adding latency without improving output quality.
               inner_citations was hallucinated because abstracts have no citations.

    v2 change (current): Reduced to 4 fields, all short.
               core_argument = 1 sentence only.
               method_tag = 3-5 word label.
               key_findings = max 2 items.
               limitations = max 1 item.
               Result: clean JSON in ~100-150 tokens. Zero truncation issues.
"""

SYSTEM_PROMPT = """You are extracting structured signals from a research paper abstract.

Return ONLY valid JSON. No prose, no markdown fences.

Output schema — 4 fields only:
{
  "core_argument": "<ONE sentence: what does this paper claim or demonstrate?>",
  "method_tag":    "<3-5 words describing the method, e.g. 'fine-tuning mBERT', 'data augmentation CNN', 'RAG with dense retrieval'>",
  "key_findings":  ["<finding 1, start with a verb: Shows/Achieves/Demonstrates/Finds>", "<finding 2>"],
  "limitations":   ["<one main limitation, or 'Not stated in abstract' if absent>"]
}

Rules:
- core_argument must be exactly ONE sentence.
- method_tag must be 3-5 words, no full sentences.
- key_findings must have exactly 2 items. Start each with a result verb.
- limitations must have exactly 1 item.
- Return nothing outside the JSON object."""

USER_PROMPT_TEMPLATE = """Topic: "{topic}"

Paper: {title} ({year})

Abstract:
{paper_text}

Return the 4-field JSON now."""