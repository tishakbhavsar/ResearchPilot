# ResearchPilot — Automated Literature Review Agent

ResearchPilot is a compact multi-step agent that converts a plain-English
research topic into a concise, cited literature brief. It combines tool-based
paper discovery with small, focused LLM calls and deterministic assembly to
produce a fast, reliable Top-10 research brief suitable for quick surveys.

## Quick start

1) Configure for GROK (edit `config.py` or use env vars)

In `config.py` set the provider and ensure the key is available from the
environment (recommended):

```python
LLM_PROVIDER = "grok"
# prefer environment variables for secrets
GROK_API_KEY = os.environ.get("GROK_API_KEY")
```

Or set the environment variable in your shell:

```bash
export GROK_API_KEY="your-grok-api-key"
```

2) Run from the terminal

```bash
# Run interactively (uses provider from config.py)
python main.py

# Run with a query (uses config provider or override)
python main.py "attention mechanisms in transformer models"

# Force use of Grok for a single run
python main.py --provider grok "federated learning for healthcare"

# Force use of Gemini for a single run
python main.py --provider gemini "your query"
```

3) Run with Streamlit UI

Install dependencies and run the web UI:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit UI lets you pick the provider (`mock`, `grok`, `gemini`) and
toggle mock fallback behavior.

## How it works (steps & outputs)

The pipeline is linear — each step augments a shared `state` dict and feeds
the next stage. Output is written to `output/lit_review.md` and `state`.

- Step 1 — `step1_parse.py` (LLM): parse the user query into
    `keywords`, `subtopics`, and `search_strings`. Output: structured search terms.
- Step 2 — `step2_fetch.py` (Tool): query Semantic Scholar (and OpenAlex as
    fallback), fetch ~30–50 candidate papers, compute a suitability score, and
    profile notable authors. Output: `papers` and scored candidates.
- Step 3 — `step3_relevance.py` (LLM/heuristic): select Top-10 papers and
    assemble `notable_researchers`. Output: `top_papers` and researcher list.
- Step 4 — `step4_deepread.py` (Tool + minimal LLM): fetch abstracts (PDFs
    when available), extract core arguments, key findings, and limitations for
    each Top-10 paper. Output: `deep_reads` (per-paper summaries).
- Step 5 — `step5_synthesise.py` (LLM): cross-paper synthesis to produce
    `agreements`, `contradictions`, `clusters`, and `gaps`. A sanitizer + repair
    pass is used to handle truncated responses; when parsing fails, a
    deterministic fallback extracts topics from `deep_reads`.
- Step 6 — `step6_report.py` (deterministic + small LLM call): assemble the
    final brief — ranked table, one-line summaries, themes, a short
    `Mini Discussion` (agreements/contradictions), and `Mini Critical Gaps`
    (actionable items). Output: `output/lit_review.md` (Markdown brief).

Typical runtime (normal mode) is configurable; with abstract-only Top-10 it
commonly runs in ~30–120s depending on network and caching.

## Output format

The generated `output/lit_review.md` contains:

- Header with topic, field, and date
- Top-10 ranked table (suitability score, authors, year, venue)
- Key researchers
- One-line paper summaries (from abstracts)
- Broad themes (clusters)
- Mini Discussion (concise agreements + contradictions)
- Navigation guide (which papers to read first for each theme)
- Critical gaps & Mini Critical Gaps (actionable next steps)

## Troubleshooting & notes

- If LLM calls fail or truncate, the pipeline attempts a repair pass and
    deterministic fallbacks so the final brief still generates.
- To reduce runtime, use abstract-only reading (current default) and enable
    caching for Semantic Scholar results (`.s2_cache`).

## Credits & external tools

- Google Gemini: https://developers.generativeai.google/
- Grok (x.ai): https://x.ai/
- Claude (Anthropic) — used as a design reference: https://www.anthropic.com/

If you'd like, I can also add a short example `config.py` snippet for GROK
and an example `requirements.txt` fragment next.