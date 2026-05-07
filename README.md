# ResearchPilot — Automated Literature Review Agent

A multi-step LLM agent that takes a plain-English research topic and produces a
fully cited, critically argued literature review — in one run.

## What it does

```
User query → Step 1 (LLM) → Step 2 (Tool) → Step 3 (LLM) →
Step 4 (Tool+LLM) → Step 5 (LLM) → [You confirm] → Step 6 (LLM) → lit_review.md
```

| Step | Type | What it does |
|------|------|-------------|
| 1 | LLM | Parse query → keywords, subtopics, search strings |
| 2 | Tool | Semantic Scholar API → 30–50 papers, scored & ranked |
| 3 | LLM | Select top 5 papers + profile notable researchers |
| 4 | Tool+LLM | Fetch ArXiv PDFs → critical reading per paper |
| 5 | LLM | Cross-paper synthesis: agreements, contradictions, gaps |
| Gate | You | Confirm or add focus note |
| 6 | LLM | Write full cited Markdown literature review |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` — set your API key and provider:

```python
LLM_PROVIDER = "grok"        # or "gemini"
GROK_API_KEY = "your-key"    # from https://console.x.ai/
```

Or use environment variables:
```bash
export GROK_API_KEY="your-key"
```

## Running

```bash
# Interactive mode
python main.py

# Query as argument
python main.py "attention mechanisms in transformer models"
python main.py "federated learning for privacy-preserving healthcare AI"

# Streamlit UI
streamlit run app.py
```

Output is saved to `output/lit_review.md`.

In the Streamlit app, you can choose `mock`, `gemini`, or `grok`, and toggle fallback to the mock provider when an API request fails.

## Chain dependency structure

Each step **cannot be removed** without breaking the chain:

- **Step 2** cannot run without Step 1's `search_strings`
- **Step 3** cannot run without Step 2's `papers` and `authors`
- **Step 4** cannot run without Step 3's `top_papers` (exactly 5 papers to read)
- **Step 5** cannot run without Step 4's `deep_reads` (needs per-paper analyses to synthesise)
- **Step 6** cannot run without Step 5's `synthesis` (needs clusters, contradictions, gaps for structure)

## Project structure

```
researchpilot/
├── main.py                  ← orchestrates the full chain (read this first)
├── config.py                ← API keys + provider (only file to edit)
├── requirements.txt
├── README.md
├── steps/
│   ├── step1_parse.py       ← LLM: extract query structure
│   ├── step2_fetch.py       ← TOOL: Semantic Scholar API + scoring
│   ├── step3_relevance.py   ← LLM: select top 5 + researcher profiles
│   ├── step4_deepread.py    ← TOOL: ArXiv PDF fetch + LLM critical read
│   ├── step5_synthesise.py  ← LLM: cross-paper synthesis
│   └── step6_report.py      ← LLM: write final cited report
├── utils/
│   ├── llm_client.py        ← single LLM call function (Grok or Gemini)
│   ├── semantic_scholar.py  ← all S2 API calls
│   ├── arxiv_client.py      ← PDF fetch + text extraction
│   ├── scorer.py            ← suitability score formula (pure Python)
│   └── writer.py            ← saves output/lit_review.md
└── output/
    └── lit_review.md        ← generated report
```

## Error handling

- **Step 2 (S2 API)**: rate limit retries with backoff; if all searches fail, pipeline continues with empty papers list
- **Step 4 (ArXiv PDF)**: per-paper try/except; falls back to abstract silently
- **Step 1/3/5 JSON parsing**: regex fallback to extract JSON from prose-wrapped responses