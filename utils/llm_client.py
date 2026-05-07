"""
utils/llm_client.py
Single reusable LLM call function. All steps call this — never the API directly.

Supports:
    - Grok (xAI) via OpenAI-compatible SDK  [default — required by assignment]
    - Gemini (Google) via google-generativeai [optional swap]
    - Mock (for testing without any API key)

Retry behaviour:
    On 429 (rate limit), reads the suggested retry_delay from the error message,
    waits that long, then retries. After MAX_RETRIES attempts, raises RuntimeError.
"""

import os
import re
import time

MAX_RETRIES = 4    # retry attempts on rate-limit errors
BASE_WAIT   = 5    # fallback wait seconds if API doesn't specify


# ── Provider resolution ────────────────────────────────────────────────────────
# app.py sets os.environ["LLM_PROVIDER"] at runtime.
# config.py is the fallback for CLI usage.

def _get_provider() -> str:
    env_val = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_val:
        return env_val
    try:
        from config import LLM_PROVIDER
        return LLM_PROVIDER.strip().lower()
    except ImportError:
        return "mock"


def _get_config(key: str, default: str = "") -> str:
    """Read a value from config.py, falling back to an env var, then default."""
    try:
        import config
        return str(getattr(config, key, default)) or os.environ.get(key, default)
    except ImportError:
        return os.environ.get(key, default)


# ── Public entry point ─────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_prompt: str,
    step_label: str = "LLM call",
    max_tokens: int = 2048,
) -> str:
    """
    Make a single LLM call with a system prompt and user prompt.

    Args:
        system_prompt: the role / instructions for the model
        user_prompt:   the actual task content for this call
        step_label:    human-readable label for logging/debugging
        max_tokens:    maximum tokens in the response (Step 6 passes 4096)

    Returns:
        str: the raw text response from the LLM
    """
    provider = _get_provider()
    print(f"    [LLM] {step_label} — calling {provider.upper()}…")

    try:
        if provider == "grok":
            return _call_grok(system_prompt, user_prompt, max_tokens, step_label)
        elif provider == "gemini":
            return _call_gemini(system_prompt, user_prompt, max_tokens, step_label)
        elif provider == "mock":
            return _call_mock(system_prompt, user_prompt, step_label)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'")

    except Exception:
        raise


# ── Grok (xAI) ─────────────────────────────────────────────────────────────────
# Grok uses the OpenAI SDK but pointed at api.x.ai — not an OpenAI account.

def _call_grok(system_prompt: str, user_prompt: str, max_tokens: int, label: str) -> str:
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    api_key = _get_config("GROK_API_KEY") or os.environ.get("GROK_API_KEY", "")
    model   = _get_config("GROK_MODEL", "grok-beta")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            text = response.choices[0].message.content or ""
            print(f"    [LLM] {label} — received {len(text)} chars")
            return text

        except RateLimitError as e:
            wait = _parse_retry_delay(str(e)) or (BASE_WAIT * attempt)
            wait = min(wait, 60)
            print(f"    [LLM] {label} — Grok 429, waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})…")
            if attempt < MAX_RETRIES:
                time.sleep(wait)

        except Exception as e:
            raise RuntimeError(f"Grok API call failed at '{label}': {e}") from e

    raise RuntimeError(f"Grok rate limit persisted after {MAX_RETRIES} retries at '{label}'")


# ── Gemini (Google) ─────────────────────────────────────────────────────────────

def _call_gemini(system_prompt: str, user_prompt: str, max_tokens: int, label: str) -> str:
    try:
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted
    except ImportError:
        raise ImportError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    api_key = _get_config("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    model   = _get_config("GEMINI_MODEL", "gemini-2.5-flash")
    label_lower = label.lower()
    wants_json = not ("step 6" in label_lower or "report" in label_lower or "lit_review" in label_lower)

    genai.configure(api_key=api_key)

    generation_config = {"max_output_tokens": max_tokens, "temperature": 0.3}

    genai_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_prompt,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = genai_model.generate_content(user_prompt)
            text = response.text or ""
            try:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason is not None:
                    print(f"    [LLM] {label} — finish_reason={finish_reason}")
            except Exception:
                pass
            print(f"    [LLM] {label} — received {len(text)} chars")
            return text

        except ResourceExhausted as e:
            wait = _parse_retry_delay(str(e)) or (BASE_WAIT * 2 * attempt)
            wait = min(wait, 60)
            print(f"    [LLM] {label} — Gemini 429, waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRIES})…")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
            else:
                # Persist full exception for debugging
                try:
                    import os, datetime, traceback
                    os.makedirs("logs", exist_ok=True)
                    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    fname = os.path.join("logs", f"llm_gemini_resource_exhausted_{ts}.txt")
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write("ResourceExhausted:\n")
                        f.write(str(e))
                        f.write("\n\nTraceback:\n")
                        f.write(traceback.format_exc())
                    print(f"    [DEBUG] Gemini error details written to {fname}")
                except Exception:
                    pass

                raise RuntimeError(
                    f"Gemini rate limit persisted after {MAX_RETRIES} retries at '{label}'"
                ) from e

        except Exception as e:
            # Save unexpected Gemini exception for debugging
            try:
                import os, datetime, traceback
                os.makedirs("logs", exist_ok=True)
                ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                fname = os.path.join("logs", f"llm_gemini_exception_{ts}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write("Exception:\n")
                    f.write(str(e))
                    f.write("\n\nTraceback:\n")
                    f.write(traceback.format_exc())
                print(f"    [DEBUG] Gemini exception written to {fname}")
            except Exception:
                pass

            raise RuntimeError(f"Gemini API call failed at '{label}': {e}") from e

    raise RuntimeError(f"Gemini exhausted retries at '{label}'")


# ── Mock (no API key needed — for testing and demo) ────────────────────────────

_MOCK_RESPONSES: dict[str, str] = {
    "step1": """{
  "topic": "Retrieval-Augmented Generation for Question Answering",
  "field": "Computer Science / Natural Language Processing",
  "time_range": "2020-2024",
  "keywords": ["retrieval-augmented generation", "RAG", "question answering", "dense retrieval", "knowledge grounding"],
  "subtopics": ["open-domain QA", "knowledge-intensive NLP", "dense passage retrieval", "generative QA"],
  "search_strings": ["retrieval augmented generation", "question answering", "rag question answering"]
}""",
    "step3": """{
  "top_paper_ids": [],
  "selection_reasoning": "Selected the highest-ranked candidate papers that best match the topic and subtopics.",
  "notable_researchers": [
    {"name": "Patrick Lewis", "h_index": 22, "affiliation": "Meta AI", "contribution": "Introduced the RAG framework combining retrieval with seq2seq generation."},
    {"name": "Danqi Chen", "h_index": 35, "affiliation": "Princeton University", "contribution": "Pioneered dense passage retrieval (DPR) for open-domain QA."},
    {"name": "Vladimir Karpukhin", "h_index": 18, "affiliation": "Facebook AI", "contribution": "Developed DPR, enabling scalable dense retrieval from Wikipedia."}
  ]
}""",
    "step4": """{
  "core_argument": "This paper argues that augmenting language models with retrieved passages significantly improves factual accuracy in question answering tasks compared to closed-book approaches.",
  "methodology": "The authors evaluate on open-domain QA benchmarks (NaturalQuestions, TriviaQA, WebQuestions) using a retriever-reader pipeline. The retriever uses dense embeddings (DPR); the reader is a seq2seq model fine-tuned on retrieved passages.",
  "key_findings": [
    "Achieves state-of-the-art on NaturalQuestions with 44.5 Exact Match, surpassing prior work by 4 EM points.",
    "Shows dense retrieval outperforms BM25 by 9 EM points on open-domain benchmarks.",
    "Demonstrates that top-100 retrieved passages improve generation over top-5.",
    "Finds end-to-end fine-tuning of retriever and reader outperforms pipeline approaches."
  ],
  "limitations": [
    "Evaluation limited to English-language Wikipedia; cross-lingual and multilingual settings not addressed.",
    "Inference latency of retrieval step not benchmarked; production viability unclear."
  ],
  "inner_citations": [
    "Karpukhin et al. (2020) — Dense Passage Retrieval for Open-Domain QA",
    "Lewis et al. (2020) — RAG",
    "Izacard & Grave (2021) — Leveraging Passage Retrieval with Generative Models (FiD)"
  ]
}""",
    "step5": """{
  "agreements": [
    "All papers agree that retrieval-augmented approaches outperform closed-book LLMs on knowledge-intensive QA benchmarks (Lewis et al., 2020; Karpukhin et al., 2020; Izacard & Grave, 2021).",
    "Dense retrieval (DPR-style) consistently outperforms sparse BM25 retrieval across all evaluated papers.",
    "Larger retrieved context — more passages fed to the reader — generally improves generation quality up to a saturation point."
  ],
  "contradictions": [
    "Lewis et al. (RAG) find end-to-end retriever+reader training essential, while Izacard & Grave (FiD) show a frozen retriever with a stronger reader is competitive at lower training cost.",
    "Karpukhin et al. argue retriever fine-tuning is critical for peak performance, whereas several follow-up papers find off-the-shelf retrievers sufficient when paired with more powerful readers."
  ],
  "gaps": [
    "Multilingual and cross-lingual RAG for question answering is understudied across all reviewed papers.",
    "Inference latency and retrieval cost at production scale are not systematically benchmarked.",
    "Robustness to noisy, outdated, or adversarially retrieved passages remains an open research question.",
    "Long-context multi-hop QA requiring retrieval across multiple documents is insufficiently addressed."
  ],
  "clusters": [
    "Dense Retrieval Methods",
    "Generative Reader Architectures",
    "Benchmarks and Evaluation",
    "Limitations and Open Challenges"
  ]
}""",
    "step6": """# Retrieval-Augmented Generation for Question Answering: A Literature Review

**Field:** Computer Science / Natural Language Processing | **Period:** 2020–2024 | **Papers Reviewed:** 5

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a leading paradigm for knowledge-intensive question answering, combining the parametric knowledge of large language models with non-parametric retrieval from external corpora. This review synthesises five key works spanning dense retrieval, generative reading, and evaluation, identifying consensus findings, active methodological debates, and open research gaps.

---

## 1. Introduction

Open-domain question answering requires models to access world knowledge not reliably encoded in model parameters. Retrieval-augmented approaches address this by fetching relevant passages at inference time from a fixed knowledge corpus. The seminal RAG framework [Lewis et al., 2020] established this paradigm, treating retrieval and generation as jointly learnable components. Subsequent work has refined both the retrieval and reader sides of the pipeline.

---

## 2. Dense Retrieval Methods

The transition from sparse (BM25) to dense retrieval is the central methodological contribution of this era. Dense Passage Retrieval (DPR) [Karpukhin et al., 2020] encodes queries and passages into a shared embedding space using dual BERT encoders, enabling maximum-inner-product search over Wikipedia. This approach outperforms BM25 by 9 Exact Match points on open-domain benchmarks [Karpukhin et al., 2020].

---

## 3. Generative Reader Architectures

Once relevant passages are retrieved, the reader must synthesise an answer. The RAG model [Lewis et al., 2020] uses a BART-based seq2seq reader that marginalises over retrieved documents. Fusion-in-Decoder (FiD) [Izacard & Grave, 2021] extends this by encoding each passage independently and fusing representations in the decoder, achieving superior performance with a frozen retriever.

---

## 4. Benchmarks and Evaluation

Evaluation across this literature centres on NaturalQuestions, TriviaQA, and WebQuestions, all measured by Exact Match. RAG achieves 44.5 EM on NaturalQuestions [Lewis et al., 2020]. Limitations of these benchmarks include their English-only nature and single-hop question structure, leaving multi-hop and multilingual QA understudied.

---

## 5. Notable Researchers

**Patrick Lewis** (Meta AI, h-index: 22) — Introduced the RAG framework combining retrieval with seq2seq generation, establishing the foundational paradigm reviewed here.

**Danqi Chen** (Princeton University, h-index: 35) — Pioneered dense retrieval for open-domain QA, whose influence is visible across all reviewed papers.

**Vladimir Karpukhin** (Facebook AI, h-index: 18) — Developed DPR, enabling scalable dense retrieval from Wikipedia at the scale required for open-domain QA.

---

## 6. Agreements in the Literature

All reviewed papers agree that retrieval-augmented approaches outperform closed-book LLMs on knowledge-intensive QA benchmarks [Lewis et al., 2020; Karpukhin et al., 2020; Izacard & Grave, 2021]. Dense retrieval consistently outperforms sparse BM25 retrieval. Larger retrieved context generally improves generation quality up to a saturation point.

---

## 7. Contradictions and Debates

A key debate concerns whether end-to-end retriever training is necessary. Lewis et al. [2020] find joint training essential, while Izacard & Grave [2021] demonstrate that a frozen retriever paired with a stronger reader is competitive at lower training cost. Similarly, Karpukhin et al. argue retriever fine-tuning is critical, while follow-up work suggests strong readers can compensate for weaker retrievers.

---

## 8. Open Gaps and Future Directions

Several questions remain unresolved across the reviewed literature. Multilingual and cross-lingual RAG is unstudied. Production-scale inference latency is not benchmarked. Robustness to noisy or adversarially retrieved passages remains open. Long-context multi-hop QA requiring retrieval across multiple documents is insufficiently addressed.

---

## 9. Conclusion

Retrieval-Augmented Generation represents a mature and well-validated approach to knowledge-intensive QA. The field has converged on dense retrieval as superior to sparse methods, while the debate between end-to-end and modular training remains active. Future work should prioritise multilingual evaluation, latency benchmarking, and robustness to retrieval noise.

---

## References

[1] Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
[2] Karpukhin, V., Oguz, B., Min, S., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP.
[3] Izacard, G. & Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*. EACL.
[4] Guu, K., Lee, K., Tung, Z., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training*. ICML.
[5] Shi, W. et al. (2024). *RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering*. arXiv:2407.13998.
""",
}


def _call_mock(system_prompt: str, user_prompt: str, label: str) -> str:
    """Return a realistic hardcoded response based on which step is calling."""
    label_lower = label.lower()
    if "step 1" in label_lower or "parse" in label_lower:
        key = "step1"
    elif "step 3" in label_lower or "relevance" in label_lower:
        key = "step3"
    elif "step 4" in label_lower or "deep_read" in label_lower:
        key = "step4"
    elif "step 5" in label_lower or "synth" in label_lower:
        key = "step5"
    elif "step 6" in label_lower or "report" in label_lower or "lit_review" in label_lower:
        key = "step6"
    else:
        return '{"result": "mock response", "note": "no matching mock for this step"}'

    return _MOCK_RESPONSES[key]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _parse_retry_delay(error_message: str) -> float | None:
    """
    Extract the suggested retry delay from a 429 error message string.

    Gemini errors contain:  'Please retry in 28.382s'
    Gemini errors contain:  'retry_delay { seconds: 28 }'
    """
    # "Please retry in 28.38s" or "Please retry in 2.05s"
    match = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2   # +2s buffer

    # "retry_delay { seconds: 28 }"
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_message)
    if match:
        return float(match.group(1)) + 2

    return None