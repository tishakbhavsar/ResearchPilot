"""
prompts/step5_prompt.py
System and user prompt for Step 5 — synthesise_critique (LLM).

PURPOSE OF THIS STEP:
    Step 4 reads each paper in isolation — deep focus, one at a time.
    Step 5 is the first and only step that sees ALL 5 analyses simultaneously.
    This is where cross-paper patterns emerge: which findings replicate, which
    contradict, which questions no paper has answered.

    Why can't Step 6 do the synthesis while writing?
    Step 6's job is to write coherent prose. If Step 6 also had to synthesise,
    it would produce a paper-by-paper structure ("Paper 1 says X, Paper 2 says Y")
    rather than a genuine thematic review. Pre-computed synthesis forces Step 6
    to write thematically, not sequentially.

    Why can't Step 4 do cross-paper synthesis?
    Step 4 runs per-paper with individual context. It cannot compare across papers
    it hasn't read yet. Synthesis requires all papers to have been read first.

PROMPT DESIGN RATIONALE:
    1. Each contradiction must name specific papers ("Lewis et al. (RAG) claims X
       while Izacard & Grave (FiD) finds Y"). Vague contradictions like "some papers
       disagree" are useless to Step 6 for generating citations.
    2. clusters are the most important output — they become the section headers of
       the final report. They must be short (2–5 words), title-cased, and emerge
       from the methodologies rather than copying the subtopics verbatim.
    3. gaps must be formulated as open research questions, not just restatements
       of paper limitations. "Paper A only tested on English" → gap: "Cross-lingual
       RAG is understudied."

PROMPT ITERATION (v1 → v2):
    v1 prompt:
        Asked for agreements, contradictions, gaps in free prose. No JSON.

    v1 problem:
        Step 6 received one large text blob. It had no way to distinguish
        agreements from contradictions programmatically. The report prompt
        became unwieldy and the LLM mixed up the categories.

    v2 change (current):
        Enforced JSON with four explicit arrays. Contradiction entries MUST name
        two specific papers by short title. This surfaces in Step 6 as citable
        claims with automatic in-text citation targets.

        Also added clusters explicitly — v1 left section headers to Step 6,
        which then invented arbitrary headers that didn't reflect the literature.
"""

SYSTEM_PROMPT = """Synthesise 5 papers into JSON with: agreements, contradictions, gaps, clusters.
Return ONLY valid JSON. Cite specific papers in each claim.
Schema: {"agreements": [...5-strings], "contradictions": [...2-4 strings], "gaps": [...3-5], "clusters": [...3-5]}"""

USER_PROMPT_TEMPLATE = """Topic: {topic}

--- ALL 5 PAPERS ---
{deep_reads_block}

Return JSON synthesis."""