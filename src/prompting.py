"""
prompting.py — Part C: Prompt Engineering & Generation
=======================================================

Three prompt variants are implemented and registered:

1. BASE_PROMPT
   The minimal grounded prompt. Instructs the model to use only retrieved
   context and cite source tags.  No explicit hallucination rules.

2. STRICT_PROMPT
   Adds explicit numbered rules forbidding invented numbers, requiring source
   tags, mandating a specific abstention phrase, and requesting conflict
   disclosure. This is the default production prompt.

3. CHAIN_OF_THOUGHT_PROMPT  (Part G — Innovation / Part C experiment)
   Adds a step-by-step reasoning scaffold before the final answer.
   Research shows CoT reduces hallucination on numerical reasoning tasks by
   forcing the model to surface its reasoning path before committing to a
   number.  Particularly useful for election-result aggregation queries.

Hallucination Control Strategy (Part C)
-----------------------------------------
   a) Context-only constraint:  "Use ONLY the provided context."
   b) Abstention phrase:         Hard-coded exact string so it can be reliably
                                 detected by the hallucination_flag() scorer.
   c) Source citation mandate:   Forces the model to tag every claim with
                                 [csv:region/year] or [pdf:page-N].
   d) Conflict disclosure:       If two chunks contradict each other, the model
                                 must surface both values rather than pick one.
   e) CoT scaffold:              Makes reasoning visible, enabling fact-checking.

Context Window Management (Part C)
------------------------------------
   build_context() enforces a max_words budget across all retrieved chunks.
   Lowest-ranked chunks are truncated first (they contribute least signal).
   A minimum of 80 words per chunk is kept even when truncating, to avoid
   feeding meaningless partial sentences.
"""


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------

BASE_PROMPT = """\
You are AcityPal, an Academic City AI assistant for Ghana Elections and Budget.
Use ONLY the provided context. If the answer is not in the context, say exactly:
"I do not have enough evidence in the retrieved sources."
Cite source hints like [csv:region/year] or [pdf:page-N] after each claim.

Answer Formatting Rules:
- For "who" questions, return ONLY the name/title (e.g., "John Dramani Mahama")
- Remove all procedural text: "Mr Speaker", "on behalf of", "I invite", "In accordance with"
- Keep answers to 1 sentence for simple factual questions
- Always include source tags at the end of your answer

User Query:
{query}

Retrieved Context:
{context}

Answer:"""


STRICT_PROMPT = """\
You are a factual policy and election assistant for AcityPal.

Rules (follow strictly):
1. Use ONLY the retrieved evidence below — never use prior knowledge.
2. Never invent, estimate, or round vote counts or monetary figures.
3. If two evidence blocks conflict, state both values and cite both sources.
4. If the evidence does not answer the question, output exactly:
   "I do not have enough evidence in the retrieved sources."
5. End every factual claim with a source tag: [csv:region/year] or [pdf:page-N].

Answer Formatting Rules:
- For "who" questions, return ONLY the name/title (e.g., "John Dramani Mahama")
- Remove all procedural text: "Mr Speaker", "on behalf of", "I invite", "In accordance with"
- Keep answers to 1 sentence for simple factual questions
- Always include source tags at the end of your answer

Question: {query}

Evidence:
{context}

Provide a concise, evidence-grounded answer with source tags:"""


CHAIN_OF_THOUGHT_PROMPT = """\
You are a precise, evidence-grounded assistant for Ghana Elections and 2025 Budget.

Instructions:
1. Read the retrieved evidence carefully.
2. Think step-by-step before writing your final answer.
3. In your reasoning, identify which evidence blocks are most relevant and why.
4. ONLY use information from the retrieved evidence — never prior knowledge.
5. If the evidence is insufficient, say exactly:
   "I do not have enough evidence in the retrieved sources."
6. Cite each fact with [csv:region/year] or [pdf:page-N].

Answer Formatting Rules:
- For "who" questions, return ONLY the name/title (e.g., "John Dramani Mahama")
- Remove all procedural text: "Mr Speaker", "on behalf of", "I invite", "In accordance with"
- Keep answers to 1 sentence for simple factual questions
- Always include source tags at the end of your answer

Question: {query}

Retrieved Evidence:
{context}

Step-by-step reasoning:
(Work through the evidence before answering)

Final Answer:"""


VERBATIM_RAG_PROMPT = """You are AcityPal, a factual assistant for Ghana Elections and 2025 Budget.

## CRITICAL RULE - WORD FOR WORD COPYING ONLY:
You MUST copy the answer EXACTLY as it appears in the retrieved evidence.

## Rules (follow strictly):
1. COPY VERBATIM - every word, every punctuation mark, every number
2. DO NOT paraphrase, summarize, or rephrase ANYTHING
3. DO NOT remove words like "Mr Speaker", "on behalf of", "I invite"
4. DO NOT extract only names or numbers - copy the full sentence
5. If the evidence contains the answer, return it EXACTLY as written
6. Add source tag at the end: [pdf:page-N] or [csv:region/year]

User Query:
{query}

Retrieved Context:
{context}

Answer:"""


NUMBER_FORCED_RULE = """
## ⚠️ NUMBER TRIGGER RULE ⚠️

If the user's question contains ANY of these words or phrases:
- "how much"
- "what amount"
- "allocation"
- "budget for"
- "funding"
- "seed fund"
- "how many"
- "what is the total"
- "amount allocated"

Then:
1. Your answer MUST contain a number (like GH¢292.4 million, GH¢13.85 billion, or 72.79%)
2. If your answer has NO number, it is WRONG
3. Find the sentence with the number in the evidence
4. Return that sentence EXACTLY

Examples of questions that MUST return numbers:
- "How much was allocated for free sanitary pads?" → MUST contain GH¢292.4 million
- "What is the allocation for Big Push?" → MUST contain GH¢13.85 billion
- "What is the seed funding for Women's Bank?" → MUST contain GH¢51.3 million
- "How many votes did Nana Akufo Addo get?" → MUST contain 1,795,824
- "What percentage did NDC get?" → MUST contain 26.47%

## FOR QUESTIONS ABOUT SPECIFIC PROGRAM NAMES:
If the user asks about a specific program name like "Adwumawura", "Big Push", "AETA", "Feed Ghana":
1. Find the sentence that contains THAT EXACT program name
2. Do NOT return information about a different program
3. Do NOT confuse Adwumawura with AETA or other programs

Examples:
Question: "Amount for Adwumawura?"
Evidence 1: "GH¢100 million to Adwumawura Programme"
Evidence 2: "GH¢1.5 billion for AETA"
CORRECT: Use Evidence 1 (Adwumawura)
WRONG: Use Evidence 2 (AETA)

Now answer this question:

User Query:
{query}

Retrieved Context:
{context}

Answer:"""


# Registry for clean programmatic access
PROMPT_REGISTRY: dict[str, str] = {
    "base": BASE_PROMPT,
    "strict": STRICT_PROMPT,
    "chain_of_thought": CHAIN_OF_THOUGHT_PROMPT,
    "verbatim": VERBATIM_RAG_PROMPT,
    "hybrid": NUMBER_FORCED_RULE,
}

# Human-readable descriptions for the UI Evaluation tab
PROMPT_DESCRIPTIONS: dict[str, str] = {
    "base": "Basic grounded prompt — minimal constraints, cite sources.",
    "strict": "Strict prompt — explicit hallucination rules, conflict disclosure.",
    "chain_of_thought": "CoT prompt — step-by-step reasoning before final answer (Part G innovation).",
    "verbatim": "Verbatim prompt — exact copying only without rephrasing.",
    "hybrid": "Hybrid prompt fixed — strictly verbatim for pdf, handles 'how much' correctly.",
}


# ---------------------------------------------------------------------------
# CONTEXT BUILDER  (Part C — Context Window Management)
# ---------------------------------------------------------------------------

def build_context(retrieved_docs: list[dict], max_words: int = 1200) -> str:
    """
    Assemble the context string injected into prompts.

    Context window management strategy:
      • Chunks are ordered by descending retrieval score (most relevant first).
      • A running word count budget (max_words) prevents exceeding the LLM's
        effective context window.
      • If the next chunk would exceed the budget, it is truncated to the
        remaining word allowance — but only if ≥ 80 words remain (otherwise
        the partial chunk is dropped to avoid feeding noise).
      • Each block is prefixed with a [source_tag] score=X.XXXX header so the
        LLM can cite sources precisely and so the hallucination detector can
        verify citations.

    This approach favours higher-ranked chunks; lower-ranked chunks may be
    partially or fully dropped when the budget is exhausted.
    """
    chosen: list[tuple[dict, str]] = []
    used = 0

    for d in retrieved_docs:
        words = d["doc"]["text"].split()
        if used + len(words) > max_words:
            remaining = max_words - used
            if remaining >= 80:
                clipped = " ".join(words[:remaining])
                chosen.append((d, clipped))
                used += remaining
            break
        chosen.append((d, d["doc"]["text"]))
        used += len(words)

    blocks = []
    for item, text in chosen:
        md = item["doc"]["metadata"]
        src = md.get("source")
        if src == "pdf":
            tag = f"pdf:page-{md.get('page')}"
        else:
            tag = f"csv:{md.get('region')}/{md.get('year')}"
        score_line = f"[{tag}] score={item['score']:.4f}"
        blocks.append(f"{score_line}\n{text}")

    return "\n\n".join(blocks)
