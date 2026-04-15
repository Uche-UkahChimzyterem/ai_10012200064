BASE_PROMPT = """You are Academic City AI assistant.
Use ONLY the provided context.
If the answer is not in context, say: 'I do not have enough evidence in the retrieved sources.'
Cite concise source hints like [csv:region/year] or [pdf:page].

User Query:
{query}

Retrieved Context:
{context}

Answer:
"""

STRICT_PROMPT = """You are a factual policy and election assistant.
Rules:
1) Use only retrieved evidence.
2) Never invent numbers.
3) If evidence conflicts, state both values and sources.
4) If no evidence, output exactly: 'I do not have enough evidence in the retrieved sources.'

Question: {query}

Evidence:
{context}

Provide a concise answer with source tags.
"""


def build_context(retrieved_docs, max_words=1200):
    chosen = []
    used = 0
    for d in retrieved_docs:
        words = d["doc"]["text"].split()
        if used + len(words) > max_words:
            remaining = max_words - used
            if remaining > 80:
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
        tag = f"pdf:page-{md.get('page')}" if src == "pdf" else f"csv:{md.get('region')}/{md.get('year')}"
        blocks.append(f"[{tag}] score={item['score']:.4f}\n{text}")
    return "\n\n".join(blocks)
