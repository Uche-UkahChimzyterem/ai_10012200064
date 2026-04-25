import os
import re
from dotenv import load_dotenv
from src.config import LOCAL_MODE

load_dotenv()


def simple_grounded_generator(query, context):
    if not context.strip():
        return "I do not have enough evidence in the retrieved sources."

    blocks = []
    for raw in context.split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        m = re.match(r"^\[(?P<tag>[^\]]+)\]\s+score=.*?\n(?P<body>.*)$", raw, flags=re.DOTALL)
        if m:
            blocks.append({"tag": m.group("tag"), "body": m.group("body").strip()})

    if not blocks:
        return "I do not have enough evidence in the retrieved sources."

    query_l = query.lower()
    query_terms = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 3]
    year_match = re.search(r"\b(19|20)\d{2}\b", query)
    target_year = int(year_match.group(0)) if year_match else None

    # Structured aggregation for broad election-result queries.
    if (
        target_year is not None
        and "election" in query_l
        and ("results" in query_l or "result" in query_l)
    ):
        totals = {}
        used_tags = []
        matched_blocks = 0
        for b in blocks:
            tag_m = re.match(r"^csv:(.*)/(\d{4})$", b["tag"])
            if not tag_m:
                continue
            year = int(tag_m.group(2))
            if year != target_year:
                continue
            matched_blocks += 1
            if b["tag"] not in used_tags:
                used_tags.append(b["tag"])
            candidates = re.findall(
                r"-\s*([^(]+)\(([^)]+)\):\s*([0-9,]+)\s*votes",
                b["body"],
            )
            for name, party, votes in candidates:
                key = (name.strip(), party.strip())
                try:
                    vote_int = int(votes.replace(",", ""))
                except ValueError:
                    continue
                totals[key] = totals.get(key, 0) + vote_int

        if matched_blocks >= 2 and len(totals) >= 2:
            ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
            lines = []
            for (name, party), votes in ranked[:3]:
                lines.append(f"{name} ({party}): {votes:,} votes")
            return (
                f"For {target_year}, across retrieved regions, the leading presidential totals are: "
                + "; ".join(lines)
                + f". Sources: {', '.join(f'[{t}]' for t in used_tags[:5])}"
            )

    # Structured extraction for election winner-style questions.
    if any(t in query_l for t in ["won", "winner", "more votes", "highest votes"]):
        best_block = None
        best_score = -1
        for b in blocks:
            b_score = sum(1 for t in query_terms if t in b["body"].lower() or t in b["tag"].lower())
            if b_score > best_score:
                best_score = b_score
                best_block = b
        if best_block:
            candidates = re.findall(
                r"-\s*([^(]+)\(([^)]+)\):\s*([0-9,]+)\s*votes",
                best_block["body"],
            )
            parsed = []
            for name, party, votes in candidates:
                try:
                    parsed.append((name.strip(), party.strip(), int(votes.replace(",", ""))))
                except ValueError:
                    continue
            if len(parsed) >= 2:
                parsed.sort(key=lambda x: x[2], reverse=True)
                top, second = parsed[0], parsed[1]
                return (
                    f"{top[0]} ({top[1]}) recorded the highest votes ({top[2]:,}), "
                    f"ahead of {second[0]} ({second[1]}) with {second[2]:,}. "
                    f"Sources: [{best_block['tag']}]"
                )

    # Structured extraction for revenue-mobilization style policy questions.
    if "revenue mobilization" in query_l or "revenue mobilisation" in query_l:
        for b in blocks:
            body = re.sub(r"\s+", " ", b["body"])
            match = re.search(
                r"(Government is proposing some revenue measures[^.]*\.)|"
                r"([^.]*improve domestic revenue mobili[sz]ation[^.]*\.)|"
                r"([^.]*commitment to revenue mobili[sz]ation[^.]*\.)",
                body,
                flags=re.IGNORECASE,
            )
            if match:
                snippet = next((g for g in match.groups() if g), "").strip()
                snippet = re.sub(r"\s+", " ", snippet)[:220].rstrip(" ,;:")
                if snippet:
                    if not snippet.endswith("."):
                        snippet += "."
                    return f"{snippet} Sources: [{b['tag']}]"
                    
    # Structured extraction for "How much" questions.
    how_much_terms = ["how much", "what amount", "allocation", "budget for", "funding", "seed fund", "how many", "what is the total", "amount allocated", "ghc", "cost"]
    if any(term in query_l for term in how_much_terms):
        amt_groups = []
        for b in blocks:
            normalized = b["body"].replace("\n", " ").replace(" - ", ". ").replace(";", ". ")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
            for i, sent in enumerate(sentences):
                score = 0
                for t in query_terms:
                    if t in sent.lower():
                        score += 15 if len(t) > 4 else 1 
                
                # If this sentence has keywords, check IT and the NEXT sentence for amounts
                has_amount = re.search(r"(GH¢|GHA|US\$|\$|million|billion|percent|percentage|\d+%)", sent, flags=re.IGNORECASE)
                next_has_amount = False
                if i + 1 < len(sentences):
                    next_has_amount = re.search(r"(GH¢|GHA|US\$|\$|million|billion|percent|percentage|\d+%)", sentences[i+1], flags=re.IGNORECASE)
                
                if score > 0:
                    if has_amount:
                        amt_groups.append((score, [sent], b["tag"]))
                    elif next_has_amount:
                        # Link keywords in this sentence to amount in next sentence
                        amt_groups.append((score, [sent, sentences[i+1]], b["tag"]))
        
        if amt_groups:
            amt_groups.sort(key=lambda x: x[0], reverse=True)
            best_score, best_sents, best_tag = amt_groups[0]
            
            # Combine sentences
            combined = " ".join(s if s.endswith((".", "!", "?")) else f"{s}." for s in best_sents)
            combined = combined.replace("GHA", "GH¢")
            return f"{combined} Sources: [{best_tag}]"
            
    scored_snippets = []

    for b in blocks:
        normalized = (
            b["body"]
            .replace("\n", " ")
            .replace(" - ", ". ")
            .replace(";", ". ")
        )
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", normalized)
            if s.strip()
        ]
        for sent in sentences[:15]: # Checked more sentences
            score = sum(1 for t in query_terms if t in sent.lower())
            if score == 0:
                continue
            if "table of contents" in sent.lower():
                continue
            if any(x in sent.lower() for x in ["table ", "figure ", "appendix "]):
                score -= 1
            if "mobilization" in query_l or "mobilisation" in query_l:
                if "mobilization" in sent.lower() or "mobilisation" in sent.lower():
                    score += 2
                if "revenue measures" in sent.lower():
                    score += 2
            if score <= 0:
                continue
            scored_snippets.append((score, sent, b["tag"]))

    if not scored_snippets:
        return "I do not have enough evidence in the retrieved sources."

    scored_snippets.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    used_tags = []
    seen = set()
    for _, sent, tag in scored_snippets:
        compact = re.sub(r"\s+", " ", sent).strip()
        key = compact.lower()
        if len(compact) < 30 or key in seen:
            continue
        seen.add(key)
        
        if len(compact) > 400: # Slightly increased limit
            compact = compact[:400].rstrip(" ,;:") + "..."
            
        chosen.append(compact)
        if tag not in used_tags:
            used_tags.append(tag)
        if len(chosen) >= 3: # Return up to 3 sentences for better context
            break

    if not chosen:
        return "I do not have enough evidence in the retrieved sources."

    answer = " ".join(
        s if s.endswith((".", "!", "?")) else f"{s}."
        for s in chosen
    )
    citations = ", ".join(f"[{t}]" for t in used_tags[:3])
    return f"{answer} Sources: {citations}"


def generate_without_retrieval(query):
    """
    Baseline for Part E comparison: pure model call without retrieved context.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "I do not have enough evidence in the retrieved sources."

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the user question directly."},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def generate_answer(prompt, query, context):
    # Use local mode if configured or if no API key is available
    if LOCAL_MODE:
        return simple_grounded_generator(query, context)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return simple_grounded_generator(query, context)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a cautious, evidence-grounded assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content
