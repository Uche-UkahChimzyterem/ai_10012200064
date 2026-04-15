import json
import re
import pandas as pd
from pypdf import PdfReader
from src.config import CSV_PATH, PDF_PATH, OUTPUT_DIR, CLEANED_CSV, CHUNKS_A, CHUNKS_B, CHUNK_COMPARE
from src.utils import ensure_dirs, write_jsonl


def clean_csv():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df["New Region"] = df["New Region"].astype(str).str.replace("?", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0).astype(int)
    df["VotesPct"] = df["Votes(%)"].astype(str).str.replace("%", "", regex=False).str.strip()
    df["VotesPct"] = pd.to_numeric(df["VotesPct"], errors="coerce").fillna(0.0)
    df = df.drop_duplicates()
    df = df.sort_values(["Year", "New Region", "Votes"], ascending=[False, True, False]).reset_index(drop=True)
    df.to_csv(CLEANED_CSV, index=False)
    return df


def pdf_to_text():
    reader = PdfReader(str(PDF_PATH))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def csv_to_docs(df):
    docs = []
    grouped = df.groupby(["Year", "New Region"], dropna=False)
    for (year, region), g in grouped:
        lines = [f"Election results for {region} in {int(year)}:"]
        total_votes = g["Votes"].sum()
        for _, row in g.sort_values("Votes", ascending=False).iterrows():
            lines.append(f"- {row['Candidate']} ({row['Party']}): {int(row['Votes'])} votes ({row['VotesPct']:.2f}%)")
        lines.append(f"Total recorded votes in this table: {int(total_votes)}")
        docs.append({
            "source": "csv",
            "doc_id": f"csv_{int(year)}_{region.replace(' ', '_').lower()}",
            "text": "\n".join(lines),
            "metadata": {"year": int(year), "region": str(region)}
        })
    return docs


def chunk_text(text, chunk_size=700, overlap=120):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        piece = words[i:i + chunk_size]
        chunks.append(" ".join(piece))
        if i + chunk_size >= len(words):
            break
        i += max(1, chunk_size - overlap)
    return chunks


def build_chunks(docs, strategy_name, chunk_size, overlap):
    out = []
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            out.append({
                "id": f"{d['doc_id']}_ch_{idx}",
                "text": ch,
                "metadata": {**d["metadata"], "source": d["source"], "strategy": strategy_name}
            })
    return out


def evaluate_chunking_quality(chunks_a, chunks_b):
    keywords = ["fiscal", "deficit", "revenue", "Nana Akufo Addo", "John Dramani Mahama", "votes"]

    def score(chunks):
        lengths = [len(c["text"].split()) for c in chunks]
        coverage = 0
        text_all = " ".join([c["text"] for c in chunks]).lower()
        for k in keywords:
            if k.lower() in text_all:
                coverage += 1
        return {
            "num_chunks": len(chunks),
            "avg_chunk_words": sum(lengths) / max(1, len(lengths)),
            "keyword_coverage": coverage / len(keywords)
        }

    return {"strategy_a": score(chunks_a), "strategy_b": score(chunks_b)}


def main():
    ensure_dirs(OUTPUT_DIR)
    df = clean_csv()
    csv_docs = csv_to_docs(df)
    pdf_pages = pdf_to_text()
    pdf_docs = [{"source": "pdf", "doc_id": f"pdf_page_{p['page']}", "text": p["text"], "metadata": {"page": p["page"]}} for p in pdf_pages]

    docs = csv_docs + pdf_docs

    chunks_a = build_chunks(docs, "A_small", chunk_size=450, overlap=80)
    chunks_b = build_chunks(docs, "B_large", chunk_size=850, overlap=150)

    write_jsonl(CHUNKS_A, chunks_a)
    write_jsonl(CHUNKS_B, chunks_b)

    compare = evaluate_chunking_quality(chunks_a, chunks_b)
    with open(CHUNK_COMPARE, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2)

    print("Data prep complete")
    print(f"Chunks A: {len(chunks_a)} | Chunks B: {len(chunks_b)}")


if __name__ == "__main__":
    main()
