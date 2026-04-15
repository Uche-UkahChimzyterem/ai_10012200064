import pickle
import re
import numpy as np
from src.config import VECTOR_INDEX_PATH, PIPELINE_LOG_PATH, LOG_DIR
from src.embedding import EmbeddingPipeline
from src.retrieval import CustomVectorStore, HybridRetriever
from src.prompting import BASE_PROMPT, STRICT_PROMPT, build_context
from src.llm import generate_answer
from src.utils import log_jsonl, ensure_dirs


class RAGPipeline:
    def __init__(self):
        ensure_dirs(LOG_DIR)
        with open(VECTOR_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        self.docs = payload["docs"]
        self.vectors = payload["vectors"]

        self.embedder = EmbeddingPipeline()
        self.store = CustomVectorStore()
        self.store.add(self.vectors, self.docs)
        self.hybrid = HybridRetriever(self.docs)

    def retrieve(self, query, top_k=5, use_hybrid=True):
        retrieval_query = query
        ql = query.lower()
        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        target_year = int(year_match.group(0)) if year_match else None
        election_intent = any(t in ql for t in ["election", "presidential", "votes", "candidate", "won"])
        if "revenue mobilization" in ql or "revenue mobilisation" in ql:
            retrieval_query = (
                f"{query} 2025 Revenue Measures domestic revenue mobilisation "
                "tax policy measures collection efficiency"
            )
        qv = self.embedder.embed_query(retrieval_query)

        # Strong precision rule for election + explicit year queries.
        if election_intent and target_year is not None:
            year_indices = []
            for i, d in enumerate(self.docs):
                md = d.get("metadata", {})
                if md.get("source") == "csv" and md.get("year") == target_year:
                    year_indices.append(i)
            if year_indices:
                vecs = np.array([self.vectors[i] for i in year_indices])
                denom = (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv)) + 1e-9
                scores = (vecs @ qv) / denom
                order = np.argsort(scores)[::-1][:top_k]
                return [{"doc": self.docs[year_indices[j]], "score": float(scores[j])} for j in order]

        if use_hybrid:
            candidate_pool = max(top_k * 4, 20)
            vector_results = self.store.search(qv, top_k=candidate_pool)
            reranked = self.hybrid.rerank(vector_results, query=retrieval_query)

            # Precision fix: for election queries with explicit year, prioritize matching year chunks.
            if election_intent and target_year is not None:
                same_year = []
                other = []
                for r in reranked:
                    md_year = r["doc"].get("metadata", {}).get("year")
                    if md_year == target_year:
                        r["score"] += 0.08
                        same_year.append(r)
                    else:
                        other.append(r)
                reranked = sorted(same_year, key=lambda x: x["score"], reverse=True) + other

            return reranked[:top_k]
        return self.store.search(qv, top_k=top_k)

    def answer(self, query, top_k=5, use_hybrid=True, prompt_variant="strict"):
        retrieved = self.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)
        context = build_context(retrieved)
        template = STRICT_PROMPT if prompt_variant == "strict" else BASE_PROMPT
        prompt = template.format(query=query, context=context)
        response = generate_answer(prompt, query=query, context=context)

        log_jsonl(PIPELINE_LOG_PATH, {
            "stage": "full_pipeline",
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "retrieved": [{"id": r["doc"]["id"], "score": r["score"], "metadata": r["doc"]["metadata"]} for r in retrieved],
            "final_prompt": prompt,
            "response": response,
        })

        return {
            "response": response,
            "retrieved_docs": [{"text": r["doc"]["text"], "metadata": r["doc"]["metadata"], "score": r["score"]} for r in retrieved],
            "final_prompt": prompt,
        }
