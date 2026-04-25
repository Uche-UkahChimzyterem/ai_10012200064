"""
retrieval.py — Part B: Custom Retrieval System
===============================================

Implements:
  • CustomVectorStore — in-memory numpy-based vector index with cosine search
  • HybridRetriever   — blends vector similarity + TF-IDF keyword relevance
                        + domain-specific boosting (Part G innovation)

Design decisions
----------------
Vector storage:
  We use a plain numpy matrix instead of FAISS/Chroma because:
  (a) corpus size (~1 000–3 000 chunks) fits well within O(N·d) brute-force;
  (b) no binary dependency, so the system is trivially portable;
  (c) all maths are transparent and auditable for exam purposes.

Hybrid retrieval (extension — Part B):
  Pure vector search on TF-IDF/hashing embeddings misses exact keyword hits
  (e.g. a rare candidate name that falls in a collision bucket).  We add
  TF-IDF keyword scoring as a complementary signal and blend them linearly:
      final_score = α × vector_score + (1 − α) × keyword_score
  Default α = 0.75 (vector-dominant) keeps semantic intent while recovering
  exact-match precision.

Domain-specific scoring boost (Part G — Innovation):
  A lightweight heuristic inspects both the query text and each chunk's
  source metadata to apply a small score bonus:
    +0.05  if a "budget/fiscal/revenue" query hits a PDF chunk
    +0.05  if an "election/votes/region" query hits a CSV chunk
    +0.03  if the explicit year in the query matches the chunk's year metadata
  This domain boost prevents cross-source contamination (e.g. a fiscal query
  pulling in irrelevant election rows that happen to share a keyword).

Failure case (Part B — Required demonstration):
  Query: "Who invented the internet?"
  Without fix: vector search returns the highest-scoring chunks regardless of
    relevance (cosine similarity is relative, not absolute). The top results
    will be election or budget chunks with low but non-zero scores simply
    because they share stop-word-filtered tokens.
  With fix: see `get_failure_case_demo()` — the hybrid re-ranker returns very
    low scores for all docs (below the irrelevance threshold), and the prompt
    template's hallucination guard triggers "I do not have enough evidence."
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Threshold below which a retrieval result is considered "irrelevant"
RELEVANCE_THRESHOLD = 0.05


class CustomVectorStore:
    """
    In-memory cosine-similarity vector index.

    Attributes
    ----------
    embeddings : np.ndarray | None
        Shape (N, D): all document embedding vectors.
    docs : list[dict] | None
        Parallel list of chunk dicts corresponding to each row of embeddings.
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.docs: list[dict] | None = None

    def add(self, embeddings: np.ndarray, docs: list[dict]) -> None:
        """Index all document embeddings. Call once at startup."""
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.docs = docs

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Top-K cosine similarity search.

        Returns a list of dicts:
          { doc: dict, score: float }
        sorted descending by score.

        Similarity formula:
          score(q, d) = (q · d) / (‖q‖ · ‖d‖ + ε)
        """
        qv = np.array(query_vector, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(qv) + 1e-9
        scores = (self.embeddings @ qv) / norms
        idx = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.docs[i], "score": float(scores[i])} for i in idx]

    def get_failure_case_demo(
        self, query_vector: np.ndarray, top_k: int = 3
    ) -> dict:
        """
        Demonstrates a known retrieval failure case (Part B).

        For an out-of-domain query (e.g. "Who invented the internet?"), pure
        vector search still returns results because cosine similarity is
        relative, not absolute — it always finds the "closest" vectors even if
        they are irrelevant.

        Returns dict with:
          results     — the top-k results (demonstrating the failure)
          max_score   — highest similarity score observed
          is_relevant — True only if max_score >= RELEVANCE_THRESHOLD
          explanation — textual explanation for the UI
        """
        results = self.search(query_vector, top_k=top_k)
        max_score = max((r["score"] for r in results), default=0.0)
        return {
            "results": results,
            "max_score": round(max_score, 4),
            "is_relevant": max_score >= RELEVANCE_THRESHOLD,
            "threshold_used": RELEVANCE_THRESHOLD,
            "explanation": (
                "Pure vector search always returns the top-K closest docs. "
                "For an out-of-domain query the max score is {:.4f} which is {} "
                "the relevance threshold ({:.2f}). "
                "Fix: the prompt template's hallucination guard rejects responses "
                "when no source tag is present, and hybrid re-ranking depresses "
                "scores further below the threshold for truly irrelevant queries."
            ).format(
                max_score,
                "ABOVE" if max_score >= RELEVANCE_THRESHOLD else "BELOW",
                RELEVANCE_THRESHOLD,
            ),
        }


class HybridRetriever:
    """
    Hybrid keyword + vector retriever with domain-specific score boosting.

    The TF-IDF keyword model is fitted on all document texts at init time.
    At query time, keyword scores are blended with incoming vector scores.

    Parameters
    ----------
    docs : list[dict]
        All chunk dicts (must include 'id', 'text', 'metadata' keys).
    alpha : float
        Blend weight for vector score. (1-alpha) goes to keyword score.
        Default 0.75 — vector-dominant with keyword precision correction.
    """

    def __init__(self, docs: list[dict], alpha: float = 0.75):
        self.docs = docs
        self.alpha = alpha
        self.index_by_id: dict[str, int] = {d["id"]: i for i, d in enumerate(docs)}
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.doc_texts = [d["text"] for d in docs]
        self.tfidf = self.vectorizer.fit_transform(self.doc_texts)

    def keyword_scores(self, query: str) -> np.ndarray:
        """Return TF-IDF cosine scores for the query against all documents."""
        q = self.vectorizer.transform([query])
        return (self.tfidf @ q.T).toarray().ravel()

    def rerank(
        self, vector_results: list[dict], query: str, alpha: float | None = None
    ) -> list[dict]:
        """
        Blend vector scores with keyword scores and apply domain boosts.

        Each result dict is extended with:
          score         — final blended + boosted score
          vector_score  — raw cosine similarity from vector store
          keyword_score — TF-IDF cosine similarity
          domain_boost  — domain-specific bonus applied

        Domain-specific boosting logic (Part G — Innovation):
          • Budget/fiscal queries → boost PDF source chunks (+0.05)
          • Election/votes queries → boost CSV source chunks (+0.05)
          • Year-explicit queries → boost chunks with matching year (+0.03)
        """
        _alpha = alpha if alpha is not None else self.alpha
        kw_scores = self.keyword_scores(query)
        q_lower = query.lower()
        rescored = []

        for item in vector_results:
            doc = item["doc"]
            base_vec = item["score"]
            idx = self.index_by_id[doc["id"]]
            kscore = float(kw_scores[idx])
            blended = _alpha * base_vec + (1 - _alpha) * kscore

            # --- Domain-specific scoring boost (Part G) ---
            md = doc.get("metadata", {})
            source = md.get("source", "")
            boost = 0.0

            budget_terms = ["budget", "fiscal", "deficit", "revenue", "policy", "tax"]
            election_terms = ["votes", "region", "candidate", "election", "ndc", "npp",
                              "presidential", "won", "winner"]

            if any(t in q_lower for t in budget_terms) and source == "pdf":
                boost += 0.05
            if any(t in q_lower for t in election_terms) and source == "csv":
                boost += 0.05

            import re as _re
            year_m = _re.search(r"\b(19|20)\d{2}\b", query)
            if year_m and "year" in md:
                if str(md["year"]) == year_m.group(0):
                    boost += 0.03

            final_score = blended + boost
            rescored.append({
                "doc": doc,
                "score": final_score,
                "vector_score": base_vec,
                "keyword_score": kscore,
                "domain_boost": boost,
            })

        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored
