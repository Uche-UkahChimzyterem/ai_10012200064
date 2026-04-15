import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class CustomVectorStore:
    def __init__(self):
        self.embeddings = None
        self.docs = None

    def add(self, embeddings, docs):
        self.embeddings = np.array(embeddings)
        self.docs = docs

    def search(self, query_vector, top_k=5):
        query_vector = np.array(query_vector)
        scores = (self.embeddings @ query_vector) / ((np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vector)) + 1e-9)
        idx = np.argsort(scores)[::-1][:top_k]
        return [{"doc": self.docs[i], "score": float(scores[i])} for i in idx]


class HybridRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.index_by_id = {d["id"]: i for i, d in enumerate(docs)}
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_texts = [d["text"] for d in docs]
        self.tfidf = self.vectorizer.fit_transform(self.doc_texts)

    def keyword_scores(self, query):
        q = self.vectorizer.transform([query])
        return (self.tfidf @ q.T).toarray().ravel()

    def rerank(self, vector_results, query, alpha=0.75):
        kw_scores = self.keyword_scores(query)
        q = query.lower()
        rescored = []
        for item in vector_results:
            doc = item["doc"]
            base = item["score"]
            idx = self.index_by_id[doc["id"]]
            kscore = float(kw_scores[idx])
            blended = alpha * base + (1 - alpha) * kscore

            # Innovation: domain-specific scoring boost for mixed-source corpus.
            md = doc.get("metadata", {})
            source = md.get("source", "")
            boost = 0.0
            if any(t in q for t in ["budget", "fiscal", "deficit", "revenue", "policy"]) and source == "pdf":
                boost += 0.05
            if any(t in q for t in ["votes", "region", "candidate", "election", "ndc", "npp"]) and source == "csv":
                boost += 0.05
            if "year" in md and str(md["year"]) in q:
                boost += 0.03

            score = blended + boost
            rescored.append(
                {
                    "doc": doc,
                    "score": score,
                    "vector_score": base,
                    "keyword_score": kscore,
                    "domain_boost": boost,
                }
            )
        rescored.sort(key=lambda x: x["score"], reverse=True)
        return rescored
