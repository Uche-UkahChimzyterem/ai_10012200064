"""
embedding.py — Part B: Custom Embedding Pipeline
=================================================

Design Rationale
----------------
We deliberately avoid external API calls for embedding so that the system
works fully offline and without incurring per-token costs.

Approach: TF-IDF-style n-gram hashing + L2 normalisation
  • HashingVectorizer converts text to a fixed-length sparse vector using the
    hashing trick — no vocabulary fitting required, so new queries are
    embedded identically at inference time.
  • Bigram range (1,2) captures adjacent word pairs ("fiscal deficit",
    "John Mahama") which are critical for election/budget domain precision.
  • L2 normalisation ensures all embedding vectors lie on the unit hypersphere,
    making cosine similarity equivalent to a simple dot product — O(d) lookup.
  • n_features=4096 gives enough hash buckets to limit collision probability
    while staying memory-efficient (4096 × float32 = 16 KB per doc).

This is a deliberate trade-off:
  Pro  — deterministic, offline, zero latency, interpretable
  Con  — no semantic generalisation (synonyms not linked)
       — can be replaced with sentence-transformers for a richer but heavier
         production system
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class EmbeddingPipeline:
    """
    Lightweight, offline embedding pipeline for the AcityPal RAG system.

    Uses bigram hashing + L2 normalisation so cosine similarity reduces to
    a fast dot product.  Deterministic: same text always yields same vector.

    Attributes
    ----------
    n_features : int
        Number of hash buckets (dimensionality of output vectors).
    vectorizer : HashingVectorizer
        Sklearn vectorizer configured for bigram, English stop-word removal.
    """

    def __init__(self, n_features: int = 4096):
        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,   # all positive counts
            norm=None,              # we normalise manually for transparency
            ngram_range=(1, 2),     # unigrams + bigrams
            stop_words="english",
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document strings.

        Returns an (N, n_features) float32 array with each row on the unit
        hypersphere (L2 norm = 1).
        """
        mat = self.vectorizer.transform(texts).astype(np.float32)
        return normalize(mat, norm="l2", axis=1).toarray()

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns a (n_features,) float32 vector on the unit hypersphere.
        """
        mat = self.vectorizer.transform([query]).astype(np.float32)
        return normalize(mat, norm="l2", axis=1).toarray()[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Convenience wrapper: returns a Python list of 1-D numpy arrays.
        Useful when individual vectors are needed (e.g. for iterating).
        """
        matrix = self.embed_documents(texts)
        return [matrix[i] for i in range(matrix.shape[0])]

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalised vectors.

        Because both vectors are already unit-length (from embed_*), this
        reduces to a dot product, returning a value in [0, 1].

        Used by the Evaluation tab to display explicit similarity scores.
        """
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-9
        return float(np.dot(vec_a, vec_b) / denom)
