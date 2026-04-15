import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class EmbeddingPipeline:
    """
    Custom embedding pipeline that is lightweight and deterministic.
    Uses n-gram hashing features + L2 normalization for cosine similarity.
    """

    def __init__(self, n_features=2048):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed_documents(self, texts):
        mat = self.vectorizer.transform(texts).astype(np.float32)
        return normalize(mat, norm="l2", axis=1).toarray()

    def embed_query(self, query):
        mat = self.vectorizer.transform([query]).astype(np.float32)
        return normalize(mat, norm="l2", axis=1).toarray()[0]
