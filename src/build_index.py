import pickle
from src.config import CHUNKS_A, VECTOR_INDEX_PATH
from src.embedding import EmbeddingPipeline
from src.utils import read_jsonl


def main():
    docs = read_jsonl(CHUNKS_A)
    emb = EmbeddingPipeline()
    vecs = emb.embed_documents([d["text"] for d in docs])
    with open(VECTOR_INDEX_PATH, "wb") as f:
        pickle.dump({"docs": docs, "vectors": vecs}, f)
    print(f"Saved vector index to {VECTOR_INDEX_PATH}")


if __name__ == "__main__":
    main()
