import pickle
from src.config import CHUNKS_A, VECTOR_INDEX_PATH
from src.embedding import EmbeddingPipeline
from src.utils import read_jsonl


def main():
    print(f"Loading chunks from {CHUNKS_A}")
    docs = read_jsonl(CHUNKS_A)
    
    # Validate that each chunk has the correct structure
    print(f"Loaded {len(docs)} chunks")
    print("Validating chunk structure...")
    for i, doc in enumerate(docs[:5]):  # Check first 5 chunks
        print(f"Chunk {i}:")
        print(f"  - ID: {doc.get('id', 'MISSING')}")
        print(f"  - Text preview: {doc.get('text', 'MISSING')[:100]}...")
        print(f"  - Metadata keys: {list(doc.get('metadata', {}).keys())}")
        print(f"  - Source: {doc.get('metadata', {}).get('source', 'MISSING')}")
    
    print("\nGenerating embeddings...")
    emb = EmbeddingPipeline()
    vecs = emb.embed_documents([d["text"] for d in docs])
    
    print(f"Saving vector index to {VECTOR_INDEX_PATH}")
    with open(VECTOR_INDEX_PATH, "wb") as f:
        pickle.dump({"docs": docs, "vectors": vecs}, f)
    
    print("Vector index rebuilt successfully!")
    print(f"  - Chunks: {len(docs)}")
    print(f"  - Vector dimensions: {vecs.shape}")
    print(f"  - Index saved to: {VECTOR_INDEX_PATH}")


if __name__ == "__main__":
    main()
