## Architecture Overview

Data flow:
User Query -> Query Embedding -> Top-k Vector Search -> Hybrid Re-rank (optional) -> Context Selection/Truncation -> Prompt Template -> LLM -> Final Answer

Components:
- `src/data_prep.py`: cleaning + chunking + chunk quality comparison
- `src/embedding.py`: custom hashing-based text embeddings
- `src/retrieval.py`: custom vector store + hybrid scoring
- `src/prompting.py`: prompt templates + context-window management
- `src/pipeline.py`: orchestration + logging
- `src/evaluation.py`: benchmark + adversarial + failure-case tests
- `app.py`: Streamlit interface

Why suitable:
- Handles mixed structured (CSV) and unstructured (PDF) sources.
- Hybrid retrieval improves multi-intent policy/election queries.
- Strict prompt and abstention rule reduce hallucination risk.
