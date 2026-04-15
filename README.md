# Academic City RAG Chatbot (Ghana Elections + 2025 Budget)

This project implements a full RAG system aligned to the exam rubric:
- Data cleaning + chunking strategy comparison
- Custom embedding pipeline + custom vector store
- Retrieval with similarity scoring + hybrid search fix for failure cases
- Prompt engineering with hallucination control
- Full pipeline with stage-by-stage logging
- Adversarial evaluation and RAG vs no-retrieval comparison
- Streamlit UI
- Architecture and experiment documentation

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Build knowledge base

```bash
python -m src.data_prep
python -m src.build_index
```

## 3) Run evaluation

```bash
python -m src.evaluation
```

## 4) Launch app

```bash
streamlit run app.py
```
