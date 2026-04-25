"""
architecture.py — Part F: Architecture & System Design
=======================================================

Contains:
  • ARCHITECTURE_MERMAID  — Mermaid diagram source (rendered in the UI)
  • ARCHITECTURE_DESCRIPTION — textual walkthrough of each component
  • DOMAIN_JUSTIFICATION  — why this design fits the Ghana Elections/Budget domain
  • DATA_FLOW_STEPS        — ordered list of pipeline stages for display
"""

# ---------------------------------------------------------------------------
# Mermaid diagram (Part F — Data Flow + Component Interaction)
# ---------------------------------------------------------------------------

ARCHITECTURE_MERMAID = """
flowchart TD
    U([👤 User Query]) --> MEM[🧠 ConversationMemory\\nContext Injection]
    MEM --> QE[🔍 Query Expansion\\nYear + Domain Keywords]
    QE --> EMB[📐 EmbeddingPipeline\\nHashingVectorizer + L2 Norm]
    EMB --> VS[🗄️ CustomVectorStore\\nNumPy Cosine Search]
    VS --> CAND[Top-4K Candidates]
    CAND --> HYB[⚖️ HybridRetriever\\nVector × 0.75 + TF-IDF × 0.25]
    HYB --> DOM[🎯 Domain Boost\\nSource + Year Metadata]
    DOM --> FB[💡 FeedbackLoop\\nHistorical Vote Boost]
    FB --> TOPK[Top-K Results\\nWith Scores]
    TOPK --> CTX[📄 build_context\\nContext Window Management]
    CTX --> PT[✍️ Prompt Template\\nStrict / Base / CoT]
    PT --> LLM[🤖 LLM / Grounded Generator\\nOpenAI or Local Fallback]
    LLM --> LOG[📋 Stage Logger\\nlogs/pipeline_logs.jsonl]
    LLM --> ANS([💬 Answer + Citations])
    ANS --> FBK[👍👎 User Feedback]
    FBK --> FB

    style U fill:#6f42ff,color:#fff,stroke:none
    style ANS fill:#00a2ff,color:#fff,stroke:none
    style LLM fill:#172554,color:#fff
    style HYB fill:#1e3a8a,color:#fff
    style DOM fill:#7c3aed,color:#fff
    style FB fill:#16a34a,color:#fff
    style MEM fill:#dc2626,color:#fff
"""

# ---------------------------------------------------------------------------
# Textual architecture description (Part F)
# ---------------------------------------------------------------------------

ARCHITECTURE_DESCRIPTION = """
## System Architecture — AcityPal RAG

### Overview
AcityPal is a Retrieval-Augmented Generation (RAG) system built for a
dual-domain corpus: **Ghana Presidential Election Results (CSV)** and the
**2025 Ghana Budget Statement (PDF)**.

### Components & Data Flow

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | **User Query** | Free-text question from the Streamlit chat UI |
| 2 | **ConversationMemory** | Injects relevant prior turns to resolve coreferences |
| 3 | **Query Expansion** | Appends domain keywords for revenue/election queries to improve recall |
| 4 | **EmbeddingPipeline** | HashingVectorizer (bigrams) + L2 norm → dense float32 vector |
| 5 | **CustomVectorStore** | NumPy brute-force cosine search; returns top-4K candidates |
| 6 | **HybridRetriever** | Blends vector score (α=0.75) + TF-IDF keyword score (1-α=0.25) |
| 7 | **Domain Boost** | +0.05 for source-topic alignment; +0.03 for year match |
| 8 | **FeedbackLoop** | Applies historical up/downvote boost to personalise ranking |
| 9 | **build_context** | Assembles top-K chunks respecting max_words budget (1 200 words) |
| 10 | **Prompt Template** | Strict / Base / Chain-of-Thought variant injected with context |
| 11 | **LLM** | OpenAI gpt-4o-mini with temperature=0.1, or regex-grounded local fallback |
| 12 | **Stage Logger** | Every stage appends structured JSON to logs/pipeline_logs.jsonl |
| 13 | **Answer + Citations** | Response with [csv:region/year] or [pdf:page-N] source tags |
| 14 | **User Feedback** | 👍/👎 persisted to logs/feedback.jsonl, updates FeedbackLoop cache |
"""

# ---------------------------------------------------------------------------
# Domain suitability justification (Part F)
# ---------------------------------------------------------------------------

DOMAIN_JUSTIFICATION = """
## Why This Design is Suitable for the Domain

### 1. Dual-Source Heterogeneous Corpus
The dataset mixes structured tabular data (election CSV → natural-language doc)
with unstructured long-form policy text (PDF).  A RAG architecture is ideal
because it separates **knowledge storage** (chunks + index) from **reasoning**
(LLM), avoiding the need to fine-tune a model on domain data.

### 2. Precision Requirements for Electoral Facts
Election queries demand exact numerical accuracy ("1,234,567 votes in Ahafo
2020").  The strict prompt template and citation mandate prevent the LLM from
hallucinating vote counts.  The domain-specific year-matching boost further
ensures the correct election year's chunks surface first.

### 3. Scalability & Cost
The custom numpy vector store works with corpora up to ~50 K chunks without
external infrastructure.  The HashingVectorizer embedding is fully offline,
eliminating API embedding costs.  The OpenAI call is only made for final
answer generation, minimising per-query cost.

### 4. Adversarial Robustness
The relevance threshold (0.05) in `get_failure_case_demo()` and the
hallucination guard ("I do not have enough evidence…") provide defence against
out-of-domain queries that would otherwise produce confidently wrong answers.

### 5. Iterative Improvement via Feedback
The FeedbackLoop enables non-parametric learning from user signals without
retraining.  This is lightweight but effective for a domain where user trust
in factual accuracy is paramount.
"""

# ---------------------------------------------------------------------------
# Ordered pipeline stage list (for stage-log display in the UI)
# ---------------------------------------------------------------------------

DATA_FLOW_STEPS: list[dict] = [
    {"step": 1, "name": "Query Received",       "component": "Streamlit UI"},
    {"step": 2, "name": "Memory Injection",     "component": "ConversationMemory"},
    {"step": 3, "name": "Query Expansion",      "component": "RAGPipeline.retrieve()"},
    {"step": 4, "name": "Embedding",            "component": "EmbeddingPipeline"},
    {"step": 5, "name": "Vector Search",        "component": "CustomVectorStore"},
    {"step": 6, "name": "Hybrid Re-rank",       "component": "HybridRetriever"},
    {"step": 7, "name": "Domain + FB Boost",    "component": "HybridRetriever + FeedbackLoop"},
    {"step": 8, "name": "Context Assembly",     "component": "build_context()"},
    {"step": 9, "name": "Prompt Construction",  "component": "PROMPT_REGISTRY"},
    {"step": 10, "name": "LLM Generation",      "component": "generate_answer()"},
    {"step": 11, "name": "Logging",             "component": "log_jsonl()"},
    {"step": 12, "name": "Response + Citation", "component": "Streamlit UI"},
]
