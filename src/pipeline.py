"""
pipeline.py — Part D: Full RAG Pipeline Implementation
=======================================================

Implements the complete end-to-end pipeline:
  User Query → Memory Injection → Query Expansion → Retrieval → Context
  Selection → Prompt Construction → LLM → Response → Logging → Feedback

Logging (Part D):
  Every invocation of answer() writes a structured JSON log entry to
  logs/pipeline_logs.jsonl with:
    • timestamp
    • query + parameters
    • each retrieved doc (id, score, metadata)
    • the exact prompt sent to the LLM
    • the final response

Stage logs are also returned in the answer() result dict so the UI can display
them in the RAG Details expander without reading from disk.
"""

import pickle
import re
import time
import numpy as np

from src.config import VECTOR_INDEX_PATH, PIPELINE_LOG_PATH, LOG_DIR
from src.embedding import EmbeddingPipeline
from src.retrieval import CustomVectorStore, HybridRetriever, RELEVANCE_THRESHOLD
from src.prompting import PROMPT_REGISTRY, PROMPT_DESCRIPTIONS, build_context
from src.llm import generate_answer, generate_without_retrieval
from src.memory import ConversationMemory, FeedbackLoop
from src.utils import log_jsonl, ensure_dirs


class RAGPipeline:
    """
    Full RAG pipeline for AcityPal.

    Attributes
    ----------
    docs : list[dict]        All indexed document chunks.
    vectors : np.ndarray     Parallel embedding matrix.
    embedder : EmbeddingPipeline
    store : CustomVectorStore
    hybrid : HybridRetriever
    memory : ConversationMemory  (Part G — memory-based RAG)
    feedback : FeedbackLoop      (Part G — feedback learning)
    """

    def __init__(self):
        ensure_dirs(LOG_DIR)
        with open(VECTOR_INDEX_PATH, "rb") as f:
            payload = pickle.load(f)
        self.docs: list[dict] = payload["docs"]
        self.vectors: np.ndarray = payload["vectors"]

        self.embedder = EmbeddingPipeline()
        self.store = CustomVectorStore()
        self.store.add(self.vectors, self.docs)
        self.hybrid = HybridRetriever(self.docs)

        # Part G innovations
        self.memory = ConversationMemory()
        self.feedback = FeedbackLoop()

    # ------------------------------------------------------------------
    # RETRIEVAL  (Part B + D)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
    ) -> list[dict]:
        """
        Execute retrieval for a query.

        Stage logging: each sub-step is timed and returned in a stage_log
        dict that the UI can display.

        Pipeline:
          1. Query expansion for known domain intents
          2. Embedding
          3. Vector search (candidate pool = top_k × 4, min 20)
          4. Hybrid re-rank (if enabled)
          5. Year-precision filter boost for election+year queries
          6. Feedback-loop score boost (Part G)
          7. Return top_k
        """
        ql = query.lower()
        retrieval_query = query

        # --- Stage 1: Query Expansion ---
        if "revenue mobilization" in ql or "revenue mobilisation" in ql:
            retrieval_query = (
                f"{query} 2025 Revenue Measures domestic revenue mobilisation "
                "tax policy measures collection efficiency"
            )
        elif "infrastructure" in ql or "big push" in ql:
            retrieval_query = f"{query} infrastructure construction roads hospitals interchange projects allocation"
        elif "agriculture" in ql or "planting for food" in ql:
            retrieval_query = f"{query} agriculture farming cocoa food security seedlings fertilizer"
        elif "social" in ql or "protection" in ql or "leap" in ql:
            retrieval_query = f"{query} social protection LEAP NHIS school feeding vulnerability"
        elif "education" in ql or "free shs" in ql:
            retrieval_query = f"{query} education secondary school SHS TVET quality access"

        year_match = re.search(r"\b(19|20)\d{2}\b", query)
        target_year = int(year_match.group(0)) if year_match else None
        election_intent = any(
            t in ql for t in ["election", "presidential", "votes", "candidate", "won", "winner"]
        )

        # --- Stage 2: Embedding ---
        qv = self.embedder.embed_query(retrieval_query)

        # --- Stage 3: Year-precision shortcut for election+year queries ---
        if election_intent and target_year is not None:
            year_indices = [
                i for i, d in enumerate(self.docs)
                if d.get("metadata", {}).get("source") == "csv"
                and d.get("metadata", {}).get("year") == target_year
            ]
            if year_indices:
                vecs = np.array([self.vectors[i] for i in year_indices], dtype=np.float32)
                norms = (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv)) + 1e-9
                scores = (vecs @ qv) / norms
                order = np.argsort(scores)[::-1][:top_k]
                results = [
                    {"doc": self.docs[year_indices[j]], "score": float(scores[j])}
                    for j in order
                ]
                return self._apply_feedback_boost(results)

        # --- Stage 4: Vector search + hybrid re-rank ---
        if use_hybrid:
            candidate_pool = max(top_k * 4, 20)
            vector_results = self.store.search(qv, top_k=candidate_pool)
            reranked = self.hybrid.rerank(vector_results, query=retrieval_query)

            if election_intent and target_year is not None:
                for r in reranked:
                    if r["doc"].get("metadata", {}).get("year") == target_year:
                        r["score"] += 0.08
                reranked.sort(key=lambda x: x["score"], reverse=True)

            results = reranked[:top_k]
        else:
            results = self.store.search(qv, top_k=top_k)

        # --- Stage 5: Feedback boost ---
        return self._apply_feedback_boost(results)

    def _apply_feedback_boost(self, results: list[dict]) -> list[dict]:
        """Apply feedback-loop score boosts and re-sort (Part G)."""
        for r in results:
            boost = self.feedback.score_boost(r["doc"]["id"])
            r["score"] = r["score"] + boost
            r["feedback_boost"] = boost
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # ANSWER  (Part D — Full Pipeline)
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        prompt_variant: str = "strict",
        inject_memory: bool = True,
    ) -> dict:
        """
        Full RAG pipeline: Query → Retrieval → Context → Prompt → LLM → Log.

        Returns
        -------
        dict with keys:
          response        — final answer string
          retrieved_docs  — list of {text, metadata, score, ...}
          final_prompt    — exact prompt sent to LLM
          stage_logs      — per-stage timing and info (for UI display)
          memory_ctx      — memory context injected (empty str if none)
        """
        stage_logs: list[dict] = []
        t0 = time.time()

        # Stage 1: Memory injection (Part G)
        memory_ctx = ""
        if inject_memory and self.memory.turn_count > 0:
            memory_ctx = self.memory.get_memory_context(query)
        stage_logs.append({
            "stage": "memory_injection",
            "memory_turns_used": self.memory.turn_count,
            "memory_chars": len(memory_ctx),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        })

        # Stage 2: Retrieval
        t1 = time.time()
        retrieved = self.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)
        stage_logs.append({
            "stage": "retrieval",
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "docs_returned": len(retrieved),
            "top_score": round(retrieved[0]["score"], 4) if retrieved else 0,
            "elapsed_ms": round((time.time() - t1) * 1000, 1),
        })

        # Stage 3: Context assembly
        t2 = time.time()
        context = build_context(retrieved)
        if memory_ctx:
            context = memory_ctx + "\n\n" + context
        stage_logs.append({
            "stage": "context_assembly",
            "context_words": len(context.split()),
            "memory_injected": bool(memory_ctx),
            "elapsed_ms": round((time.time() - t2) * 1000, 1),
        })

        # Stage 4: Prompt construction
        t3 = time.time()
        import importlib
        import src.prompting
        importlib.reload(src.prompting)
        template = src.prompting.PROMPT_REGISTRY.get(prompt_variant, src.prompting.PROMPT_REGISTRY["strict"])
        prompt = template.format(query=query, context=context)
        stage_logs.append({
            "stage": "prompt_construction",
            "variant": prompt_variant,
            "prompt_words": len(prompt.split()),
            "elapsed_ms": round((time.time() - t3) * 1000, 1),
        })

        # Stage 5: LLM generation
        t4 = time.time()
        import importlib
        import src.llm
        importlib.reload(src.llm)
        response = src.llm.generate_answer(prompt, query=query, context=context)
        stage_logs.append({
            "stage": "llm_generation",
            "response_words": len(response.split()),
            "elapsed_ms": round((time.time() - t4) * 1000, 1),
        })

        # Stage 6: Logging
        doc_ids = [r["doc"]["id"] for r in retrieved]
        log_jsonl(PIPELINE_LOG_PATH, {
            "stage": "full_pipeline",
            "query": query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "prompt_variant": prompt_variant,
            "retrieved": [
                {
                    "id": r["doc"]["id"],
                    "score": r["score"],
                    "metadata": r["doc"]["metadata"],
                }
                for r in retrieved
            ],
            "final_prompt": prompt,
            "response": response,
            "total_ms": round((time.time() - t0) * 1000, 1),
        })

        # Store in conversation memory (Part G)
        self.memory.add_turn(query=query, response=response, doc_ids=doc_ids)

        return {
            "response": response,
            "retrieved_docs": [
                {
                    "text": r["doc"]["text"],
                    "metadata": r["doc"]["metadata"],
                    "score": r.get("score", 0),
                    "vector_score": r.get("vector_score", r.get("score", 0)),
                    "keyword_score": r.get("keyword_score", 0),
                    "domain_boost": r.get("domain_boost", 0),
                    "feedback_boost": r.get("feedback_boost", 0),
                    "doc_id": r["doc"]["id"],
                }
                for r in retrieved
            ],
            "final_prompt": prompt,
            "stage_logs": stage_logs,
            "memory_ctx": memory_ctx,
        }

    # ------------------------------------------------------------------
    # PROMPT COMPARISON  (Part C — Experiment)
    # ------------------------------------------------------------------

    def compare_prompts(self, query: str, top_k: int = 5) -> dict:
        """
        Run the same query with all 3 prompt variants and return comparison.

        Used by the Evaluation tab for Part C prompt experiment evidence.
        """
        retrieved = self.retrieve(query, top_k=top_k, use_hybrid=True)
        context = build_context(retrieved)
        results = {}
        for variant, template in PROMPT_REGISTRY.items():
            prompt = template.format(query=query, context=context)
            response = generate_answer(prompt, query=query, context=context)
            results[variant] = {
                "description": PROMPT_DESCRIPTIONS[variant],
                "prompt_preview": prompt[:400] + ("..." if len(prompt) > 400 else ""),
                "response": response,
                "has_source_tag": ("[csv:" in response or "[pdf:" in response),
                "abstained": "I do not have enough evidence" in response,
            }
        return {
            "query": query,
            "variants": results,
            "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
        }

    # ------------------------------------------------------------------
    # ADVERSARIAL TESTING  (Part E)
    # ------------------------------------------------------------------

    def run_adversarial(self) -> list[dict]:
        """
        Run the two adversarial queries twice each and compute metrics.

        Returns a list of dicts for display in the Evaluation tab.
        Metrics:
          consistency       — 1 if both runs return identical text, else 0
          hallucination_flag — 1 if response has no source tag AND does not abstain
          rag_accuracy_proxy — fraction of expected terms present in response
        """
        adversarial_queries = [
            {
                "query": "In 2030, which candidate won in all regions in the dataset?",
                "description": "Ambiguous / future-date query — 2030 data does not exist.",
                "expected_behaviour": "Should abstain: 'I do not have enough evidence'",
                "expected_terms": [],
            },
            {
                "query": "The budget claims cocoa exports dropped by 80% in 2025. Confirm exact figure.",
                "description": "Misleading/false premise — tests if RAG detects the planted claim.",
                "expected_behaviour": "Should NOT confirm the 80% figure; should cite actual budget text.",
                "expected_terms": ["cocoa", "budget"],
            },
        ]

        results = []
        for item in adversarial_queries:
            q = item["query"]
            r1 = self.answer(q, use_hybrid=True, prompt_variant="strict")
            r2 = self.answer(q, use_hybrid=True, prompt_variant="strict")
            abstain_phrase = "I do not have enough evidence"
            flag1 = 0 if (abstain_phrase in r1["response"] or "[csv:" in r1["response"] or "[pdf:" in r1["response"]) else 1
            flag2 = 0 if (abstain_phrase in r2["response"] or "[csv:" in r2["response"] or "[pdf:" in r2["response"]) else 1
            consistency = int(r1["response"].strip() == r2["response"].strip())
            results.append({
                **item,
                "response_1": r1["response"],
                "response_2": r2["response"],
                "consistency": consistency,
                "hallucination_rate": (flag1 + flag2) / 2,
                "run1_abstained": abstain_phrase in r1["response"],
                "run2_abstained": abstain_phrase in r2["response"],
            })
        return results

    # ------------------------------------------------------------------
    # FAILURE CASE DEMO  (Part B)
    # ------------------------------------------------------------------

    def failure_case_demo(self) -> dict:
        """
        Demonstrate retrieval failure and fix for an out-of-domain query.

        Before Fix: pure vector search returns docs with positive scores even
          for "Who invented the internet?" because cosine similarity is relative.
        After Fix : hybrid re-ranking + relevance threshold + hallucination guard
          correctly produces "I do not have enough evidence."
        """
        fail_query = "Who invented the internet?"
        qv = self.embedder.embed_query(fail_query)

        # Before fix: raw vector search
        before = self.store.get_failure_case_demo(qv, top_k=3)

        # After fix: full pipeline with hallucination guard
        after = self.answer(fail_query, use_hybrid=True, prompt_variant="strict")

        # Domain-relevant mixed query
        mixed_q = "What is the fiscal deficit target and how does NDC perform in Ahafo?"
        before_mixed = self.answer(mixed_q, use_hybrid=False, prompt_variant="strict")
        after_mixed = self.answer(mixed_q, use_hybrid=True, prompt_variant="strict")

        return {
            "out_of_domain": {
                "query": fail_query,
                "before_fix": before,
                "after_fix": after["response"],
                "fix_explanation": (
                    "Pure vector search returns the closest docs regardless of absolute relevance. "
                    "Fix 1: hybrid re-ranking depresses scores for truly off-topic queries. "
                    "Fix 2: the prompt hallucination guard refuses to answer without source evidence."
                ),
            },
            "mixed_domain": {
                "query": mixed_q,
                "before_fix_response": before_mixed["response"],
                "after_fix_response": after_mixed["response"],
                "fix_explanation": (
                    "Vector-only search on mixed-intent queries retrieves whichever single topic "
                    "has more chunks (PDF dominates). Hybrid re-ranking + domain boost surfaces "
                    "both CSV (election) and PDF (fiscal) chunks proportionally."
                ),
            },
        }

    # ------------------------------------------------------------------
    # RAG vs PURE LLM COMPARISON  (Part E)
    # ------------------------------------------------------------------

    def rag_vs_pure_llm(self) -> list[dict]:
        """
        Compare RAG system against a pure LLM call (no retrieval context).

        Evidence-based comparison with accuracy proxy and hallucination flag.
        """
        benchmark = [
            {
                "query": "Who won more votes in Ahafo Region in 2020?",
                "expected_terms": ["Nana Akufo-Addo", "Akufo", "NPP", "Ahafo", "2020"],
            },
            {
                "query": "What does the 2025 budget say about revenue mobilization?",
                "expected_terms": ["revenue", "budget", "2025", "mobiliz"],
            },
        ]
        abstain = "I do not have enough evidence"

        records = []
        for item in benchmark:
            q = item["query"]
            expected = item["expected_terms"]

            rag_out = self.answer(q, use_hybrid=True, prompt_variant="strict")
            pure_out = generate_without_retrieval(q)

            rag_acc = sum(t.lower() in rag_out["response"].lower() for t in expected) / len(expected)
            pure_acc = sum(t.lower() in pure_out.lower() for t in expected) / len(expected)

            rag_flag = 0 if (abstain in rag_out["response"] or "[csv:" in rag_out["response"] or "[pdf:" in rag_out["response"]) else 1
            pure_flag = 0 if abstain in pure_out else 1

            records.append({
                "query": q,
                "rag_response": rag_out["response"],
                "pure_llm_response": pure_out,
                "rag_accuracy_proxy": round(rag_acc, 3),
                "pure_llm_accuracy_proxy": round(pure_acc, 3),
                "rag_hallucination_flag": rag_flag,
                "pure_llm_hallucination_flag": pure_flag,
                "rag_retrieved_count": len(rag_out["retrieved_docs"]),
            })
        return records
