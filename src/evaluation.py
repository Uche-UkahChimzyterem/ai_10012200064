import json
from src.pipeline import RAGPipeline
from src.config import EVAL_RESULTS_PATH
from src.llm import generate_without_retrieval


def hallucination_flag(answer):
    abstain = "I do not have enough evidence in the retrieved sources."
    # Proxy: abstention or explicit source tagging is treated as lower hallucination risk.
    if abstain in answer:
        return 0
    if "[csv:" in answer or "[pdf:" in answer:
        return 0
    return 1


def run_experiments():
    rag = RAGPipeline()

    benchmark_queries = [
        {"q": "Who won more votes in Ahafo Region in 2020?", "expected_terms": ["Nana Akufo Addo", "Ahafo", "2020"]},
        {"q": "What does the 2025 budget say about revenue mobilization?", "expected_terms": ["revenue", "budget", "2025"]},
    ]
    adversarial_queries = [
        "In 2030, which candidate won in all regions in the dataset?",
        "The budget claims cocoa exports dropped by 80% in 2025. Confirm exact figure and policy response.",
    ]

    records = {"benchmark": [], "adversarial": [], "prompt_comparison": [], "failure_case_and_fix": {}, "rag_vs_pure_llm": []}

    for item in benchmark_queries:
        q = item["q"]
        with_rag = rag.answer(q, use_hybrid=True, prompt_variant="strict")
        no_fix = rag.answer(q, use_hybrid=False, prompt_variant="strict")
        score_rag = sum(t.lower() in with_rag["response"].lower() for t in item["expected_terms"]) / len(item["expected_terms"])
        score_plain = sum(t.lower() in no_fix["response"].lower() for t in item["expected_terms"]) / len(item["expected_terms"])
        records["benchmark"].append({
            "query": q,
            "rag_accuracy_proxy": score_rag,
            "no_fix_accuracy_proxy": score_plain,
            "rag_hallucination_flag": hallucination_flag(with_rag["response"]),
            "no_fix_hallucination_flag": hallucination_flag(no_fix["response"]),
            "rag_response": with_rag["response"],
            "no_fix_response": no_fix["response"],
        })

        pure_llm = generate_without_retrieval(q)
        score_pure = sum(t.lower() in pure_llm.lower() for t in item["expected_terms"]) / len(item["expected_terms"])
        records["rag_vs_pure_llm"].append(
            {
                "query": q,
                "rag_response": with_rag["response"],
                "pure_llm_response": pure_llm,
                "rag_accuracy_proxy": score_rag,
                "pure_llm_accuracy_proxy": score_pure,
                "rag_hallucination_flag": hallucination_flag(with_rag["response"]),
                "pure_llm_hallucination_flag": hallucination_flag(pure_llm),
            }
        )

    prompt_q = "Summarize key election insight for Ahafo 2020 with source evidence."
    base_out = rag.answer(prompt_q, prompt_variant="base")
    strict_out = rag.answer(prompt_q, prompt_variant="strict")
    records["prompt_comparison"].append({
        "query": prompt_q,
        "base_prompt_response": base_out["response"],
        "strict_prompt_response": strict_out["response"],
    })

    for q in adversarial_queries:
        r1 = rag.answer(q, use_hybrid=True, prompt_variant="strict")
        r2 = rag.answer(q, use_hybrid=True, prompt_variant="strict")
        records["adversarial"].append({
            "query": q,
            "response_1": r1["response"],
            "response_2": r2["response"],
            "consistency": int(r1["response"].strip() == r2["response"].strip()),
            "hallucination_rate_proxy": (hallucination_flag(r1["response"]) + hallucination_flag(r2["response"])) / 2,
        })

    fail_query = "What is the fiscal deficit target and how does NDC perform in Ahafo?"
    fail_before = rag.answer(fail_query, use_hybrid=False, prompt_variant="strict")
    fail_after = rag.answer(fail_query, use_hybrid=True, prompt_variant="strict")
    records["failure_case_and_fix"] = {
        "query": fail_query,
        "before_fix_vector_only": fail_before["response"],
        "after_fix_hybrid": fail_after["response"],
        "fix_summary": "Hybrid re-ranking blends vector and keyword relevance to reduce topic drift for mixed queries.",
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved evaluation results to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    run_experiments()
