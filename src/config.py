from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "Ghana_Election_Result.csv"
PDF_PATH = ROOT / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

CLEANED_CSV = OUTPUT_DIR / "cleaned_election_data.csv"
CHUNKS_A = OUTPUT_DIR / "chunks_strategy_a.jsonl"
CHUNKS_B = OUTPUT_DIR / "chunks_strategy_b.jsonl"
CHUNK_COMPARE = OUTPUT_DIR / "chunking_comparison.json"
VECTOR_INDEX_PATH = OUTPUT_DIR / "vector_index.pkl"
EVAL_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"
PIPELINE_LOG_PATH = LOG_DIR / "pipeline_logs.jsonl"
