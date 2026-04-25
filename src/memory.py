"""
memory.py — Part G: Innovation Component — Memory-Based RAG + Feedback Loop
============================================================================

This module implements TWO innovation features:

1. ConversationMemory
   -----------------
   Stores prior Q&A turns in-session and injects relevant turn summaries
   as additional context to subsequent queries.  This allows the system to:
     • Resolve coreferences ("what about NDC there?" → "in Ahafo 2020")
     • Avoid repeating already-stated caveats
     • Surface connections across multi-turn reasoning

   Memory injection strategy:
     The last N turns are serialised as plain text and prepended to the
     retrieved context block. A recency weight (newer = higher weight) is
     applied so stale turns do not dominate.

2. FeedbackLoop
   ------------
   Users can up/downvote any assistant answer (👍 / 👎 from the UI).
   Votes are persisted to logs/feedback.jsonl.

   At retrieval time, FeedbackLoop.score_boost(doc_id) returns a small
   additive bonus derived from historical positive feedback for chunks that
   appeared in highly-rated answers.  This is a lightweight relevance-learning
   mechanism that improves over successive interactions without retraining.

   Feedback schema (per JSONL line):
     {
       "ts":       "2026-04-19T01:00:00",
       "query":    str,
       "doc_ids":  [str, ...],   ← chunk IDs in the retrieved set
       "vote":     +1 | -1,
       "response": str
     }
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.config import LOG_DIR
from src.utils import ensure_dirs

FEEDBACK_LOG = LOG_DIR / "feedback.jsonl"
MEMORY_MAX_TURNS = 4          # how many past turns to keep in context
MEMORY_RECENCY_DECAY = 0.85   # weight multiplier per turn back in history
FEEDBACK_BOOST_SCALE = 0.02   # max boost per positively-rated document


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """
    Maintains a short-term memory of prior Q&A turns within a session.

    Usage
    -----
        memory = ConversationMemory()
        # after each turn:
        memory.add_turn(query="...", response="...", doc_ids=[...])
        # before next turn:
        extra_ctx = memory.get_memory_context(current_query)
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS):
        self.max_turns = max_turns
        self._turns: list[dict] = []   # [{query, response, doc_ids, weight}]

    def add_turn(self, query: str, response: str, doc_ids: list[str]) -> None:
        """Record a completed Q&A turn."""
        self._turns.append({"query": query, "response": response, "doc_ids": doc_ids})
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def get_memory_context(self, current_query: str) -> str:
        """
        Build a memory context block for injection alongside retrieved chunks.

        Each past turn gets a recency weight: the most recent turn has weight
        1.0, the turn before has 0.85, then 0.72, and so on.  Only turns
        with a term-overlap score > 0 with the current query are included,
        preventing irrelevant history from polluting the context.

        Returns an empty string if no relevant history exists.
        """
        if not self._turns:
            return ""

        current_terms = set(re.findall(r"\w+", current_query.lower()))
        current_terms -= {"what", "the", "a", "is", "in", "of", "and", "for",
                          "did", "does", "how", "who", "which", "where", "when"}

        blocks: list[str] = []
        n = len(self._turns)
        for i, turn in enumerate(reversed(self._turns)):
            weight = MEMORY_RECENCY_DECAY ** i
            turn_terms = set(re.findall(r"\w+", turn["query"].lower()))
            overlap = len(current_terms & turn_terms)
            if overlap == 0:
                continue
            relevance = min(1.0, overlap / max(1, len(current_terms)))
            effective_weight = weight * relevance
            if effective_weight < 0.1:
                continue
            blocks.append(
                f"[memory:turn-{n - i}] relevance={effective_weight:.2f}\n"
                f"Previous Q: {turn['query']}\n"
                f"Previous A: {turn['response'][:300]}{'...' if len(turn['response']) > 300 else ''}"
            )

        if not blocks:
            return ""
        return "=== Conversation History (for context continuity) ===\n" + "\n\n".join(blocks)

    def clear(self) -> None:
        """Reset memory (e.g. when starting a new chat)."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------

class FeedbackLoop:
    """
    Persistent feedback collector that learns retrieval preferences over time.

    Votes are appended to logs/feedback.jsonl and loaded at startup to
    pre-populate the relevance score cache.
    """

    def __init__(self):
        ensure_dirs(LOG_DIR)
        # doc_id → cumulative vote total (positive = net upvoted)
        self._scores: dict[str, float] = defaultdict(float)
        self._load_existing()

    def _load_existing(self) -> None:
        """Read all past votes from disk and accumulate scores."""
        if not FEEDBACK_LOG.exists():
            return
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    vote = rec.get("vote", 0)
                    for doc_id in rec.get("doc_ids", []):
                        self._scores[doc_id] += vote
                except (json.JSONDecodeError, KeyError):
                    continue

    def record_vote(
        self,
        query: str,
        doc_ids: list[str],
        vote: int,
        response: str,
    ) -> None:
        """
        Persist a user vote (+1 upvote, -1 downvote).

        Updates the in-memory score cache immediately so subsequent queries
        in the same session benefit from the feedback without a restart.
        """
        assert vote in (1, -1), "vote must be +1 or -1"
        record = {
            "ts": datetime.utcnow().isoformat(),
            "query": query,
            "doc_ids": doc_ids,
            "vote": vote,
            "response": response,
        }
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for doc_id in doc_ids:
            self._scores[doc_id] += vote

    def score_boost(self, doc_id: str) -> float:
        """
        Return a retrieval score boost for a given document chunk.

        Boost is scaled by FEEDBACK_BOOST_SCALE and capped at ±0.04 so that
        feedback learning does not completely override the base retrieval signal.
        """
        raw = self._scores.get(doc_id, 0.0)
        # Sigmoid-style normalisation so extreme vote counts don't dominate
        normalised = raw / (1 + abs(raw))
        return float(normalised * FEEDBACK_BOOST_SCALE)

    def get_stats(self) -> dict:
        """Return summary statistics for the feedback store."""
        if not self._scores:
            return {"total_rated_docs": 0, "avg_score": 0.0, "top_docs": []}
        sorted_docs = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
        return {
            "total_rated_docs": len(self._scores),
            "avg_score": round(sum(self._scores.values()) / len(self._scores), 3),
            "top_docs": [{"doc_id": d, "score": s} for d, s in sorted_docs[:5]],
        }
