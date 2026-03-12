"""
agents/nodes.py — Individual LangGraph node functions.

Each node is a pure function: RAGState → partial RAGState update.
Add new nodes here to extend the pipeline without touching the graph wiring.
"""

from __future__ import annotations
import json

from agents.state import RAGState
from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------
def node_retrieve(state: RAGState, retriever) -> RAGState:
    """Fetch top-k chunks for the current query."""
    query = state["query"]
    log.info(f"[retrieve] query='{query[:80]}...' retry={state['retry_count']}")

    chunks = retriever.retrieve(query)
    return {**state, "retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# Node: evaluate_context_quality
# ---------------------------------------------------------------------------
def node_evaluate_context(state: RAGState, llm) -> RAGState:
    """
    Ask the LLM to score the retrieved context 0–1.
    Low score → trigger query rewrite.
    """
    from generation.prompt_templates import build_quality_prompt

    if not state["retrieved_chunks"]:
        return {**state, "context_quality_score": 0.0, "context_quality_reason": "No chunks retrieved"}

    prompt = build_quality_prompt(state["query"], state["retrieved_chunks"])
    try:
        raw = llm.generate(prompt)
        clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean)
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
    except Exception as e:
        log.warning(f"Quality eval parse failed: {e} — defaulting to 0.5")
        score, reason = 0.5, "Parse error"

    log.info(f"[evaluate_context] score={score:.2f} | {reason}")
    threshold = cfg.agent.quality_score_threshold
    should_rewrite = (
        score < threshold
        and state["retry_count"] < cfg.agent.max_retries
    )
    return {**state, "context_quality_score": score, "context_quality_reason": reason, "should_rewrite": should_rewrite}


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------
def node_rewrite_query(state: RAGState, llm) -> RAGState:
    """Rewrite the query to improve retrieval on next attempt."""
    from generation.prompt_templates import build_rewrite_prompt

    prompt = build_rewrite_prompt(state["query"], state["retrieved_chunks"])
    new_query = llm.generate(prompt).strip().strip('"').strip("'")
    log.info(f"[rewrite_query] '{state['query']}' → '{new_query}'")
    return {
        **state,
        "query": new_query,
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Node: generate_answer
# ---------------------------------------------------------------------------
def node_generate(state: RAGState, llm) -> RAGState:
    """Generate the final answer from retrieved context."""
    from generation.prompt_templates import build_answer_prompt

    prompt = build_answer_prompt(state["query"], state["retrieved_chunks"])
    log.info(f"[generate] Generating answer for: '{state['original_query'][:60]}...'")
    answer = llm.generate(prompt)
    return {**state, "answer": answer, "final": True}


# ---------------------------------------------------------------------------
# Node: evaluate_answer (RAGAs)
# ---------------------------------------------------------------------------
def node_evaluate_answer(state: RAGState) -> RAGState:
    """Run RAGAs metrics on the final answer. Non-blocking — won't crash the pipeline."""
    from evaluation.ragas_eval import evaluate

    if not cfg.evaluation.enable_ragas:
        return {**state, "ragas_scores": None}

    try:
        scores = evaluate(
            query=state["original_query"],
            answer=state["answer"],
            chunks=state["retrieved_chunks"],
        )
        log.info(f"[ragas] scores={scores}")
    except Exception as e:
        log.warning(f"RAGAs evaluation failed (non-fatal): {e}")
        scores = None

    return {**state, "ragas_scores": scores}
