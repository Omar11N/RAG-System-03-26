"""
agents/graph.py — Assembles the LangGraph agentic RAG pipeline.

Flow:
  retrieve → evaluate_context → [rewrite → retrieve (loop)] → generate → evaluate_answer

Swap strategy: add/remove nodes and edges here to change pipeline behavior.
"""

from __future__ import annotations
from functools import partial

from langgraph.graph import StateGraph, END

from agents.state import RAGState
from agents import nodes
from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge: should we retry retrieval?
# ---------------------------------------------------------------------------
def should_retry(state: RAGState) -> str:
    if state.get("should_rewrite", False):
        log.info(
            f"Context quality {state['context_quality_score']:.2f} < "
            f"{cfg.agent.quality_score_threshold} — rewriting query "
            f"(attempt {state['retry_count'] + 1}/{cfg.agent.max_retries})"
        )
        return "rewrite"
    return "generate"


# ---------------------------------------------------------------------------
# Build compiled graph
# ---------------------------------------------------------------------------
def build_rag_graph(retriever, llm):
    """
    Returns a compiled LangGraph app.
    Pass retriever and llm at construction time so nodes stay pure functions.
    """
    builder = StateGraph(RAGState)

    # Bind dependencies to nodes
    builder.add_node("retrieve",          partial(nodes.node_retrieve, retriever=retriever))
    builder.add_node("evaluate_context",  partial(nodes.node_evaluate_context, llm=llm))
    builder.add_node("rewrite_query",     partial(nodes.node_rewrite_query, llm=llm))
    builder.add_node("generate",          partial(nodes.node_generate, llm=llm))
    builder.add_node("evaluate_answer",   nodes.node_evaluate_answer)

    # Edges
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve",         "evaluate_context")
    builder.add_conditional_edges(
        "evaluate_context",
        should_retry,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )
    builder.add_edge("rewrite_query",    "retrieve")
    builder.add_edge("generate",         "evaluate_answer")
    builder.add_edge("evaluate_answer",  END)

    graph = builder.compile()
    log.info("LangGraph RAG pipeline compiled")
    return graph


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------
def run_query(graph, query: str) -> dict:
    """Execute the full pipeline for a single query and return the state."""
    initial_state: RAGState = {
        "query": query,
        "original_query": query,
        "retrieved_chunks": [],
        "context_quality_score": 0.0,
        "context_quality_reason": "",
        "answer": "",
        "retry_count": 0,
        "should_rewrite": False,
        "ragas_scores": None,
        "final": False,
    }
    final_state = graph.invoke(initial_state)
    return final_state
