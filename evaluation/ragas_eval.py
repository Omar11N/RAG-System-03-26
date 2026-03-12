"""
evaluation/ragas_eval.py — RAGAs-powered answer quality evaluation.

Scores each response on:
  - faithfulness       (is the answer grounded in the context?)
  - answer_relevancy   (does it address the question?)
  - context_precision  (are the retrieved chunks relevant?)
  - context_recall     (did we retrieve enough information?)

Swap strategy: add/remove metrics in cfg.evaluation.metrics
or swap the judge LLM via cfg.evaluation.ragas_llm_provider.
"""

from __future__ import annotations

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


def _get_ragas_llm():
    """Resolve the LLM to use as RAGAs judge."""
    provider = cfg.evaluation.ragas_llm_provider
    model = cfg.evaluation.ragas_llm_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=cfg.generation.openai_api_key)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=cfg.generation.anthropic_api_key)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=cfg.generation.gemini_api_key)
    else:
        raise ValueError(f"Unsupported RAGAs LLM provider: {provider}")


def evaluate(
    query: str,
    answer: str,
    chunks: list[dict],
    ground_truth: str | None = None,
) -> dict[str, float]:
    """
    Run RAGAs metrics and return a dict of {metric_name: score}.
    `ground_truth` is optional; if absent, recall-based metrics are skipped.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        log.warning("ragas / datasets not installed — skipping evaluation")
        return {}

    available_metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

    selected = [
        available_metrics[m]
        for m in cfg.evaluation.metrics
        if m in available_metrics
    ]

    # context_recall requires ground_truth
    if ground_truth is None and context_recall in selected:
        selected.remove(context_recall)

    context_list = [c["sentence_chunk"] for c in chunks]

    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [context_list],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    try:
        judge_llm = _get_ragas_llm()
        result = ragas_evaluate(dataset, metrics=selected, llm=judge_llm)
        scores = {str(k): round(float(v), 4) for k, v in result.items()}
        return scores
    except Exception as e:
        log.error(f"RAGAs evaluation error: {e}")
        return {}
