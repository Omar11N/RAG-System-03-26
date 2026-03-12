"""
generation/prompt_templates.py — All prompts in one place for easy iteration.
"""

# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
RAG_ANSWER_PROMPT = """\
You are a precise, helpful assistant answering questions about a document.

Use ONLY the context passages below to answer. If the context doesn't contain
the answer, say so clearly — do not hallucinate.

Context passages:
{context}

Question: {query}

Provide a thorough, well-structured answer with specific references to the context.
Answer:"""


# ---------------------------------------------------------------------------
# Query rewriting (used on retry)
# ---------------------------------------------------------------------------
QUERY_REWRITE_PROMPT = """\
The following query did not retrieve high-quality context passages from a document.
Rewrite it to be more specific and likely to match relevant text in the document.
Return ONLY the rewritten query — no explanation.

Original query: {query}
Previous low-quality context summary: {context_summary}

Rewritten query:"""


# ---------------------------------------------------------------------------
# Retrieval quality check (before generating an answer)
# ---------------------------------------------------------------------------
CONTEXT_QUALITY_PROMPT = """\
You are evaluating whether a set of retrieved passages contains enough relevant
information to answer the question below.

Question: {query}

Retrieved passages:
{context}

Score the relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
Return ONLY a JSON object like: {{"score": 0.75, "reason": "brief explanation"}}
"""


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] (Page {chunk.get('page_number', '?')})\n{chunk['sentence_chunk']}")
    return "\n\n".join(lines)


def build_answer_prompt(query: str, chunks: list[dict]) -> str:
    return RAG_ANSWER_PROMPT.format(context=format_context(chunks), query=query)


def build_quality_prompt(query: str, chunks: list[dict]) -> str:
    return CONTEXT_QUALITY_PROMPT.format(context=format_context(chunks), query=query)


def build_rewrite_prompt(query: str, chunks: list[dict]) -> str:
    context_summary = " ".join(c["sentence_chunk"][:100] for c in chunks[:3])
    return QUERY_REWRITE_PROMPT.format(query=query, context_summary=context_summary)
