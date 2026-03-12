"""
ingestion/graph_indexer.py — Lightweight GraphRAG layer.

Extracts entities and relationships from each chunk using an LLM,
stores them in a NetworkX graph, and enables multi-hop context retrieval
for queries that span multiple document sections.

Swap strategy: replace `_extract_entities_llm` with any NER approach
(spaCy NER, GLiNER, full Microsoft GraphRAG) without changing the graph API.
"""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)

GRAPH_CACHE_PATH = Path(cfg.cache_dir) / "knowledge_graph.pkl"


# ---------------------------------------------------------------------------
# Entity extraction (LLM-backed)
# ---------------------------------------------------------------------------
ENTITY_EXTRACTION_PROMPT = """\
Extract key entities and relationships from the text below.
Return ONLY a JSON object with this exact schema — no markdown, no extra text:
{{
  "entities": ["entity1", "entity2"],
  "relationships": [["entity1", "relation", "entity2"]]
}}

Text:
{text}
"""


def _extract_entities_llm(text: str, llm_client) -> dict:
    """Call the configured LLM to extract entities from a chunk."""
    try:
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:1500])  # truncate for speed
        response = llm_client.generate(prompt)
        # Strip markdown fences if present
        clean = response.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)
    except Exception as e:
        log.debug(f"Entity extraction failed for chunk: {e}")
        return {"entities": [], "relationships": []}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
class KnowledgeGraphIndex:
    """
    Builds a NetworkX graph where:
    - Nodes = entities extracted from chunks
    - Edges = relationships between entities
    - Each node/edge is linked back to source chunk indices
    """

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()

    def build(self, chunks: list[dict], llm_client=None) -> "KnowledgeGraphIndex":
        """Build the graph from a list of chunk dicts."""
        if GRAPH_CACHE_PATH.exists():
            log.info("Loading knowledge graph from cache")
            return self.load()

        if llm_client is None:
            log.warning("No LLM client provided — skipping GraphRAG entity extraction")
            return self

        log.info(f"Building knowledge graph from {len(chunks)} chunks...")
        from tqdm.auto import tqdm

        for idx, chunk in tqdm(enumerate(chunks), desc="Extracting entities", total=len(chunks)):
            extracted = _extract_entities_llm(chunk["sentence_chunk"], llm_client)

            for entity in extracted.get("entities", []):
                entity = entity.strip().lower()
                if not entity:
                    continue
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, chunk_indices=[idx], page=chunk["page_number"])
                else:
                    self.graph.nodes[entity]["chunk_indices"].append(idx)

            for rel in extracted.get("relationships", []):
                if len(rel) == 3:
                    src, relation, tgt = rel
                    src, tgt = src.strip().lower(), tgt.strip().lower()
                    if src and tgt:
                        self.graph.add_edge(src, tgt, relation=relation, chunk_index=idx)

        log.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        self._save()
        return self

    def get_related_chunk_indices(self, query_entities: list[str], hops: int = 2) -> list[int]:
        """
        Given a list of query entities, walk the graph `hops` steps and
        return the union of all chunk indices encountered.
        """
        visited_chunks: set[int] = set()
        for entity in query_entities:
            entity = entity.strip().lower()
            if not self.graph.has_node(entity):
                continue
            # BFS up to `hops` away
            for neighbor in nx.single_source_shortest_path(self.graph, entity, cutoff=hops):
                node_data = self.graph.nodes[neighbor]
                visited_chunks.update(node_data.get("chunk_indices", []))
        return list(visited_chunks)

    def _save(self):
        GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_CACHE_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        log.info(f"Knowledge graph saved to {GRAPH_CACHE_PATH}")

    def load(self) -> "KnowledgeGraphIndex":
        with open(GRAPH_CACHE_PATH, "rb") as f:
            self.graph = pickle.load(f)
        log.info(
            f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self
