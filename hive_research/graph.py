from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from hive_datatype import (
    Edge,
    HiveGraph,
    Node,
    NodeType,
)

from .config import Config

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    def __init__(self, config: Config, graph_id: str = "main") -> None:
        self.config = config
        self.graph_id = graph_id
        self.graph_dir = Path(config.graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self._hive = self._load()

    def _path(self) -> Path:
        return self.graph_dir / f"{self.graph_id}.json"

    def _load(self) -> HiveGraph:
        path = self._path()
        if path.exists():
            try:
                hive = HiveGraph.from_json_file(str(path))
                valid_ids = {n.id for n in hive.nodes}
                before = len(hive.edges)
                hive.edges = [e for e in hive.edges if e.source in valid_ids and e.target in valid_ids]
                if len(hive.edges) < before:
                    logger.warning("Removed %d edges with invalid node refs", before - len(hive.edges))
                return hive
            except Exception as e:
                logger.warning("Failed to load graph, starting fresh: %s", e)
        return HiveGraph(id=self.graph_id)

    def save(self) -> None:
        self._hive.to_json_file(str(self._path()))

    @property
    def hive(self) -> HiveGraph:
        return self._hive

    def add_paper(
        self,
        paper_id: str,
        title: str,
        authors: str = "",
        published: str = "",
        abstract: str = "",
        categories: list[str] | None = None,
        affiliations: str = "",
    ) -> Node:
        existing = self._hive.get_node(paper_id)
        if existing:
            if affiliations and not existing.affiliations:
                existing.affiliations = affiliations
                self.save()
            return existing
        node = Node(
            id=paper_id,
            type=NodeType.PAPER,
            label=title,
            graph_id=self.graph_id,
            arxiv_id=paper_id,
            authors=authors,
            published=published,
            abstract=abstract,
            categories=categories or [],
            affiliations=affiliations,
        )
        self._hive.nodes.append(node)
        return node

    def add_concept(
        self,
        concept_id: str,
        label: str,
        definition: str = "",
        concept_type: str = "concept",
    ) -> Node:
        existing = self._hive.get_node(concept_id)
        if existing:
            return existing
        node = Node(
            id=concept_id,
            type=NodeType.CONCEPT,
            label=label,
            graph_id=self.graph_id,
            definition=definition,
            concept_type=concept_type,
        )
        self._hive.nodes.append(node)
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "related_to",
    ) -> Edge:
        for e in self._hive.edges:
            if e.source == source and e.target == target and e.relation == relation:
                return e
        edge = Edge(source=source, target=target, relation=relation)
        self._hive.edges.append(edge)
        return edge

    def find_similar_concept(
        self,
        label: str,
        threshold: float | None = None,
    ) -> Node | None:
        threshold = threshold if threshold is not None \
            else self.config.graph_similarity_threshold
        label_lower = label.lower()
        label_tokens = set(label_lower.split())
        best_score = 0.0
        best_node: Node | None = None
        for node in self._hive.concepts:
            node_tokens = set(node.label.lower().split())
            if not node_tokens or not label_tokens:
                continue
            intersection = label_tokens & node_tokens
            union = label_tokens | node_tokens
            score = len(intersection) / len(union) if union else 0.0
            if score > best_score:
                best_score = score
                best_node = node
        if best_score >= threshold and best_node is not None:
            return best_node
        return None

    def get_paper(self, paper_id: str) -> Node | None:
        return self._hive.get_node(paper_id)

    def get_concept(self, concept_id: str) -> Node | None:
        return self._hive.get_node(concept_id)

    @property
    def papers(self) -> list[Node]:
        return self._hive.papers

    @property
    def concepts(self) -> list[Node]:
        return self._hive.concepts

    @property
    def edges(self) -> list[Edge]:
        return self._hive.edges

    def stats(self) -> dict[str, int]:
        s = self._hive.stats
        return {
            "papers": s.papers,
            "graph_papers": s.graph_papers,
            "concepts": s.concepts,
            "graph_refs": s.graph_refs,
            "relations": s.relations,
            "cross_edges": s.cross_edges,
        }

    def filter_nodes(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        relation: str | None = None,
        connected_only: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        concept_type: str | None = None,
        categories: list[str] | None = None,
    ) -> list[Node]:
        nodes = list(self._hive.nodes)

        if query:
            q = query.lower()
            nodes = [
                n for n in nodes
                if q in (n.label or "").lower()
                or q in (getattr(n, "authors", "") or "").lower()
                or q in (getattr(n, "abstract", "") or "").lower()
                or q in (getattr(n, "definition", "") or "").lower()
                or q in (getattr(n, "affiliations", "") or "").lower()
            ]

        if types:
            type_set = set(types)
            nodes = [n for n in nodes if str(n.type).lower() in type_set]

        if relation:
            linked_ids: set[str] = set()
            for e in self._hive.edges:
                if e.relation == relation:
                    linked_ids.add(e.source)
                    linked_ids.add(e.target)
            nodes = [n for n in nodes if n.id in linked_ids]

        if connected_only:
            edge_ids: set[str] = set()
            for e in self._hive.edges:
                edge_ids.add(e.source)
                edge_ids.add(e.target)
            nodes = [n for n in nodes if n.id in edge_ids]

        if date_from or date_to:
            filtered = []
            for n in nodes:
                pub = getattr(n, "published", "") or ""
                if pub:
                    pub_date = pub[:10]
                    if date_from and pub_date < date_from:
                        continue
                    if date_to and pub_date > date_to:
                        continue
                filtered.append(n)
            nodes = filtered

        if concept_type:
            nodes = [
                n for n in nodes
                if getattr(n, "concept_type", "") == concept_type
            ]

        if categories:
            cat_set = set(categories)
            nodes = [
                n for n in nodes
                if cat_set & set(getattr(n, "categories", []) or [])
            ]

        return nodes

    def filter_edges(
        self,
        relation: str | None = None,
        source_types: list[str] | None = None,
        target_types: list[str] | None = None,
        node_ids: set[str] | None = None,
    ) -> list[Edge]:
        edges = list(self._hive.edges)

        if relation:
            edges = [e for e in edges if e.relation == relation]

        if node_ids:
            edges = [e for e in edges if e.source in node_ids or e.target in node_ids]

        if source_types or target_types:
            node_map = {n.id: n for n in self._hive.nodes}
            if source_types:
                st_set = set(source_types)
                edges = [
                    e for e in edges
                    if (src := node_map.get(e.source)) and str(src.type).lower() in st_set
                ]
            if target_types:
                tt_set = set(target_types)
                edges = [
                    e for e in edges
                    if (tgt := node_map.get(e.target)) and str(tgt.type).lower() in tt_set
                ]

        return edges

    def filtered_to_node_link(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        relation: str | None = None,
        connected_only: bool = False,
        date_from: str | None = None,
        date_to: str | None = None,
        concept_type: str | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        nodes = self.filter_nodes(
            query=query,
            types=types,
            relation=relation,
            connected_only=connected_only,
            date_from=date_from,
            date_to=date_to,
            concept_type=concept_type,
            categories=categories,
        )
        node_ids = {n.id for n in nodes}
        edges = [e for e in self._hive.edges if e.source in node_ids and e.target in node_ids]
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return {
            "nodes": [{"id": n.id, "label": n.label, "type": n.type} | {
                k: getattr(n, k, "")
                for k in ("authors", "published", "abstract", "definition",
                          "affiliations", "categories", "concept_type", "arxiv_id")
                if getattr(n, k, None)
            } for n in nodes],
            "links": [{"source": e.source, "target": e.target, "relation": e.relation}
                      for e in edges],
        }

    def to_node_link(self) -> dict[str, Any]:
        return self._hive.to_node_link_dict()
