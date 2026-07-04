from __future__ import annotations

from typing import Any

from .graph import KnowledgeGraph


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _author_score(p1: Any, p2: Any) -> float:
    aa = set(a.strip().lower() for a in p1.authors.split(",") if a.strip())
    ab = set(a.strip().lower() for a in p2.authors.split(",") if a.strip())
    return len(aa & ab) / max(len(aa | ab), 1)


def _abstract_score(p1: Any, p2: Any) -> float:
    return jaccard_tokens(p1.abstract, p2.abstract)


def _concept_score(kg: KnowledgeGraph, pid1: str, pid2: str) -> float:
    c1 = set()
    c2 = set()
    for e in kg.edges:
        if e.source == pid1:
            c1.add(e.target)
        elif e.source == pid2:
            c2.add(e.target)
    if not c1 or not c2:
        return 0.0
    return len(c1 & c2) / len(c1 | c2)


def _edge_score(kg: KnowledgeGraph, pid1: str, pid2: str) -> float:
    shared = 0
    for e in kg.edges:
        if {e.source, e.target} == {pid1, pid2}:
            shared += 1
    return min(shared / 5.0, 1.0)


ALGORITHMS: dict[str, dict[str, Any]] = {
    "combined": {
        "label": "Combined (default)",
        "desc": "Authors + Abstract + Edges",
        "fn": lambda kg, p1, p2, pid1, pid2: (
            0.4 * _author_score(p1, p2)
            + 0.4 * _abstract_score(p1, p2)
            + 0.2 * _edge_score(kg, pid1, pid2)
        ),
    },
    "abstract": {
        "label": "Abstract Jaccard",
        "desc": "Abstract token overlap",
        "fn": lambda kg, p1, p2, pid1, pid2: _abstract_score(p1, p2),
    },
    "author": {
        "label": "Author Overlap",
        "desc": "Shared authors",
        "fn": lambda kg, p1, p2, pid1, pid2: _author_score(p1, p2),
    },
    "concept": {
        "label": "Concept Overlap",
        "desc": "Shared graph concepts",
        "fn": lambda kg, p1, p2, pid1, pid2: _concept_score(kg, pid1, pid2),
    },
}


def paper_similarity_matrix(
    kg: KnowledgeGraph,
    paper_ids: list[str] | None = None,
    algorithm: str = "combined",
) -> list[dict[str, Any]]:
    algo = ALGORITHMS.get(algorithm, ALGORITHMS["combined"])
    papers = kg.papers
    if paper_ids:
        papers = [p for p in papers if p.id in paper_ids]
    results = []
    for i, p1 in enumerate(papers):
        for p2 in papers[i + 1:]:
            score = algo["fn"](kg, p1, p2, p1.id, p2.id)
            results.append({
                "source": p1.id,
                "source_title": p1.label,
                "target": p2.id,
                "target_title": p2.label,
                "score": round(score, 4),
                "author_overlap": round(_author_score(p1, p2), 4),
                "abstract_sim": round(_abstract_score(p1, p2), 4),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def shared_concepts(kg: KnowledgeGraph, paper_a: str, paper_b: str) -> list[str]:
    a_concepts = set()
    b_concepts = set()
    for e in kg.edges:
        if e.source == paper_a:
            a_concepts.add(e.target)
        elif e.source == paper_b:
            b_concepts.add(e.target)
    return list(a_concepts & b_concepts)
