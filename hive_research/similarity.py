from __future__ import annotations

from typing import Any

from .graph import KnowledgeGraph


def jaccard_tokens(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def paper_similarity_matrix(
    kg: KnowledgeGraph,
) -> list[dict[str, Any]]:
    papers = kg.papers
    results = []
    for i, p1 in enumerate(papers):
        for p2 in papers[i + 1:]:
            authors_a = set(a.strip().lower() for a in p1.authors.split(",") if a.strip())
            authors_b = set(a.strip().lower() for a in p2.authors.split(",") if a.strip())
            author_overlap = len(authors_a & authors_b) / max(len(authors_a | authors_b), 1)
            abstract_sim = jaccard_tokens(p1.abstract, p2.abstract)
            shared = 0
            for e in kg.edges:
                if {e.source, e.target} == {p1.id, p2.id}:
                    shared += 1
            score = 0.4 * author_overlap + 0.4 * abstract_sim + 0.2 * min(shared / 5.0, 1.0)
            if score > 0:
                results.append({
                    "source": p1.id,
                    "source_title": p1.label,
                    "target": p2.id,
                    "target_title": p2.label,
                    "score": round(score, 4),
                    "author_overlap": round(author_overlap, 4),
                    "abstract_sim": round(abstract_sim, 4),
                    "shared_edges": shared,
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
