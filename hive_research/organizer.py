from __future__ import annotations

import logging
from typing import Any

from .arxiv_fetcher import PaperInfo, fetch_by_id, search_arxiv
from .config import Config
from .graph import KnowledgeGraph
from .llm import LLMInterface
from .pipeline import PaperPipeline
from .pool import ResearchPool
from .rag import RAGEngine
from .similarity import paper_similarity_matrix

logger = logging.getLogger(__name__)


class Organizer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMInterface(config)
        self.kg = KnowledgeGraph(config)
        self.pipeline = PaperPipeline(config, self.llm, self.kg)
        self.rag = RAGEngine(config, self.llm, self.kg)
        self.pool = ResearchPool(config.root_dir / "pool")

    def add_by_id(self, arxiv_id: str) -> dict[str, Any]:
        paper = fetch_by_id(arxiv_id)
        if not paper:
            return {"status": "error", "message": f"Paper {arxiv_id} not found"}
        result = self.pipeline.process_paper(paper)
        if result["status"] == "added":
            pdf_text = ""
            pdf_path = self.config.papers_dir / f"{arxiv_id}.pdf"
            if pdf_path.exists():
                from .parser import extract_text
                pdf_text = extract_text(pdf_path)
            if pdf_text:
                n = self.rag.index_paper(arxiv_id, pdf_text)
                result["rag_chunks"] = n
        return result

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        mr = max_results or self.config.arxiv_max_results
        papers = search_arxiv(query, max_results=mr)
        return [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "authors": p.authors_str,
                "published": p.published,
                "abstract": p.abstract[:300],
                "categories": p.categories,
            }
            for p in papers
        ]

    def add_by_search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        mr = max_results or self.config.arxiv_max_results
        papers = search_arxiv(query, max_results=mr)
        results = []
        for p in papers:
            r = self.add_by_id(p.arxiv_id)
            results.append(r)
        return results

    def query_rag(self, question: str) -> dict[str, Any]:
        return self.rag.answer(question)

    def similarity(self) -> list[dict[str, Any]]:
        return paper_similarity_matrix(self.kg)

    def stats(self) -> dict[str, Any]:
        return {
            **self.kg.stats(),
            "rag": self.rag.stats(),
        }

    def graph_data(self) -> dict[str, Any]:
        return self.kg.to_node_link()
