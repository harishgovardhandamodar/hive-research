from __future__ import annotations

import logging
from typing import Any

from .arxiv_fetcher import PaperInfo, fetch_by_id, fetch_by_id_with_meta, search_arxiv
from .config import Config
from .graph import KnowledgeGraph
from .llm import LLMInterface
from .pipeline import PaperPipeline
from .pool import ResearchPool
from .rag import RAGEngine
from .similarity import paper_similarity_matrix
from .web_ingest import WebIngester

logger = logging.getLogger(__name__)


class Organizer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMInterface(config)
        self.kg = KnowledgeGraph(config)
        self.pipeline = PaperPipeline(config, self.llm, self.kg)
        self.rag = RAGEngine(config, self.llm, self.kg)
        self.pool = ResearchPool(config.root_dir / "pool")
        self.web = WebIngester(self.llm, self.kg)

    def add_by_id(self, arxiv_id: str, with_lineage: bool = False) -> dict[str, Any]:
        result = fetch_by_id_with_meta(arxiv_id)
        if result["status"] == "error":
            return result
        paper = result["paper"]
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

    def fetch_lineage(self, arxiv_id: str) -> dict[str, Any]:
        pdf_path = self.config.papers_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            return {"status": "error", "message": f"No PDF found for {arxiv_id}"}
        from .parser import extract_text
        pdf_text = extract_text(pdf_path)
        if not pdf_text:
            return {"status": "error", "message": "Could not extract text from PDF"}
        refs = self.pipeline.fetch_lineage(arxiv_id, pdf_text)
        return {"status": "ok", "arxiv_id": arxiv_id, "references": refs}

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

    def generate_definitions(self) -> dict[str, Any]:
        from hive_datatype import NodeType
        concepts_no_def = [
            n for n in self.kg._hive.nodes
            if n.type == NodeType.CONCEPT and not n.definition
        ]
        if not concepts_no_def:
            return {"status": "ok", "generated": 0}
        generated = 0
        for node in concepts_no_def:
            paper_ids = set()
            for e in self.kg._hive.edges:
                if e.source == node.id:
                    paper_ids.add(e.target)
                elif e.target == node.id:
                    paper_ids.add(e.source)
            context = ""
            for pid in list(paper_ids)[:2]:
                p = self.kg.get_paper(pid)
                if p and p.abstract:
                    context += f"\nPaper '{p.label}': {p.abstract[:500]}"
            if not context:
                continue
            prompt = (
                f"Define the concept '{node.label}' concisely in 1-2 sentences "
                f"based on these papers:\n{context}\n\n"
                'Respond with JSON: {"definition": "..."}'
            )
            result = self.llm.extract_structured(prompt, model=self.config.ollama_fast_model)
            definition = result.get("definition", "")
            if definition:
                node.definition = definition[:200]
                generated += 1
        if generated:
            self.kg.save()
        return {"status": "ok", "generated": generated}
