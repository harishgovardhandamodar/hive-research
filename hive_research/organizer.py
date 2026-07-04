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

    def notes_path_for(self, paper_id: str) -> str | None:
        n = self.kg.get_paper(paper_id)
        if not n:
            return None
        from .pipeline import _sanitize_id
        safe = _sanitize_id(n.label) or paper_id
        p = Path(self.config.vault_dir) / f"{safe}.md"
        return str(p) if p.exists() else None

    def refresh_papers(self) -> dict[str, Any]:
        import json as _json
        from .parser import extract_text
        from hive_datatype import NodeType
        import threading
        refreshed = [0]
        def _do_refresh():
            for node in self.kg._hive.nodes:
                if node.type != NodeType.PAPER:
                    continue
                has_extra = False
                if node.definition:
                    try:
                        parsed = _json.loads(node.definition)
                        has_extra = bool(parsed.get("notes") or parsed.get("experiment") or parsed.get("results"))
                    except Exception:
                        has_extra = False
                if has_extra:
                    continue
                pdf_path = self.config.papers_dir / f"{node.arxiv_id}.pdf"
                if not pdf_path.exists():
                    continue
                text = extract_text(pdf_path)
                if not text:
                    continue
                logger.info("Refreshing %s — %s", node.arxiv_id, node.label[:60])
                analysis = self.pipeline._analyze_text(text, node.label)
                notes = analysis.get("notes", "")
                experiment = analysis.get("experiment", {})
                results = analysis.get("results", {})
                extra = _json.dumps({"notes": notes, "experiment": experiment, "results": results})
                node.definition = extra[:2000]
                paper_info = fetch_by_id(node.arxiv_id.split("v")[0] if "v" in (node.arxiv_id or "") else node.arxiv_id)
                if paper_info:
                    summary = analysis.get("summary", "")
                    tags = analysis.get("tags", [])
                    concepts_data = analysis.get("concepts", [])
                    self.pipeline._write_note(node.arxiv_id, paper_info, summary, tags, concepts_data, notes, experiment, results)
                refreshed[0] += 1
            if refreshed[0]:
                self.kg.save()
                logger.info("Refresh complete: %d papers updated", refreshed[0])
        t = threading.Thread(target=_do_refresh, daemon=True)
        t.start()
        return {"status": "started", "message": f"Refreshing papers in background. Check activity log for progress."}

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
