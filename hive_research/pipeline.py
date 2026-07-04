from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from .arxiv_fetcher import PaperInfo, download_pdf, fetch_by_id
from .config import Config
from .graph import KnowledgeGraph
from .llm import LLMInterface
from .parser import extract_referenced_arxiv_ids, extract_sections, extract_text

logger = logging.getLogger(__name__)


def _sanitize_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:60]


class PaperPipeline:
    def __init__(
        self,
        config: Config,
        llm: LLMInterface,
        kg: KnowledgeGraph,
    ) -> None:
        self.config = config
        self.llm = llm
        self.kg = kg

    def process_paper(self, paper: PaperInfo) -> dict[str, Any]:
        paper_id = paper.arxiv_id
        existing = self.kg.get_paper(paper_id)
        if existing:
            return {"status": "exists", "paper_id": paper_id}

        node = self.kg.add_paper(
            paper_id=paper_id,
            title=paper.title,
            authors=paper.authors_str,
            published=paper.published,
            abstract=paper.abstract,
            categories=paper.categories,
            affiliations=paper.affiliations_str,
        )

        pdf_text = ""
        pdf_path = None
        if self.config.arxiv_download_pdf:
            pdf_path = download_pdf(paper_id, self.config.papers_dir)
            if pdf_path and pdf_path.exists():
                pdf_text = extract_text(pdf_path)
        text_for_analysis = pdf_text or paper.abstract

        analysis = self._analyze_text(text_for_analysis, paper.title)

        concepts = analysis.get("concepts", [])
        relations = analysis.get("relations", [])
        summary = analysis.get("summary", "")
        tags = analysis.get("tags", [])
        notes = analysis.get("notes", "")
        experiment = analysis.get("experiment", {})
        results = analysis.get("results", {})

        import json as _json
        extra = _json.dumps({"notes": notes, "experiment": experiment, "results": results})
        node.definition = extra[:2000]

        for tag in tags:
            tag_id = _sanitize_id(tag)
            matched = self.kg.find_similar_concept(tag)
            if matched:
                cid = matched.id
            else:
                cid = tag_id
                self.kg.add_concept(cid, tag, definition=f"A paper tagged with '{tag}'.", concept_type="tag")
            self.kg.add_edge(paper_id, cid, "related_to")

        for c in concepts:
            cid = _sanitize_id(c.get("name", "")) or _sanitize_id(c.get("label", ""))
            if not cid:
                continue
            label = c.get("name", c.get("label", cid))
            definition = c.get("definition", "")
            matched = self.kg.find_similar_concept(label)
            if matched:
                cid = matched.id
                if definition and not matched.definition:
                    matched.definition = definition
            else:
                self.kg.add_concept(
                    cid,
                    label,
                    definition=definition,
                    concept_type=c.get("type", "concept"),
                )
            rel = c.get("relation", "related_to")
            self.kg.add_edge(paper_id, cid, rel)

        for r in relations:
            raw_src = r.get("source", paper_id)
            raw_tgt = r.get("target", "")
            rel = r.get("relation", "related_to")
            if raw_src and raw_tgt:
                src = self._resolve_id(raw_src, paper_id)
                tgt = self._resolve_id(raw_tgt, paper_id)
                if src and tgt:
                    self.kg.add_edge(src, tgt, rel)

        note_path = self._write_note(paper_id, paper, summary, tags, concepts, notes, experiment, results)
        self.kg.save()

        result = {
            "status": "added",
            "paper_id": paper_id,
            "concepts": len(concepts),
            "tags": len(tags),
            "relations": len(relations),
            "note_path": str(note_path) if note_path else None,
            "has_notes": bool(notes),
            "has_experiment": bool(experiment and isinstance(experiment, dict) and any(v for v in experiment.values())),
            "has_results": bool(results and isinstance(results, dict) and any(v for v in results.values())),
        }

        if pdf_text:
            refs = self.fetch_lineage(paper_id, pdf_text)
            if refs:
                result["lineage"] = refs
                logger.info("Lineage: %d prior papers linked for %s", len(refs), paper_id)

        return result

    def fetch_lineage(self, paper_id: str, pdf_text: str, max_refs: int = 10) -> list[dict[str, Any]]:
        ref_ids = extract_referenced_arxiv_ids(pdf_text)
        if not ref_ids:
            return []
        fetched = []
        for i, aid in enumerate(ref_ids[:max_refs]):
            if self.kg.get_paper(aid):
                self.kg.add_edge(paper_id, aid, "cites")
                fetched.append({"arxiv_id": aid, "status": "exists"})
                continue
            if i > 0:
                time.sleep(3)
            prior = fetch_by_id(aid)
            if prior is None:
                continue
            self.kg.add_paper(
                paper_id=aid,
                title=prior.title,
                authors=prior.authors_str,
                published=prior.published,
                abstract=prior.abstract,
                categories=prior.categories,
                affiliations=prior.affiliations_str,
            )
            self.kg.add_edge(paper_id, aid, "cites")
            fetched.append({"arxiv_id": aid, "title": prior.title[:80], "status": "added"})
            logger.info("Lineage: linked prior paper %s — %s", aid, prior.title[:80])
        if fetched:
            self.kg.save()
        return fetched

    def _resolve_id(self, name: str, fallback: str) -> str:
        sid = _sanitize_id(name)
        node = self.kg.get_paper(sid) or self.kg.get_concept(sid)
        if node:
            return node.id
        for n in self.kg._hive.nodes:
            if n.label.lower() == name.lower():
                return n.id
        for n in self.kg._hive.nodes:
            if name.lower() in n.label.lower() or n.label.lower() in name.lower():
                return n.id
        return sid or fallback

    def _analyze_text(
        self,
        text: str,
        title: str,
    ) -> dict[str, Any]:
        max_chars = 8000
        truncated = text[:max_chars]

        fast_prompt = (
            f"Paper: {title}\n\n"
            f"{truncated[:2000]}\n\n"
            'Extract up to 5 key tags (short keywords) as a JSON list: {"tags": [...]}'
        )
        tags_result = self.llm.extract_structured(
            fast_prompt,
            model=self.config.ollama_fast_model,
        )
        tags = tags_result.get("tags", [])

        main_prompt = (
            f"Title: {title}\n\n"
            f"{truncated}\n\n"
            "Extract the following as JSON. Do NOT include markdown formatting.\n"
            "{\n"
            '  "summary": "2-3 sentence summary",\n'
            '  "notes": "3-5 key technical insights or observations",\n'
            '  "experiment": {"methodology": "...", "dataset": "...", "setup": "..."},\n'
            '  "results": {"main_findings": "...", "metrics": {"metric_name": "value"}},\n'
            '  "concepts": [{"name": "...", "definition": "...", "relation": "introduces|uses|proposes|related_to"}],\n'
            '  "relations": [{"source": "...", "target": "...", "relation": "..."}]\n'
            "}"
        )
        analysis = self.llm.extract_structured(main_prompt)
        analysis["tags"] = tags
        return analysis

    def _write_note(
        self,
        paper_id: str,
        paper: PaperInfo,
        summary: str,
        tags: list[str],
        concepts: list[dict[str, Any]],
        notes: str = "",
        experiment: dict[str, Any] | None = None,
        results: dict[str, Any] | None = None,
    ) -> Path | None:
        vault = Path(self.config.vault_dir)
        vault.mkdir(parents=True, exist_ok=True)
        safe_title = _sanitize_id(paper.title) or paper_id
        path = vault / f"{safe_title}.md"
        lines = [
            "---",
            f"arxiv_id: {paper_id}",
            f"title: \"{paper.title}\"",
            f"authors: \"{paper.authors_str}\"",
            f"published: {paper.published}",
            f"tags: [{', '.join(tags)}]",
            "---",
            "",
        ]
        if summary:
            lines.extend(["## Summary", "", summary, ""])
        if notes:
            lines.extend(["## Notes", "", notes, ""])
        if experiment and isinstance(experiment, dict):
            exp = {}
            for k, v in experiment.items():
                if v is None or v == "":
                    continue
                if isinstance(v, list):
                    exp[k] = ", ".join(str(x) for x in v)
                elif isinstance(v, dict):
                    exp[k] = "; ".join(f"{sk}: {sv}" for sk, sv in v.items() if sv)
                else:
                    exp[k] = str(v)
            if exp:
                lines.extend(["## Experiment", ""])
                for k, v in exp.items():
                    lines.append(f"- **{k.capitalize()}**: {v}")
                lines.append("")
        if results and isinstance(results, dict):
            res_parts = []
            mf = results.get("main_findings")
            if mf:
                if isinstance(mf, list):
                    res_parts.extend(mf)
                else:
                    res_parts.append(str(mf))
            if results.get("metrics") and isinstance(results["metrics"], dict):
                m_items = []
                for mk, mv in results["metrics"].items():
                    if mv is None or mv == "":
                        continue
                    if isinstance(mv, (list, tuple)):
                        m_items.append(f"{mk}: {', '.join(str(x) for x in mv)}")
                    else:
                        m_items.append(f"{mk}: {mv}")
                if m_items:
                    res_parts.append("Metrics: " + " | ".join(m_items))
            if res_parts:
                lines.extend(["## Results", ""])
                for part in res_parts:
                    lines.append(part)
                lines.append("")
        if concepts:
            lines.extend(["## Concepts", ""])
            for c in concepts:
                name = c.get("name", c.get("label", ""))
                rel = c.get("relation", "")
                lines.append(f"- **{name}** ({rel})")
            lines.append("")
        lines.extend(["## Links", "", f"- [arXiv](https://arxiv.org/abs/{paper_id})"])
        if any(c.get("definition") for c in concepts):
            lines.extend(["", "## Definitions", ""])
            for c in concepts:
                if c.get("definition"):
                    lines.append(f"- **{c.get('name', c.get('label', ''))}**: {c['definition']}")
        safe_lines = [str(item) if not isinstance(item, str) else item for item in lines]
        with open(path, "w") as f:
            f.write("\n".join(safe_lines))
        return path
