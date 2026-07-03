from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .arxiv_fetcher import PaperInfo, download_pdf, fetch_by_id
from .config import Config
from .graph import KnowledgeGraph
from .llm import LLMInterface
from .parser import extract_sections, extract_text

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

        for tag in tags:
            tag_id = _sanitize_id(tag)
            matched = self.kg.find_similar_concept(tag)
            if matched:
                cid = matched.id
            else:
                cid = tag_id
                self.kg.add_concept(cid, tag, concept_type="tag")
            self.kg.add_edge(paper_id, cid, "related_to")

        for c in concepts:
            cid = _sanitize_id(c.get("name", "")) or _sanitize_id(c.get("label", ""))
            if not cid:
                continue
            label = c.get("name", c.get("label", cid))
            matched = self.kg.find_similar_concept(label)
            if matched:
                cid = matched.id
            else:
                self.kg.add_concept(
                    cid,
                    label,
                    definition=c.get("definition", ""),
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

        note_path = self._write_note(paper_id, paper, summary, tags, concepts)
        self.kg.save()

        return {
            "status": "added",
            "paper_id": paper_id,
            "concepts": len(concepts),
            "tags": len(tags),
            "relations": len(relations),
            "note_path": str(note_path) if note_path else None,
        }

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
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path
