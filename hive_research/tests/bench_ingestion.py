#!/usr/bin/env python3
"""
End-to-end ingestion benchmark for hive-research.

Measures per-stage latency for the full pipeline:
arXiv fetch → PDF download → text extraction → LLM analysis →
knowledge graph population → RAG indexing.

Usage:
    python -m hive_research.tests.bench_ingestion
    python -m hive_research.tests.bench_ingestion --arxiv 2409.13004
    python -m hive_research.tests.bench_ingestion --url https://arxiv.org/abs/2409.13004v1
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bench")


ARXIV_ID_RX = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")


def parse_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RX.search(text)
    return m.group(1) if m else None


@contextmanager
def timed(label: str) -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    t0 = time.perf_counter()
    yield result
    result["elapsed"] = time.perf_counter() - t0
    result["label"] = label


def fmt(t: float) -> str:
    if t < 1:
        return f"{t * 1000:.0f} ms"
    if t < 60:
        return f"{t:.1f} s"
    return f"{t / 60:.1f} m"


BAR = "\u2500" * 60


def run_benchmark(arxiv_id: str, temp_root: Path | None = None) -> dict[str, Any]:
    # ── Setup: isolated temp directory ──────────────────────────────
    own_temp = temp_root is None
    if temp_root is None:
        temp_root = Path("/tmp") / f"hive_bench_{arxiv_id.replace('.', '_')}"

    if temp_root.exists():
        shutil.rmtree(temp_root)

    (temp_root / "papers").mkdir(parents=True)
    (temp_root / "graph").mkdir(parents=True)
    (temp_root / "vault").mkdir(parents=True)
    (temp_root / "rag").mkdir(parents=True)

    # Isolated config that points at temp dirs
    from hive_research.config import Config

    class BenchConfig(Config):
        def __init__(self) -> None:
            self.data = {
                "directories": {
                    "root": str(temp_root),
                    "papers": str(temp_root / "papers"),
                    "graph": str(temp_root / "graph"),
                    "vault": str(temp_root / "vault"),
                },
                "arxiv": {"download_pdf": True, "max_results": 10},
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3.2:3b",
                    "fast_model": "llama3.2:3b",
                    "embed_model": "nomic-embed-text",
                    "max_tokens": 4096,
                    "temperature": 0.1,
                },
                "graph": {"similarity_threshold": 0.85},
                "rag": {"chunk_size": 512, "chunk_overlap": 64, "top_k": 5},
                "server": {"host": "127.0.0.1", "port": 8080},
            }

    config: Config = BenchConfig()

    # Import lazily so we can measure imports separately if needed
    from hive_research.arxiv_fetcher import PaperInfo, fetch_by_id, download_pdf
    from hive_research.parser import extract_text
    from hive_research.llm import LLMInterface
    from hive_research.graph import KnowledgeGraph
    from hive_research.pipeline import PaperPipeline
    from hive_research.rag import RAGEngine

    llm = LLMInterface(config)
    kg = KnowledgeGraph(config, graph_id="bench")
    pipeline = PaperPipeline(config, llm, kg)
    rag = RAGEngine(config, llm, kg)

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # ── 1. arXiv metadata fetch ─────────────────────────────────────
    print(f"\n  Benchmarking ingestion of arXiv:{arxiv_id}\n")
    print(BAR)
    paper: PaperInfo | None = None

    with timed("1. arXiv metadata fetch") as t:
        paper = fetch_by_id(arxiv_id)
    timings["arxiv_fetch"] = t["elapsed"]
    print(f"  {t['label']:45s} {fmt(t['elapsed'])}")

    if paper is None:
        print(f"\n  ERROR: Paper {arxiv_id} not found on arXiv.\n")
        if own_temp:
            shutil.rmtree(temp_root)
        return {"status": "error", "message": "paper not found"}

    print(f"  {'':45s} Title: {paper.title[:80]}")

    # ── 2. PDF download ──────────────────────────────────────────────
    pdf_path: Path | None = None
    with timed("2. PDF download") as t:
        pdf_path = download_pdf(arxiv_id, config.papers_dir)
    timings["pdf_download"] = t["elapsed"]
    print(f"  {t['label']:45s} {fmt(t['elapsed'])}")

    if pdf_path is None or not pdf_path.exists():
        print(f"  {'':45s} (no PDF — falling back to abstract-only)")
    else:
        pdf_size = pdf_path.stat().st_size
        print(f"  {'':45s} {pdf_size / 1024:.0f} KB on disk")

    # ── 3. PDF text extraction ──────────────────────────────────────
    pdf_text = ""
    with timed("3. PDF text extraction") as t:
        if pdf_path and pdf_path.exists():
            pdf_text = extract_text(pdf_path)
    timings["text_extract"] = t["elapsed"]
    if pdf_text:
        n_chars = len(pdf_text)
        n_words = len(pdf_text.split())
        extra = f"{n_chars:,} chars, {n_words:,} words"
    else:
        extra = "fallback: using abstract only"
    print(f"  {t['label']:45s} {fmt(t['elapsed'])} ({extra})")

    text_for_analysis = pdf_text or paper.abstract

    # ── 4. LLM tag extraction (fast model) ──────────────────────────
    tags: list[str] = []
    with timed("4. LLM tag extraction (fast model)") as t:
        fast_prompt = (
            f"Paper: {paper.title}\n\n"
            f"{text_for_analysis[:2000]}\n\n"
            'Extract up to 5 key tags (short keywords) as a JSON list: {"tags": [...]}'
        )
        tags_result = llm.extract_structured(
            fast_prompt, model=config.ollama_fast_model
        )
        tags = tags_result.get("tags", [])
    timings["llm_tags"] = t["elapsed"]
    print(f"  {t['label']:45s} {fmt(t['elapsed'])} ({len(tags)} tags)")

    # ── 5. LLM concept/relation extraction (main model) ─────────────
    concepts: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    summary = ""

    truncated = text_for_analysis[:8000]
    with timed("5. LLM concept/relation extraction (main model)") as t:
        main_prompt = (
            f"Title: {paper.title}\n\n"
            f"{truncated}\n\n"
            "Extract the following as JSON. Do NOT include markdown formatting.\n"
            "{\n"
            '  "summary": "2-3 sentence summary",\n'
            '  "concepts": [{"name": "...", "definition": "...", "relation": "introduces|uses|proposes|related_to"}],\n'
            '  "relations": [{"source": "...", "target": "...", "relation": "..."}]\n'
            "}"
        )
        analysis = llm.extract_structured(main_prompt)
        concepts = analysis.get("concepts", [])
        relations = analysis.get("relations", [])
        summary = analysis.get("summary", "")
    timings["llm_concepts"] = t["elapsed"]
    print(f"  {t['label']:45s} {fmt(t['elapsed'])} ({len(concepts)} concepts, {len(relations)} relations)")

    # ── 6. Paper node + graph population ────────────────────────────
    with timed("6. Knowledge graph population") as t:
        node = kg.add_paper(
            paper_id=arxiv_id,
            title=paper.title,
            authors=paper.authors_str,
            published=paper.published,
            abstract=paper.abstract,
            categories=paper.categories,
        )
        for tag in tags:
            from hive_research.pipeline import _sanitize_id
            tag_id = _sanitize_id(tag)
            matched = kg.find_similar_concept(tag)
            cid = matched.id if matched else tag_id
            if not matched:
                kg.add_concept(cid, tag, concept_type="tag")
            kg.add_edge(arxiv_id, cid, "related_to")
        for c in concepts:
            cid = _sanitize_id(c.get("name", "")) or _sanitize_id(c.get("label", ""))
            if not cid:
                continue
            label = c.get("name", c.get("label", cid))
            matched = kg.find_similar_concept(label)
            if matched:
                cid = matched.id
            else:
                kg.add_concept(
                    cid, label,
                    definition=c.get("definition", ""),
                    concept_type=c.get("type", "concept"),
                )
            rel = c.get("relation", "related_to")
            kg.add_edge(arxiv_id, cid, rel)
        for r in relations:
            src = r.get("source", arxiv_id)
            tgt = r.get("target", "")
            rel = r.get("relation", "related_to")
            if src and tgt:
                kg.add_edge(src, tgt, rel)
        kg.save()
    timings["graph_populate"] = t["elapsed"]
    stats = kg.stats()
    print(f"  {t['label']:45s} {fmt(t['elapsed'])} (graph: {stats['papers']}P/{stats['concepts']}C/{stats['relations']}E)")

    # ── 7. Note writing ─────────────────────────────────────────────
    from hive_research.pipeline import _sanitize_id as si
    note_path: Path | None = None
    with timed("7. Note writing") as t:
        vault = Path(config.vault_dir)
        safe_title = si(paper.title) or arxiv_id
        note_path = vault / f"{safe_title}.md"
        lines = [
            "---",
            f"arxiv_id: {arxiv_id}",
            f"title: \"{paper.title}\"",
            f"authors: \"{paper.authors_str}\"",
            f"published: {paper.published}",
            f"tags: [{', '.join(tags)}]",
            "---", "",
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
        lines.extend(["## Links", "", f"- [arXiv](https://arxiv.org/abs/{arxiv_id})"])
        with open(note_path, "w") as f:
            f.write("\n".join(lines))
    timings["note_write"] = t["elapsed"]
    print(f"  {t['label']:45s} {fmt(t['elapsed'])} ({note_path.name})")

    # ── 8. RAG indexing (chunk + embed) ─────────────────────────────
    n_chunks = 0
    if pdf_text:
        with timed("8. RAG indexing (chunk + embed)") as t:
            n_chunks = rag.index_paper(arxiv_id, pdf_text)
        timings["rag_index"] = t["elapsed"]
        extra = f"{n_chunks} chunks"
    else:
        with timed("8. RAG indexing (skipped — no PDF)") as t:
            pass
        timings["rag_index"] = 0.0
        extra = "skipped (no PDF)"
    print(f"  {t['label']:45s} {fmt(timings['rag_index'])} ({extra})")

    # ── 9. RAG query validation ────────────────────────────────────
    if n_chunks > 0:
        with timed("9. RAG search (embed query + cosine)") as t:
            results = rag.search("What is this paper about?")
        timings["rag_search"] = t["elapsed"]
        print(f"  {t['label']:45s} {fmt(t['elapsed'])} ({len(results)} results)")
    else:
        timings["rag_search"] = 0.0

    total_elapsed = time.perf_counter() - total_start
    timings["total"] = total_elapsed

    # ── Summary ─────────────────────────────────────────────────────
    print(BAR)
    print(f"\n  {'TOTAL':45s} {fmt(total_elapsed)}")
    print()

    breakdown = {
        "arxiv_fetch": timings["arxiv_fetch"],
        "pdf_download": timings["pdf_download"],
        "text_extract": timings["text_extract"],
        "llm_tags": timings["llm_tags"],
        "llm_concepts": timings["llm_concepts"],
        "graph_populate": timings["graph_populate"],
        "note_write": timings["note_write"],
        "rag_index": timings["rag_index"],
        "rag_search": timings["rag_search"],
        "total": total_elapsed,
    }

    result = {
        "status": "ok",
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": paper.authors_str,
        "pdf_size_kb": round(pdf_size / 1024, 1) if pdf_path and pdf_path.exists() else 0,
        "text_chars": n_chars if pdf_text else 0,
        "text_words": n_words if pdf_text else 0,
        "tags": tags,
        "num_concepts": len(concepts),
        "num_relations": len(relations),
        "graph": stats,
        "rag_chunks": n_chunks,
        "timings": breakdown,
    }

    # ── Cleanup ─────────────────────────────────────────────────────
    if own_temp:
        shutil.rmtree(temp_root)

    return result


def print_table(results: dict[str, Any]) -> None:
    t = results.get("timings", {})
    print("\n  Stage Timing Table")
    print(f"  {'─' * 54}")
    print(f"  {'Stage':<42s} {'Duration':>10s}")
    print(f"  {'─' * 54}")

    stages = [
        ("arXiv metadata fetch", "arxiv_fetch"),
        ("PDF download", "pdf_download"),
        ("Text extraction", "text_extract"),
        ("Tag extraction (fast LLM)", "llm_tags"),
        ("Concept extraction (main LLM)", "llm_concepts"),
        ("Graph population", "graph_populate"),
        ("Note writing", "note_write"),
        ("RAG indexing", "rag_index"),
        ("RAG search", "rag_search"),
        ("TOTAL", "total"),
    ]
    for label, key in stages:
        val = t.get(key, 0.0)
        pct = (val / t.get("total", 1)) * 100 if t.get("total", 0) > 0 else 0
        print(f"  {label:<42s} {fmt(val):>8s}  ({pct:4.1f}%)")

    print(f"  {'─' * 54}")
    print(f"  Paper: {results.get('title', '?')}")
    print(f"  Graph: {results.get('graph', {}).get('papers', 0)} papers, "
          f"{results.get('graph', {}).get('concepts', 0)} concepts, "
          f"{results.get('graph', {}).get('relations', 0)} edges")
    print(f"  RAG:   {results.get('rag_chunks', 0)} chunks indexed\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end ingestion benchmark for hive-research"
    )
    parser.add_argument("--arxiv", type=str, default=None,
                        help="arXiv ID (e.g. 2409.13004)")
    parser.add_argument("--url", type=str, default=None,
                        help="arXiv URL (e.g. https://arxiv.org/abs/2409.13004v1)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temp directory after run")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON report instead of table")
    args = parser.parse_args()

    raw = args.arxiv or args.url or "2409.13004"
    arxiv_id = parse_arxiv_id(raw)
    if arxiv_id is None:
        print(f"Could not parse arXiv ID from '{raw}'")
        sys.exit(1)

    temp_root: Path | None = None
    if args.keep_temp:
        temp_root = Path("/tmp") / f"hive_bench_{arxiv_id.replace('.', '_')}"

    try:
        results = run_benchmark(arxiv_id, temp_root=temp_root)
    except Exception as e:
        logger.exception("Benchmark failed")
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_table(results)

    if temp_root and temp_root.exists():
        print(f"  Temp data preserved at: {temp_root}\n")


if __name__ == "__main__":
    main()
