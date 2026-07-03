from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .arxiv_fetcher import search_arxiv
from .similarity import jaccard_tokens

logger = logging.getLogger(__name__)

CACHE_TTL = 12 * 3600
REFRESH_INTERVAL = 12 * 3600
MAX_PER_TOPIC = 10

DEFAULT_TOPICS = [
    {"name": "Knowledge graphs", "query": "knowledge graph embedding"},
    {"name": "Federated learning", "query": "federated learning"},
    {"name": "AI security", "query": "AI security adversarial machine learning"},
    {"name": "LLM security", "query": "large language model security"},
    {"name": "AI alignment", "query": "AI alignment"},
    {"name": "Adversarial ML", "query": "adversarial machine learning"},
    {"name": "Graph neural networks", "query": "graph neural network"},
    {"name": "Vision-language models", "query": "vision language model"},
]

TOPIC_COLORS = [
    "#60a5fa", "#34d399", "#fbbf24", "#f87171",
    "#c084fc", "#22d3ee", "#fb923c", "#a78bfa",
]


class ResearchPool:
    def __init__(self, store_dir: str | Path) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._topics_path = self.store_dir / "pool_topics.json"
        self._cache_path = self.store_dir / "pool_cache.json"
        self._store_path = self.store_dir / "pool_store.json"

        self._lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._topics_lock = threading.Lock()

        self._topics: list[dict[str, Any]] = self._load_topics()
        self._store: dict[str, Any] = self._load_store()
        self._cache: dict[str, Any] = self._load_cache()

        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    # ── Topic management ───────────────────────────────────────────

    def get_topics(self) -> list[dict[str, Any]]:
        with self._topics_lock:
            return [dict(t) for t in self._topics]

    def add_topic(self, name: str, query: str, **kwargs: Any) -> None:
        with self._topics_lock:
            self._topics = [t for t in self._topics if t.get("name") != name]
            topic: dict[str, Any] = {"name": name, "query": query}
            topic.update(kwargs)
            self._topics.append(topic)
            self._save_topics()

    def remove_topic(self, name: str) -> None:
        with self._topics_lock:
            self._topics = [t for t in self._topics if t.get("name") != name]
            self._save_topics()

    def _load_topics(self) -> list[dict[str, Any]]:
        if self._topics_path.exists():
            try:
                with open(self._topics_path) as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        return data
            except Exception:
                pass
        return [dict(t) for t in DEFAULT_TOPICS]

    def _save_topics(self) -> None:
        tmp = self._topics_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._topics, f, indent=2)
        os.replace(tmp, self._topics_path)

    # ── arXiv feed (cached) ────────────────────────────────────────

    def get(self) -> dict[str, Any]:
        with self._lock:
            ts = self._cache.get("timestamp", 0)
            age = time.time() - ts
            if age < CACHE_TTL and self._cache.get("data"):
                return self._cache["data"]
        if age > CACHE_TTL:
            self._bg_refresh()
        return self._cache.get("data", {})

    def refresh(self) -> dict[str, Any]:
        return self._do_refresh()

    def _bg_refresh(self) -> None:
        threading.Thread(target=self._do_refresh, daemon=True).start()

    def _do_refresh(self) -> dict[str, Any]:
        try:
            data = self._fetch_all()
            with self._lock:
                self._cache = {"timestamp": time.time(), "data": data}
                self._save_cache()
            return data
        except Exception as e:
            logger.error("Pool refresh failed: %s", e)
            return self._cache.get("data", {})

    def _fetch_all(self) -> dict[str, Any]:
        topics = self.get_topics()
        result: dict[str, Any] = {}
        for i, topic in enumerate(topics):
            name = topic.get("name", "untitled")
            query = topic.get("query", "")
            if i > 0:
                time.sleep(4)
            try:
                papers = search_arxiv(query, max_results=MAX_PER_TOPIC)
                entries = []
                for p in papers:
                    entry = {
                        "arxiv_id": p.arxiv_id,
                        "title": p.title,
                        "authors": p.authors,
                        "authors_str": p.authors_str,
                        "published": p.published,
                        "abstract": p.abstract[:500],
                        "categories": p.categories,
                        "pdf_url": p.pdf_url,
                    }
                    entries.append(entry)
                    self._observe(entry, name)
                result[name] = entries
                logger.info("Pool topic '%s': %d papers", name, len(entries))
            except Exception as e:
                logger.warning("Pool topic '%s' fetch failed: %s", name, e)
                result[name] = []
        self._save_store()
        return result

    def _bg_loop(self) -> None:
        time.sleep(30)
        self._bg_refresh()
        while True:
            time.sleep(REFRESH_INTERVAL)
            self._bg_refresh()

    # ── Observed papers store ──────────────────────────────────────

    def _observe(self, entry: dict[str, Any], topic: str) -> None:
        aid = entry["arxiv_id"]
        now = datetime.utcnow().isoformat()
        with self._store_lock:
            if aid in self._store:
                rec = self._store[aid]
                if topic not in rec.get("topics", []):
                    rec.setdefault("topics", []).append(topic)
                rec["last_seen"] = now
            else:
                self._store[aid] = {
                    "arxiv_id": aid,
                    "title": entry.get("title", ""),
                    "authors": entry.get("authors", []),
                    "authors_str": entry.get("authors_str", ""),
                    "published": entry.get("published", ""),
                    "abstract": entry.get("abstract", "")[:500],
                    "categories": entry.get("categories", []),
                    "pdf_url": entry.get("pdf_url", ""),
                    "topics": [topic],
                    "tags": [],
                    "imported": False,
                    "imported_at": None,
                    "first_seen": now,
                    "last_seen": now,
                }

    def get_observed_papers(self) -> list[dict[str, Any]]:
        with self._store_lock:
            papers = list(self._store.values())
        papers.sort(key=lambda p: p.get("last_seen", ""), reverse=True)
        now_ts = time.time()
        for p in papers:
            fs = p.get("first_seen")
            try:
                p["is_new"] = fs and (now_ts - datetime.fromisoformat(fs).timestamp()) < 86400
            except Exception:
                p["is_new"] = False
        return papers

    def mark_imported(self, arxiv_id: str) -> None:
        with self._store_lock:
            if arxiv_id in self._store:
                self._store[arxiv_id]["imported"] = True
                self._store[arxiv_id]["imported_at"] = datetime.utcnow().isoformat()
                self._save_store()

    def update_tags(self, arxiv_id: str, tags: list[str]) -> None:
        with self._store_lock:
            if arxiv_id in self._store:
                self._store[arxiv_id]["tags"] = tags
                self._save_store()

    # ── Pool graph (Jaccard similarity) ────────────────────────────

    def get_pool_graph(self) -> dict[str, Any]:
        papers = self.get_observed_papers()
        nodes = []
        for p in papers:
            nodes.append({
                "id": p["arxiv_id"],
                "label": p.get("title", "")[:60],
                "type": "paper",
                "title": p.get("title", ""),
                "abstract": (p.get("abstract", "") or "")[:300],
                "imported": p.get("imported", False),
                "is_new": p.get("is_new", False),
                "topics": p.get("topics", []),
            })
        edges = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                ti = (nodes[i]["title"] or "") + " " + (nodes[i]["abstract"] or "")
                tj = (nodes[j]["title"] or "") + " " + (nodes[j]["abstract"] or "")
                score = jaccard_tokens(ti, tj)
                if score >= 0.12:
                    edges.append({
                        "source": nodes[i]["id"],
                        "target": nodes[j]["id"],
                        "similarity": round(score, 4),
                    })
        return {"nodes": nodes, "edges": edges}

    # ── Persistence ────────────────────────────────────────────────

    def _load_store(self) -> dict[str, Any]:
        if self._store_path.exists():
            try:
                with open(self._store_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_store(self) -> None:
        tmp = self._store_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._store, f, indent=2)
        os.replace(tmp, self._store_path)

    def _load_cache(self) -> dict[str, Any]:
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"timestamp": 0, "data": {}}

    def _save_cache(self) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._cache, f, indent=2)
        os.replace(tmp, self._cache_path)
