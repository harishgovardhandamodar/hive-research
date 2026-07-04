from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .graph import KnowledgeGraph
from .llm import LLMInterface

logger = logging.getLogger(__name__)


class PaperVectorStore:
    def __init__(
        self,
        config: Config,
        llm: LLMInterface,
        kg: KnowledgeGraph,
    ) -> None:
        self.config = config
        self.llm = llm
        self.kg = kg
        self.store_dir = Path(config.root_dir) / "vectors"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._paper_ids: list[str] = []
        self._load()

    def _index_path(self) -> Path:
        return self.store_dir / "index.json"

    def _vectors_path(self) -> Path:
        return self.store_dir / "vectors.npy"

    def _save(self) -> None:
        with open(self._index_path(), "w") as f:
            json.dump({"paper_ids": self._paper_ids}, f)
        if self._vectors is not None:
            np.save(str(self._vectors_path()), self._vectors)

    def _load(self) -> None:
        if self._index_path().exists() and self._vectors_path().exists():
            try:
                with open(self._index_path()) as f:
                    data = json.load(f)
                self._paper_ids = data.get("paper_ids", [])
                self._vectors = np.load(str(self._vectors_path()))
                logger.info(
                    "Loaded %d paper vectors from %s",
                    len(self._paper_ids), self._vectors_path(),
                )
            except Exception as e:
                logger.warning("Failed to load paper vectors: %s", e)
                self._paper_ids = []
                self._vectors = None

    @property
    def ready(self) -> bool:
        return self._vectors is not None and len(self._paper_ids) > 0

    @property
    def dimension(self) -> int:
        if self._vectors is not None and self._vectors.shape[1]:
            return int(self._vectors.shape[1])
        return 0

    def compute_all(self) -> int:
        papers = self.kg.papers
        if not papers:
            logger.warning("No papers to compute vectors for")
            return 0

        ids: list[str] = []
        vecs: list[np.ndarray] = []
        for p in papers:
            text = f"{p.label or ''} {p.abstract or ''}"
            text = text.strip()
            if not text:
                continue
            try:
                raw = self.llm.embed(text[:2048])
                if raw and isinstance(raw, list):
                    ids.append(p.id)
                    vecs.append(np.array(raw, dtype=np.float32))
            except Exception as e:
                logger.warning("Failed to embed %s: %s", p.id, e)

        if not vecs:
            logger.warning("No vectors computed")
            return 0

        self._paper_ids = ids
        self._vectors = np.stack(vecs)
        self._save()
        logger.info("Computed and saved %d paper vectors (dim=%d)", len(ids), self.dimension)
        return len(ids)

    def get_index(self, paper_id: str) -> int | None:
        try:
            return self._paper_ids.index(paper_id)
        except ValueError:
            return None

    def get_vector(self, paper_id: str) -> np.ndarray | None:
        idx = self.get_index(paper_id)
        if idx is not None and self._vectors is not None:
            return self._vectors[idx]
        return None

    def cosine(self, pid_a: str, pid_b: str) -> float:
        if not self.ready:
            return 0.0
        ia, ib = self.get_index(pid_a), self.get_index(pid_b)
        if ia is None or ib is None or self._vectors is None:
            return 0.0
        va = self._vectors[ia]
        vb = self._vectors[ib]
        dot = float(va @ vb)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _proj_path(self) -> Path:
        return self.store_dir / "projection.npy"

    def project(self) -> list[dict[str, Any]]:
        if not self.ready or self._vectors is None:
            return []
        # Try loading cached projection
        if self._proj_path().exists():
            try:
                proj = np.load(str(self._proj_path()))
                if proj.shape[0] == len(self._paper_ids) and proj.shape[1] == 2:
                    results = []
                    for i, pid in enumerate(self._paper_ids):
                        n = self._find_node(pid)
                        results.append({
                            "id": pid,
                            "x": round(float(proj[i, 0]), 4),
                            "y": round(float(proj[i, 1]), 4),
                            "label": n.label if n else pid,
                            "authors": getattr(n, "authors", "")[:60] if n else "",
                        })
                    return results
            except Exception:
                pass

        # Compute PCA via SVD
        X = self._vectors - self._vectors.mean(axis=0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        proj = U[:, :2] * S[:2]

        # Cache
        np.save(str(self._proj_path()), proj)

        results = []
        for i, pid in enumerate(self._paper_ids):
            n = self._find_node(pid)
            results.append({
                "id": pid,
                "x": round(float(proj[i, 0]), 4),
                "y": round(float(proj[i, 1]), 4),
                "label": n.label if n else pid,
                "authors": getattr(n, "authors", "")[:60] if n else "",
            })
        return results

    def clear(self) -> None:
        self._vectors = None
        self._paper_ids = []
        for p in [self._index_path(), self._vectors_path(), self._proj_path()]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("Paper vectors cleared")

    def _find_node(self, pid: str) -> Any:
        for p in self.kg.papers:
            if p.id == pid:
                return p
        return None

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "count": len(self._paper_ids),
            "dimension": self.dimension,
            "path": str(self._vectors_path()),
        }
