"""Скоринг пула линейной смесью признаков, Recall@K и подбор весов."""
from __future__ import annotations

import numpy as np
import torch

from .pool import FEATURES


class PoolArrays:
    """Пул в виде плотных тензоров (n_queries x max_pool); пустые места помечены valid=False."""

    def __init__(self, pool: dict, item_ids: np.ndarray | None = None, gt: list[set] | None = None,
                 extra: dict | None = None):
        """extra: {имя: массив длины пула} — дополнительные признаки кандидатов (например, mc)."""
        qidx, doc, feats = pool["qidx"], pool["doc"], pool["feats"]
        self.names = list(FEATURES)
        if extra:
            feats = np.column_stack([feats] + [np.asarray(v, np.float32) for v in extra.values()])
            self.names += list(extra)
        n = int(pool["n_queries"])
        counts = np.bincount(qidx, minlength=n)
        m = int(counts.max())
        pos = np.arange(len(qidx)) - np.repeat(np.cumsum(counts) - counts, counts)
        self.F = torch.zeros((n, m, feats.shape[1]), dtype=torch.float32)
        self.F[qidx, pos] = torch.from_numpy(feats)
        self.doc = np.full((n, m), -1, dtype=np.int64)
        self.doc[qidx, pos] = doc
        self.valid = torch.from_numpy(self.doc >= 0)
        self.n = n
        self.q_weight = torch.ones(n) / n
        if gt is not None:
            cand_ids = np.where(self.doc >= 0, item_ids[np.clip(self.doc, 0, None)], None)
            rel = np.array([[x in gt[i] for x in cand_ids[i]] for i in range(n)], dtype=bool)
            self.rel = torch.from_numpy(rel)
            self.n_rel = torch.tensor([len(g) for g in gt], dtype=torch.float32)

    def set_query_weights(self, w: np.ndarray) -> None:
        """Веса запросов в метрике, например чтобы доля знакомых текстов совпала с бенчмарком."""
        w = torch.as_tensor(w, dtype=torch.float32)
        self.q_weight = w / w.sum()

    def score(self, w: dict) -> torch.Tensor:
        """w: веса признаков; geo/tau задают бонус geo * exp(-dist / tau)."""
        s = torch.zeros(self.F.shape[:2])
        for i, name in enumerate(self.names):
            if name != "dist" and w.get(name, 0.0):
                s = s + w[name] * self.F[..., i]
        if w.get("geo", 0.0):
            s = s + w["geo"] * torch.exp(-self.F[..., self.names.index("dist")] / w.get("tau", 50.0))
        return s.masked_fill(~self.valid, -1e9)

    def _top_idx(self, w: dict, k: int) -> torch.Tensor:
        return self.score(w).topk(min(k, self.F.shape[1]), dim=1).indices

    def top(self, w: dict, k: int = 50) -> np.ndarray:
        return np.take_along_axis(self.doc, self._top_idx(w, k).numpy(), 1)

    def per_query_recall(self, w: dict, k: int = 50) -> torch.Tensor:
        return torch.gather(self.rel, 1, self._top_idx(w, k)).sum(1).float() / self.n_rel

    def recall(self, w: dict, k: int = 50) -> float:
        return float((self.per_query_recall(w, k) * self.q_weight).sum())

    def pool_recall(self) -> float:
        return float(((self.rel.sum(1).float() / self.n_rel) * self.q_weight).sum())


def coordinate_ascent(pa: PoolArrays, w0: dict, grids: dict, rounds: int = 3, k: int = 50, verbose: bool = True):
    """Покоординатный подбор весов по сеткам grids, максимизируя Recall@k."""
    w, best = dict(w0), pa.recall(w0, k)
    if verbose:
        print(f"start R@{k}={best:.4f} {w}")
    for r in range(rounds):
        improved = False
        for key, grid in grids.items():
            for v in grid:
                cand = {**w, key: v}
                rec = pa.recall(cand, k)
                if rec > best + 1e-5:
                    best, w, improved = rec, cand, True
        if verbose:
            print(f"round {r}: R@{k}={best:.4f} {w}")
        if not improved:
            break
    return w, best
