"""Пул кандидатов: объединение топов всех каналов с точными скорами каждого канала.

Для запроса пул = top-K_CH каждого канала глобально + top-K_CH каждого канала внутри локации запроса
+ top-K_MIX базовой смеси (с гео-бонусом и без). Для каждого кандидата сохраняются скоры всех каналов,
совпадение локации и расстояние, поэтому веса смеси потом подбираются без пересчёта каналов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .channels import CHANNELS, ChannelIndex

FEATURES = CHANNELS + ["loc", "dist"]
BASE_WEIGHTS = {"tw": 0.5, "tc": 0.5, "bm": 1.0, "sp": 0.5, "e5": 1.0, "hx": 1.0, "loc": 1.0}
EARTH_RADIUS_KM = 6371.0
NO_COORDS_KM = 20000.0


def location_centroids(*item_frames: pd.DataFrame) -> pd.DataFrame:
    """Медианные координаты объявлений каждой локации."""
    items = pd.concat([f[["item_location_id", "item_latitude", "item_longitude"]] for f in item_frames])
    return items.dropna().groupby("item_location_id")[["item_latitude", "item_longitude"]].median()


def _item_coords(corpus: pd.DataFrame, centroids: pd.DataFrame) -> np.ndarray:
    lat = corpus["item_latitude"].to_numpy(np.float64)
    lon = corpus["item_longitude"].to_numpy(np.float64)
    bad = np.isnan(lat) | np.isnan(lon)
    if bad.any():
        c = centroids.reindex(corpus.loc[bad, "item_location_id"])
        lat[bad], lon[bad] = c["item_latitude"].to_numpy(), c["item_longitude"].to_numpy()
    return np.radians(np.stack([lat, lon], 1)).astype(np.float32)


def _haversine(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """q: (B, 2), d: (N, 2) в радианах -> (B, N) км."""
    dlat = d[None, :, 0] - q[:, None, 0]
    dlon = d[None, :, 1] - q[:, None, 1]
    a = torch.sin(dlat / 2) ** 2 + torch.cos(q[:, None, 0]) * torch.cos(d[None, :, 0]) * torch.sin(dlon / 2) ** 2
    km = 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))
    return torch.nan_to_num(km, nan=NO_COORDS_KM)


def build_pool(index: ChannelIndex, corpus: pd.DataFrame, queries: pd.DataFrame, centroids: pd.DataFrame,
               hist: np.ndarray | None = None, k_ch: int = 100, k_mix: int = 300, batch: int = 128) -> dict:
    Q = index.encode_queries(queries, hist)
    item_loc = torch.from_numpy(corpus["item_location_id"].to_numpy(np.int64))
    item_xy = torch.from_numpy(_item_coords(corpus, centroids))
    qc = centroids.reindex(queries["search_location_id"].to_numpy())
    q_xy = torch.from_numpy(np.radians(qc[["item_latitude", "item_longitude"]].to_numpy(np.float64)).astype(np.float32))
    q_loc = torch.from_numpy(queries["search_location_id"].to_numpy(np.int64))

    qid_parts, doc_parts, feat_parts = [], [], []
    n = len(queries)
    for start in range(0, n, batch):
        rows = slice(start, min(start + batch, n))
        S = index.scores(Q, rows)
        S["bm"] = S["bm"] / S["bm"].amax(1, keepdim=True).clamp_min(1e-6)
        loc = (item_loc[None, :] == q_loc[rows, None]).float()
        dist = _haversine(q_xy[rows], item_xy)

        cand = []
        for ch in CHANNELS:
            cand.append(S[ch].topk(k_ch, dim=1).indices)
            cand.append((S[ch] + 10.0 * loc).topk(k_ch, dim=1).indices)
        mix = sum(w * S[ch] for ch, w in BASE_WEIGHTS.items() if ch != "loc") + BASE_WEIGHTS["loc"] * loc
        cand.append(mix.topk(k_mix, dim=1).indices)
        cand.append((mix + torch.exp(-dist / 300.0)).topk(k_mix, dim=1).indices)
        cand = torch.cat(cand, 1).numpy()

        for i in range(cand.shape[0]):
            docs = torch.from_numpy(np.unique(cand[i]))
            f = torch.stack([S[ch][i, docs] for ch in CHANNELS] + [loc[i, docs], dist[i, docs]], 1)
            qid_parts.append(np.full(len(docs), start + i, dtype=np.int32))
            doc_parts.append(docs.numpy().astype(np.int32))
            feat_parts.append(f.numpy().astype(np.float32))
        if (start // batch) % 10 == 0:
            print(f"  pool {min(start + batch, n)}/{n}", flush=True)

    return {
        "qidx": np.concatenate(qid_parts),
        "doc": np.concatenate(doc_parts),
        "feats": np.concatenate(feat_parts),
        "n_queries": n,
    }


def save_pool(pool: dict, path) -> None:
    np.savez(path, **pool)


def load_pool(path, n_queries: int | None = None, n_items: int | None = None) -> dict:
    """Загрузка кэша с проверкой, что пул построен для тех же запросов и корпуса."""
    z = np.load(path)
    pool = {k: z[k] for k in z.files}
    stale = (n_queries is not None and int(pool["n_queries"]) != n_queries) or (
        n_items is not None and int(pool["doc"].max()) >= n_items)
    if stale:
        raise ValueError(f"Кэш пула {path} построен для других данных — перезапустите с --rebuild")
    return pool
