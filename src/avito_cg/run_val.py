"""Офлайн-валидация: пул кандидатов на локальном корпусе и подбор весов смеси.

uv run python -m avito_cg.run_val                  # сплит по тексту, один запрос на текст
uv run python -m avito_cg.run_val --split key      # сплит по полной комбинации search_*, как в бенчмарке
uv run python -m avito_cg.run_val --all-instances  # все запросы, а не один на текст
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .channels import ChannelIndex
from .data import CACHE_DIR, build_local, one_per_text
from .pool import build_pool, load_pool, location_centroids, save_pool
from .prior import QueryNeighbors
from .text import normalize
from .tune import PoolArrays, coordinate_ascent

SEEN_SHARE = 0.37  # доля текстов бенчмарка, которые встречаются в train

W0 = {"tw": 0.0, "tc": 0.5, "bm": 2.0, "sp": 0.5, "e5": 5.0, "hx": 1.0,
      "loc": 1.0, "geo": 1.5, "tau": 300, "mc": 0.5}
GRIDS = {
    "tw": [0, 0.25, 0.5, 1.0, 1.5],
    "tc": [0, 0.25, 0.5, 1.0, 1.5],
    "bm": [0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "sp": [0, 0.1, 0.25, 0.5, 1.0],
    "e5": [0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "hx": [0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "mc": [0, 0.1, 0.25, 0.5, 1.0],
    "loc": [0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "geo": [0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "tau": [30, 100, 200, 300, 500, 1000],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["text", "key"], default="text")
    ap.add_argument("--all-instances", action="store_true", help="все запросы вместо одного на текст")
    ap.add_argument("--rebuild", action="store_true", help="пересчитать пул кандидатов")
    args = ap.parse_args()

    d = build_local(split_by=args.split)
    corpus = d["corpus"]
    queries = d["val_queries"] if args.all_instances else one_per_text(d["val_queries"])
    tag = f"{args.split}_{'all' if args.all_instances else 'opt'}"
    path = CACHE_DIR / f"pool_val_{tag}.npz"

    t = time.time()
    index = ChannelIndex(corpus, "local")
    nb = QueryNeighbors(d["train_pairs"], {"ids": corpus["item_id"].to_numpy(), "emb": index.E.numpy()})
    W = nb.weights(queries)
    print(f"index+neighbors {time.time() - t:.0f}s")
    if args.rebuild or not path.exists():
        t = time.time()
        save_pool(build_pool(index, corpus, queries, location_centroids(corpus), hist=nb.hist_vectors(W)), path)
        print(f"pool {time.time() - t:.0f}s")
    del index
    pool = load_pool(path, n_queries=len(queries), n_items=len(corpus))

    gt = [d["val_gt"][q] for q in queries["query_id"]]
    mc = nb.mc_feature(W, pool, corpus["item_microcat_id"].to_numpy())
    pa = PoolArrays(pool, corpus["item_id"].to_numpy(), gt, extra={"mc": mc})
    print(f"queries {pa.n}, mean pool {len(pool['doc']) / pa.n:.0f}, pool recall {pa.pool_recall():.4f}")
    for ch in ["tw", "tc", "bm", "sp", "e5", "hx", "mc"]:
        print(f"  only {ch}+loc: R@50={pa.recall({ch: 1.0, 'loc': 1.0}):.4f}")

    is_seen = None
    if args.split == "key":
        seen = set(d["train_pairs"]["search_query"].astype(str).map(normalize))
        is_seen = queries["search_query"].astype(str).map(normalize).isin(seen).to_numpy()
        share = is_seen.mean()
        pa.set_query_weights(np.where(is_seen, SEEN_SHARE / share, (1 - SEEN_SHARE) / (1 - share)))
        print(f"pool recall (reweighted to {SEEN_SHARE:.0%} seen) {pa.pool_recall():.4f}")

    w, best = coordinate_ascent(pa, W0, GRIDS)
    print(f"best R@50={best:.4f}  R@100={pa.recall(w, 100):.4f}  R@200={pa.recall(w, 200):.4f}")
    for key in [k for k in w if k != "tau" and w[k]]:
        print(f"  without {key}: R@50={pa.recall({**w, key: 0}):.4f}")

    if is_seen is not None:
        per_q = pa.per_query_recall(w).numpy()
        print(f"  seen texts {is_seen.mean():.1%}: R@50 seen={per_q[is_seen].mean():.4f} "
              f"unseen={per_q[~is_seen].mean():.4f}")
    (CACHE_DIR / f"weights_{tag}.json").write_text(json.dumps(w, indent=1))


if __name__ == "__main__":
    main()
