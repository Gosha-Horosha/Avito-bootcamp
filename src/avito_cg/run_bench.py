"""Генерация answer.csv: запросы benchmark_queries, корпус benchmark_items, веса из configs/weights.json.

uv run python -m avito_cg.run_bench
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from . import e5
from .channels import ChannelIndex
from .data import CACHE_DIR, DATA_DIR, ROOT, build_local, load_benchmark
from .pool import build_pool, load_pool, location_centroids, save_pool
from .prior import QueryNeighbors
from .tune import PoolArrays

K = 50


def validate_answer(path: Path, sep: str) -> None:
    """Проверка формата: все query_id, 2 колонки, до 50 уникальных существующих item_id в строке."""
    queries = pd.read_parquet(DATA_DIR / "benchmark_queries.parquet", columns=["query_id"])
    items = set(pd.read_parquet(DATA_DIR / "benchmark_items.parquet", columns=["item_id"])["item_id"].astype(str))
    ans = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(ans.columns) == ["query_id", "answer"], ans.columns
    assert ans["query_id"].is_unique
    assert set(ans["query_id"]) == set(queries["query_id"].astype(str)), "не все query_id"
    for a in ans["answer"]:
        ids = [x for x in a.split(sep) if x]
        assert 0 < len(ids) <= K and len(set(ids)) == len(ids), a[:80]
        assert all(x in items for x in ids), "неизвестный item_id"
    print(f"OK: {len(ans)} строк, формат корректен ({path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "configs" / "weights.json")
    ap.add_argument("--out", type=Path, default=ROOT / "answer.csv")
    ap.add_argument("--sep", default=",", help="разделитель item_id в колонке answer")
    ap.add_argument("--rebuild", action="store_true", help="пересчитать пул кандидатов")
    args = ap.parse_args()

    w = json.loads(args.weights.read_text())
    b = load_benchmark()
    corpus, queries = b["corpus"], b["queries"]
    path = CACHE_DIR / "pool_bench.npz"

    # похожие запросы ищутся по всему train; e5 объявлений train берутся из общего кэша
    train_items = build_local()["corpus"]
    nb = QueryNeighbors(b["train_pairs"], {"ids": train_items["item_id"].to_numpy(),
                                           "emb": e5.encode_items(train_items)})
    W = nb.weights(queries)
    if args.rebuild or not path.exists():
        index = ChannelIndex(corpus, "bench")
        # центроиды и по объявлениям train: у 17% запросов бенчмарка в корпусе нет объявлений их локации
        centroids = location_centroids(corpus, train_items)
        save_pool(build_pool(index, corpus, queries, centroids, hist=nb.hist_vectors(W)), path)
        del index
    pool = load_pool(path, n_queries=len(queries), n_items=len(corpus))

    mc = nb.mc_feature(W, pool, corpus["item_microcat_id"].to_numpy())
    top = PoolArrays(pool, extra={"mc": mc}).top(w, K)
    item_ids = corpus["item_id"].to_numpy()

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(["query_id", "answer"])
        for qid, docs in zip(queries["query_id"], top):
            ids = list(dict.fromkeys(item_ids[d] for d in docs if d >= 0))[:K]
            wr.writerow([qid, args.sep.join(ids)])
    validate_answer(args.out, args.sep)


if __name__ == "__main__":
    main()
