"""Загрузка данных, дедупликация, локальный сплит и ground truth.

train -> удаление полных дублей (search_* + item_id) -> GroupShuffleSplit(0.1, seed=42)
Корпус для валидации — все объявления train. Запрос — полная комбинация полей search_*.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
CACHE_DIR = ROOT / "artifacts" / "cache"

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_location_id",
    "item_microcat_id",
    "item_latitude",
    "item_longitude",
]
PAIR_COLS = QUERY_COLS + ["item_id", "item_microcat_id"]


def query_key(df: pd.DataFrame) -> pd.Series:
    return df[QUERY_COLS].fillna("").astype(str).agg("\x1f".join, axis=1)


def _clean_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["item_id"] = df["item_id"].astype(str)
    for col in ("item_latitude", "item_longitude"):
        # в parquet координаты хранятся как decimal.Decimal
        df[col] = pd.to_numeric(df[col].astype(object), errors="coerce").astype("float32")
    for col in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        df[col] = df[col].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_train_dedup() -> pd.DataFrame:
    train = pd.read_parquet(DATA_DIR / "train.parquet", columns=QUERY_COLS + ITEM_COLS)
    train = train.drop_duplicates(subset=QUERY_COLS + ["item_id"]).reset_index(drop=True)
    train["search_infm_params_text"] = train["search_infm_params_text"].fillna("")
    train["item_id"] = train["item_id"].astype(str)
    return train


def build_local(test_size: float = 0.1, seed: int = 42, force: bool = False, split_by: str = "text") -> dict:
    """Локальная валидация: corpus, train_pairs, val_queries, val_gt (кэшируется в parquet).

    split_by="text" — по тексту запроса: тексты val не встречаются в train (основной режим).
    split_by="key"  — по полной комбинации search_*: часть текстов val есть в train, как у 37% бенчмарка.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if split_by == "text" else f"_{split_by}"
    paths = {k: CACHE_DIR / f"local{suffix}_{k}.parquet" for k in ("corpus", "train_pairs", "val_pairs")}
    if not force and all(p.exists() for p in paths.values()):
        corpus = pd.read_parquet(paths["corpus"])
        train_pairs = pd.read_parquet(paths["train_pairs"])
        val_pairs = pd.read_parquet(paths["val_pairs"])
    else:
        train = load_train_dedup()
        groups = train["search_query"].fillna("") if split_by == "text" else query_key(train)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, va_idx = next(splitter.split(train, groups=groups))
        corpus = _clean_items(train[ITEM_COLS].drop_duplicates("item_id"))
        train_pairs = train.iloc[tr_idx][PAIR_COLS].reset_index(drop=True)
        val_pairs = train.iloc[va_idx][PAIR_COLS].reset_index(drop=True)
        corpus.to_parquet(paths["corpus"])
        train_pairs.to_parquet(paths["train_pairs"])
        val_pairs.to_parquet(paths["val_pairs"])

    val_pairs = val_pairs.assign(query_key=query_key(val_pairs))
    val_queries = (
        val_pairs[QUERY_COLS + ["query_key"]]
        .drop_duplicates("query_key")
        .reset_index(drop=True)
        .rename(columns={"query_key": "query_id"})
    )
    val_gt = val_pairs.groupby("query_key")["item_id"].agg(set).to_dict()
    return dict(corpus=corpus, train_pairs=train_pairs, val_queries=val_queries, val_gt=val_gt)


def one_per_text(val_queries: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """По одному запросу на текст: в бенчмарке все тексты уникальны."""
    return (
        val_queries.sample(frac=1.0, random_state=seed)
        .drop_duplicates("search_query")
        .reset_index(drop=True)
    )


def load_benchmark() -> dict:
    queries = pd.read_parquet(DATA_DIR / "benchmark_queries.parquet")
    queries["query_id"] = queries["query_id"].astype(str)
    queries["search_infm_params_text"] = queries["search_infm_params_text"].fillna("")
    corpus = _clean_items(pd.read_parquet(DATA_DIR / "benchmark_items.parquet", columns=ITEM_COLS))
    train_pairs = load_train_dedup()[PAIR_COLS]
    return dict(corpus=corpus, train_pairs=train_pairs, queries=queries)
