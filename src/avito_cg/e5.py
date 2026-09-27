"""Эмбеддинги multilingual-e5-base для запросов и объявлений.

Объявление кодируется коротким текстом: заголовок + «Вид/Тип услуги». Описания покрывает BM25,
а полный текст на GTX 1650 кодировался бы часами. Эмбеддинги кэшируются по тексту, поэтому
одинаковые заголовки считаются один раз, а кэш общий для валидации и бенчмарка.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import CACHE_DIR, ROOT
from .text import service_params

MODEL_DIR = ROOT / "artifacts" / "e5_baseline" / "multilingual-e5-base"
CACHE = CACHE_DIR / "e5_text_cache"
DIM = 768
MAX_LEN = 64


def _model():
    from sentence_transformers import SentenceTransformer

    # fp32: на GTX 1650 fp16 медленнее
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(MODEL_DIR), device=device)
    model.max_seq_length = MAX_LEN
    return model


def item_texts(corpus: pd.DataFrame) -> pd.Series:
    params = corpus["item_infm_params_text"].map(service_params)
    return (corpus["item_title_raw"].str.strip() + ". " + params).str.lower()


def _load_cache() -> tuple[list[str], np.ndarray]:
    texts, embs = [], []
    for f in sorted(CACHE.glob("part_*.parquet")):
        texts += pd.read_parquet(f)["text"].tolist()
        embs.append(np.load(f.with_suffix(".npy")))
    return texts, (np.concatenate(embs) if embs else np.zeros((0, DIM), np.float32))


def encode_to_cache(texts: list[str], chunk: int = 20_000, batch_size: int = 128) -> None:
    """Докодирует недостающие тексты частями: прерванный прогон продолжается с места остановки."""
    CACHE.mkdir(parents=True, exist_ok=True)
    have, _ = _load_cache()
    todo = sorted(set(texts) - set(have), key=len)
    print(f"e5 cache: {len(have):,} texts, to encode {len(todo):,}", flush=True)
    if not todo:
        return
    model = _model()
    n_parts = len(list(CACHE.glob("part_*.parquet")))
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        vecs = model.encode(["passage: " + t for t in part], batch_size=batch_size,
                            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        name = CACHE / f"part_{n_parts:04d}"
        np.save(name.with_suffix(".npy"), vecs.astype(np.float32))
        pd.DataFrame({"text": part}).to_parquet(name.with_suffix(".parquet"))
        n_parts += 1
        print(f"  e5 {min(s + chunk, len(todo)):,}/{len(todo):,}", flush=True)


def encode_items(corpus: pd.DataFrame) -> np.ndarray:
    texts = item_texts(corpus)
    encode_to_cache(texts.unique().tolist())
    have, emb = _load_cache()
    pos = pd.Series(np.arange(len(have)), index=have)
    pos = pos[~pos.index.duplicated()]
    return emb[pos.reindex(texts).to_numpy()]


def encode_queries(texts: list[str]) -> np.ndarray:
    vecs = _model().encode(["query: " + t for t in texts], batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    return vecs.astype(np.float32)
