"""Каналы поиска. Каждый канал даёт матрицу скоров (запросы x объявления).

tw  TF-IDF по леммам заголовка (uni+bigram)
tc  TF-IDF по символьным n-граммам заголовка
bm  BM25 по леммам заголовка, описания и «Вид/Тип услуги»
sp  TF-IDF «Вид/Тип услуги» из фильтров запроса и параметров объявления
e5  косинус e5-эмбеддингов запроса и объявления
hx  косинус «вектора истории» (см. prior.py) и объявления
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import e5
from .data import CACHE_DIR
from .text import lemmatize, normalize, service_params

CHANNELS = ["tw", "tc", "bm", "sp", "e5", "hx"]


def corpus_texts(corpus: pd.DataFrame, name: str) -> pd.DataFrame:
    """Нормализованные и лемматизированные тексты корпуса (лемматизация долгая — кэшируется)."""
    path = CACHE_DIR / f"{name}_texts.parquet"
    if path.exists():
        cached = pd.read_parquet(path)
        if len(cached) == len(corpus) and (cached["item_id"].values == corpus["item_id"].values).all():
            return cached
    title = corpus["item_title_raw"]
    params = corpus["item_infm_params_text"].map(service_params)
    out = pd.DataFrame({
        "item_id": corpus["item_id"].values,
        "title_norm": title.map(normalize),
        "title_lem": title.map(lemmatize),
        "full_lem": (title + " " + corpus["item_description_raw"] + " " + params).map(lemmatize),
        "params_lem": params.map(lemmatize),
    })
    out.to_parquet(path)
    return out


def query_texts(queries: pd.DataFrame) -> pd.DataFrame:
    q = queries["search_query"].fillna("").astype(str)
    params = queries["search_infm_params_text"].fillna("").astype(str).map(service_params)
    return pd.DataFrame({
        "raw": q,
        "title_norm": q.map(normalize),
        "title_lem": q.map(lemmatize),
        "params_lem": params.map(lemmatize),
    })


class BM25Matrix:
    """Okapi BM25 в виде разреженной матрицы: веса документов считаются один раз, скор = q @ W.T."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, min_df: int = 2):
        self.k1, self.b = k1, b
        self.vec = CountVectorizer(min_df=min_df, token_pattern=r"\S+", dtype=np.float32)

    def fit(self, docs):
        tf = self.vec.fit_transform(docs).tocsr()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.W = sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
        return self

    def transform_queries(self, queries):
        q = self.vec.transform(queries)
        q.data[:] = 1.0
        return q.tocsr()


class ChannelIndex:
    def __init__(self, corpus: pd.DataFrame, name: str):
        t = corpus_texts(corpus, name)
        self.tw = TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, dtype=np.float32)
        self.Dtw = self.tw.fit_transform(t["title_lem"]).T.tocsr()
        self.tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                  sublinear_tf=True, dtype=np.float32)
        self.Dtc = self.tc.fit_transform(t["title_norm"]).T.tocsr()
        self.bm = BM25Matrix().fit(t["full_lem"])
        self.Dbm = self.bm.W.T.tocsr()
        self.sp = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, dtype=np.float32)
        self.Dsp = self.sp.fit_transform(t["params_lem"]).T.tocsr()

        self.E = torch.from_numpy(e5.encode_items(corpus))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.E_dev = self.E.to(self.device)

    def encode_queries(self, queries: pd.DataFrame, hist: np.ndarray | None = None) -> dict:
        """hist — векторы истории (n_queries x 768); без них канал hx нулевой."""
        qt = query_texts(queries)
        if hist is None:
            hist = np.zeros((len(queries), self.E.shape[1]), np.float32)
        return {
            "tw": self.tw.transform(qt["title_lem"]).tocsr(),
            "tc": self.tc.transform(qt["title_norm"]).tocsr(),
            "bm": self.bm.transform_queries(qt["title_lem"]),
            "sp": self.sp.transform(qt["params_lem"]).tocsr(),
            "e5": torch.from_numpy(e5.encode_queries(qt["raw"].tolist())),
            "hx": torch.from_numpy(np.ascontiguousarray(hist, dtype=np.float32)),
        }

    def scores(self, Q: dict, rows: slice) -> dict:
        out = {}
        for ch, D in (("tw", self.Dtw), ("tc", self.Dtc), ("bm", self.Dbm), ("sp", self.Dsp)):
            out[ch] = torch.from_numpy((Q[ch][rows] @ D).toarray())
        for ch in ("e5", "hx"):
            out[ch] = (Q[ch][rows].to(self.device) @ self.E_dev.T).cpu()
        return out
