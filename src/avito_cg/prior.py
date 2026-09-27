"""Сигналы от похожих запросов из train.

Для запроса берутся k ближайших train-текстов (косинус char TF-IDF в степени power, веса нормированы)
и по ним агрегируются:
  mc — распределение микрокатегорий выбранных объявлений, признак P(микрокатегория кандидата | запрос);
  hx — «вектор истории», средний e5-эмбеддинг выбранных объявлений.

item_id из train напрямую не используются: в корпусе бенчмарка их около 9.6%.
Если текст запроса встречался в train (37% бенчмарка), ближайший сосед — он сам.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalize


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(m.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, 1e-9)).dot(m).tocsr().astype(np.float32)


class QueryNeighbors:
    def __init__(self, train_pairs: pd.DataFrame, item_emb: dict | None = None, k: int = 30, power: float = 2.0):
        """item_emb: {"ids": item_id объявлений train, "emb": их e5-эмбеддинги} — нужен только для hx."""
        self.k, self.power = k, power
        texts = train_pairs["search_query"].astype(str).map(normalize).astype("category")
        self.train_texts = texts.cat.categories
        codes = texts.cat.codes.to_numpy()
        n_texts = len(self.train_texts)

        self.mcats = np.sort(train_pairs["item_microcat_id"].unique())
        self.mc_index = {m: i for i, m in enumerate(self.mcats)}
        counts = sp.csr_matrix(
            (np.ones(len(train_pairs), np.float32), (codes, train_pairs["item_microcat_id"].map(self.mc_index))),
            shape=(n_texts, len(self.mcats)),
        )
        self.mc_dist = _row_normalize(counts)

        self.hist = None
        if item_emb is not None:
            ids = item_emb["ids"]
            pos = pd.Series(np.arange(len(ids)), index=ids).reindex(train_pairs["item_id"]).to_numpy()
            ok = ~np.isnan(pos)
            clicks = sp.csr_matrix((np.ones(ok.sum(), np.float32), (codes[ok], pos[ok].astype(np.int64))),
                                   shape=(n_texts, len(ids)))
            self.hist = np.asarray(_row_normalize(clicks) @ item_emb["emb"], dtype=np.float32)

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                   sublinear_tf=True, dtype=np.float32)
        self.T = self.vec.fit_transform(self.train_texts).T.tocsr()

    def weights(self, queries: pd.DataFrame, batch: int = 1024) -> sp.csr_matrix:
        """Веса соседей: (n_queries x n_train_texts), в каждой строке k ненулевых, сумма 1."""
        Q = self.vec.transform(queries["search_query"].astype(str).map(normalize)).tocsr()
        parts = []
        for s in range(0, Q.shape[0], batch):
            sim = torch.from_numpy((Q[s:s + batch] @ self.T).toarray())
            val, idx = sim.topk(self.k, dim=1)
            w = val.clamp_min(0) ** self.power
            w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
            rows = np.repeat(np.arange(len(idx)), self.k)
            parts.append(sp.csr_matrix((w.numpy().ravel(), (rows, idx.numpy().ravel())),
                                       shape=(len(idx), self.T.shape[1])))
        return sp.vstack(parts).tocsr()

    def hist_vectors(self, W: sp.csr_matrix) -> np.ndarray:
        H = np.asarray(W @ self.hist, dtype=np.float32)
        return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)

    def mc_feature(self, W: sp.csr_matrix, pool: dict, item_mcat: np.ndarray) -> np.ndarray:
        """P(микрокатегория кандидата | запрос) для каждой строки пула."""
        prior = (W @ self.mc_dist).toarray()
        col = np.array([self.mc_index.get(m, -1) for m in item_mcat])[pool["doc"]]
        f = prior[pool["qidx"], np.clip(col, 0, None)]
        f[col < 0] = 0.0
        return f
