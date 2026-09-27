import re
import bm25s
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_TEXT_COLS = [
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
]

def join_text(df, columns):
    return (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def make_query_key(df):
    return (
        df[QUERY_COLS]
        .fillna("")
        .astype(str)
        .agg("\x1f".join, axis=1)
    )

# Один и тот же search_query не попадёт и в train-, и в validation-пары.

from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\User\Desktop\NLP_avito_interns\dataset")
TRAIN_PATH = DATA_DIR / "train.parquet"

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Не найден файл: {TRAIN_PATH}")

train = pd.read_parquet(TRAIN_PATH)

print("Загружен train:", train.shape)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(
    splitter.split(train, groups=train["search_query"].fillna(""))
)

val_pairs = train.iloc[val_idx].copy()
val_pairs["query_key"] = make_query_key(val_pairs)

# Один validation-запрос — одна комбинация текста, локации и фильтров.
val_queries = (
    val_pairs[QUERY_COLS]
    .drop_duplicates()
    .reset_index(drop=True)
)
val_queries["query_key"] = make_query_key(val_queries)

# Фиксированная выборка для первого замера.
val_sample = (
    val_queries.sample(n=min(1000, len(val_queries)), random_state=42)
    .reset_index(drop=True)
)

# BM25-каталог: уникальные объявления из train.
catalog = (
    train[["item_id"] + ITEM_TEXT_COLS]
    .drop_duplicates("item_id")
    .reset_index(drop=True)
)
item_ids = catalog["item_id"].astype(str).to_numpy()

doc_texts = join_text(catalog, ITEM_TEXT_COLS).tolist()
query_texts = join_text(
    val_sample,
    ["search_query", "search_infm_params_text"],
).tolist()

print("Уникальных объявлений для индекса:", len(catalog))
print("Validation-запросов в первом замере:", len(val_sample))

# Создаём индекс BM25.
doc_tokens = bm25s.tokenize(doc_texts)
retriever = bm25s.BM25(method="lucene")
retriever.index(doc_tokens)

# Получаем до 50 кандидатов для каждого validation-запроса.
query_tokens = bm25s.tokenize(query_texts)
results, scores = retriever.retrieve(query_tokens, k=50)

relevant_by_query = (
    val_pairs.groupby("query_key")["item_id"]
    .agg(lambda s: set(s.astype(str)))
    .to_dict()
)

recalls = []

for row, query in val_sample.iterrows():
    query_key = query["query_key"]
    predicted_ids = set(item_ids[results[row]])
    relevant_ids = relevant_by_query[query_key]
    recalls.append(len(predicted_ids & relevant_ids) / len(relevant_ids))

print(f"Предварительный Recall@50 на выборке: {np.mean(recalls):.6f}")