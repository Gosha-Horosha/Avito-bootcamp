from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

artifact_dir = Path(
    r"C:\Users\User\Desktop\ml_running\artifacts\e5_baseline"
)
train_path = Path(r"C:\Users\User\Desktop\NLP_avito_interns\dataset\train.parquet")
train = pd.read_parquet(train_path)

print("Загружен train:", train.shape)

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

def make_query_key(df):
    return (
        df[QUERY_COLS]
        .fillna("")
        .astype(str)
        .agg("\x1f".join, axis=1)
    )

# Восстанавливаем тот же validation split.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, val_idx = next(
    splitter.split(train, groups=train["search_query"].fillna(""))
)
val_pairs = train.iloc[val_idx].copy()
val_pairs["query_key"] = make_query_key(val_pairs)

saved_queries = pd.read_csv(
    artifact_dir / "validation_queries.csv",
    dtype={"query_key": "string"},
    keep_default_na=False,
)
query_embeddings = np.load(
    artifact_dir / "validation_query_embeddings.npy",
    mmap_mode="r",
)
item_embeddings = np.load(
    artifact_dir / "train_catalog_embeddings.npy",
    mmap_mode="r",
)
item_ids = pd.read_csv(
    artifact_dir / "train_catalog_ids.csv",
    dtype={"item_id": "string"},
)["item_id"].to_numpy()

assert len(saved_queries) == len(query_embeddings)
assert len(item_ids) == len(item_embeddings)
assert item_embeddings.shape[1] == query_embeddings.shape[1]

# Проверяем, что кодирование объявлений не оборвалось на середине.
bad_vectors = 0
for start in range(0, len(item_embeddings), 20_000):
    block = np.asarray(item_embeddings[start:start + 20_000])
    bad_vectors += np.count_nonzero(np.linalg.norm(block, axis=1) < 0.99)

print("Нулевых/неполных векторов объявлений:", bad_vectors)