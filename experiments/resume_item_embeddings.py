from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


ITEM_TEXT_COLS = [
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
]


def join_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume incomplete multilingual E5 item embeddings in-place."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=256)
    args = parser.parse_args()

    train_path = args.train_path.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    model_path = artifacts_dir / "multilingual-e5-base"
    embeddings_path = artifacts_dir / "train_catalog_embeddings.npy"
    ids_path = artifacts_dir / "train_catalog_ids.csv"

    for path in (train_path, model_path, embeddings_path, ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    train = pd.read_parquet(train_path)
    required = {"item_id", *ITEM_TEXT_COLS}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Missing train columns: {sorted(missing)}")

    catalog = (
        train[["item_id"] + ITEM_TEXT_COLS]
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )

    saved_ids = pd.read_csv(ids_path, dtype={"item_id": "string"})["item_id"]
    current_ids = catalog["item_id"].astype("string")
    if saved_ids.tolist() != current_ids.tolist():
        raise ValueError(
            "Catalog item_id order differs from the embedding file; refusing to write."
        )

    embeddings = np.load(embeddings_path, mmap_mode="r+")
    if embeddings.shape[0] != len(catalog) or embeddings.shape[1] != 768:
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; catalog rows={len(catalog)}"
        )

    incomplete = []
    scan_step = 20_000
    for start in range(0, len(embeddings), scan_step):
        end = min(start + scan_step, len(embeddings))
        norms = np.linalg.norm(np.asarray(embeddings[start:end]), axis=1)
        incomplete.extend((np.flatnonzero(norms < 0.99) + start).tolist())

    print(f"Catalog rows: {len(catalog):,}")
    print(f"Incomplete vectors to encode: {len(incomplete):,}")
    if not incomplete:
        print("All item vectors are already complete.")
        return

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = SentenceTransformer(str(model_path), device="cpu")
    model.max_seq_length = args.max_seq_length

    texts = join_text(catalog, ITEM_TEXT_COLS).tolist()
    passage_texts = ["passage: " + text for text in texts]
    del texts

    for start in tqdm(range(0, len(incomplete), args.batch_size), unit="batch"):
        idx = incomplete[start:start + args.batch_size]
        batch_texts = [passage_texts[i] for i in idx]
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings[idx] = batch_embeddings.astype(np.float32, copy=False)
        embeddings.flush()

    remaining = 0
    for start in range(0, len(embeddings), scan_step):
        block = np.asarray(embeddings[start:start + scan_step])
        remaining += int(np.count_nonzero(np.linalg.norm(block, axis=1) < 0.99))

    print(f"Remaining incomplete vectors: {remaining:,}")
    print(f"Updated: {embeddings_path}")


if __name__ == "__main__":
    main()
