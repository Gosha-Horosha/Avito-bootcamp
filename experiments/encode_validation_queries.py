from __future__ import annotations

import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

def main() -> None:


    parser = argparse.ArgumentParser(
        description="Encode validation queries with a locally saved multilingual E5 model."
    )
    default_dir = Path(__file__).resolve().parent / "artifacts" / "e5_baseline"
    parser.add_argument("--artifacts-dir", type=Path, default=default_dir)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-seq-length", type=int, default=128)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.resolve()
    queries_path = artifacts_dir / "validation_queries.csv"
    model_path = artifacts_dir / "multilingual-e5-base"
    output_path = artifacts_dir / "validation_query_embeddings.npy"

    for path in (queries_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    queries = pd.read_csv(
        queries_path,
        dtype={"query_key": "string"},
        keep_default_na=False,
    )
    required = {"search_query", "search_infm_params_text", "query_key"}
    missing = required - set(queries.columns)
    if missing:
        raise ValueError(f"Missing columns in validation_queries.csv: {sorted(missing)}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    if device == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    texts = (
        "query: "
        + queries["search_query"].astype(str)
        + " "
        + queries["search_infm_params_text"].astype(str)
    ).tolist()

    print(f"Queries: {len(texts):,}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Loading local model: {model_path}")

    model = SentenceTransformer(str(model_path), device=device)
    model.max_seq_length = args.max_seq_length

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    np.save(output_path, embeddings.astype(np.float32, copy=False))
    print(f"Saved: {output_path}")
    print(f"Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
