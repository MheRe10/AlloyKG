"""Offline retrieval-only runner (no LLM / no API keys).

This script is a lightweight stand-in for the "query" portion of src/ragall.py,
but it avoids calling any online LLMs.

What it does:
- Loads text chunks produced by the end-to-end pipeline from:
  data/end_to_end/rag_storage/kv_store_text_chunks.json
- Embeds chunks and a query using a local SentenceTransformer model.
- Prints top-k most similar chunks as evidence.

Notes:
- You still need to have run the document processing step at least once (so the
  chunk store exists).
- For the embedding model, set LOCAL_EMBEDDING_MODEL_DIR to a trained model
  folder (e.g., models/tsdae-embedding-smoke) or pass --embedding-model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    file_path: str


def _should_keep_chunk(content: str, *, min_chars: int, exclude_prefixes: List[str]) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    if min_chars > 0 and len(text) < min_chars:
        return False
    for prefix in exclude_prefixes:
        if prefix and text.lower().startswith(prefix.lower()):
            return False
    return True


def _load_chunks(
    chunks_json_path: str,
    *,
    max_chunks: int,
    min_chars: int,
    exclude_prefixes: List[str],
) -> List[Chunk]:
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    chunks: List[Chunk] = []
    for chunk_id, item in data.items():
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", ""))
        file_path = str(item.get("file_path", ""))
        if not _should_keep_chunk(content, min_chars=min_chars, exclude_prefixes=exclude_prefixes):
            continue
        chunks.append(Chunk(chunk_id=chunk_id, content=content, file_path=file_path))
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break

    if not chunks:
        raise RuntimeError(f"No chunks loaded from: {chunks_json_path}")
    return chunks


def _topk_dot(query_vec, doc_vecs, k: int) -> List[Tuple[int, float]]:
    # When vectors are L2-normalized, dot product equals cosine similarity.
    import numpy as np

    q = np.asarray(query_vec)
    dv = np.asarray(doc_vecs)
    scores = dv @ q

    if k <= 0:
        k = 5
    k = min(k, int(scores.shape[0]))

    # Partial top-k for speed
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def main() -> int:
    p = argparse.ArgumentParser(description="AlloyKG offline retrieval (no LLM)")
    p.add_argument(
        "--storage-dir",
        default=os.path.join("data", "end_to_end", "rag_storage"),
        help="RAG storage directory (contains kv_store_text_chunks.json).",
    )
    p.add_argument(
        "--chunks-file",
        default="kv_store_text_chunks.json",
        help="Chunk store filename inside --storage-dir.",
    )
    p.add_argument(
        "--embedding-model",
        default=os.getenv("LOCAL_EMBEDDING_MODEL_DIR", ""),
        help="Local SentenceTransformer model folder. If empty, uses LOCAL_EMBEDDING_MODEL_DIR.",
    )
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--max-chunks",
        type=int,
        default=500,
        help="Limit number of chunks to embed (0 = no limit).",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Drop very short chunks before embedding.",
    )
    p.add_argument(
        "--exclude-prefix",
        action="append",
        default=["Image Content Analysis"],
        help=(
            "Exclude chunks whose content starts with this prefix. "
            "Can be specified multiple times. Default filters out image-analysis chunks."
        ),
    )
    p.add_argument(
        "--preview-chars",
        type=int,
        default=220,
        help="How many characters of each chunk to print.",
    )

    args = p.parse_args()

    if not args.embedding_model:
        raise SystemExit(
            "Missing embedding model. Set LOCAL_EMBEDDING_MODEL_DIR or pass --embedding-model. "
            "Example: $env:LOCAL_EMBEDDING_MODEL_DIR=\"models/tsdae-embedding-smoke\""
        )

    chunks_path = os.path.join(args.storage_dir, args.chunks_file)
    if not os.path.isfile(chunks_path):
        raise SystemExit(
            f"Chunk store not found: {chunks_path}\n"
            "Run src/ragall.py document processing first, or point --storage-dir to the correct folder."
        )

    chunks = _load_chunks(
        chunks_path,
        max_chunks=int(args.max_chunks),
        min_chars=int(args.min_chars),
        exclude_prefixes=list(args.exclude_prefix),
    )

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)

    texts = [c.content for c in chunks]
    doc_vecs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    query_vec = model.encode([args.query], show_progress_bar=False, normalize_embeddings=True)[0]

    hits = _topk_dot(query_vec, doc_vecs, k=int(args.top_k))

    print("\n=== Offline top-k chunks (no LLM) ===")
    print(f"Query: {args.query}")
    for rank, (i, score) in enumerate(hits, 1):
        c = chunks[i]
        preview = c.content.replace("\n", " ").strip()
        if args.preview_chars > 0:
            preview = preview[: int(args.preview_chars)]
        print(f"\n{rank:02d}. score={score:.4f}")
        print(f"    chunk_id: {c.chunk_id}")
        print(f"    file: {c.file_path}")
        print(f"    text: {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
