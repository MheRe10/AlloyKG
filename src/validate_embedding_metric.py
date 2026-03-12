"""Post-training quick validation metric for a local SentenceTransformer embedding model.

This prints ONE line that can be pasted into reports:
- A simple hold-out metric: mean cosine similarity between clean sentences and their
  noised versions (word-dropout), using the trained model.

Note: TSDAE is unsupervised; this is not a gold-label accuracy metric.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from typing import Any, Dict, Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


def _noise(rng: random.Random, text: str, dropout: float = 0.3) -> str:
    words = [w for w in str(text).split() if w]
    if len(words) <= 2:
        return str(text)

    kept = [w for w in words if rng.random() > dropout]
    if len(kept) < 2:
        kept = words[:2]
    return " ".join(kept)


def _iter_blocks_from_json(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected format: a list of blocks, each a dict with keys like {type,text,...}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def _extract_texts_from_blocks(blocks: Iterable[Dict[str, Any]]) -> Iterable[str]:
    for b in blocks:
        if str(b.get("type", "")).lower() != "text":
            continue
        text = str(b.get("text", ""))
        if text.strip():
            yield text


def _split_to_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _build_corpus(
    *,
    input_glob: str,
    min_chars: int,
    max_sentences: int,
    seed: int,
) -> List[str]:
    paths = sorted(glob.glob(input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No input files found for --input-glob: {input_glob}")

    sentences: List[str] = []
    for path in paths:
        blocks = _iter_blocks_from_json(path)
        for text in _extract_texts_from_blocks(blocks):
            for sent in _split_to_sentences(text):
                if len(sent) >= min_chars:
                    sentences.append(sent)

    random.Random(seed).shuffle(sentences)
    if max_sentences > 0:
        sentences = sentences[:max_sentences]

    if not sentences:
        raise ValueError(
            "No sentences after filtering. Try lowering --min-chars or check parsed JSON."
        )
    return sentences


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a trained embedding model (quick metric)")
    p.add_argument("--model-dir", required=True, help="Local embedding model directory")
    p.add_argument(
        "--input-glob",
        default="data/paper_json/**/*_content_list.json",
        help="Parsed paper JSON glob (same as train_embedding.py)",
    )
    p.add_argument("--max-sentences", type=int, default=200)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.3)

    args = p.parse_args()

    sentences = _build_corpus(
        input_glob=str(args.input_glob),
        min_chars=int(args.min_chars),
        max_sentences=int(args.max_sentences),
        seed=int(args.seed),
    )
    val_size = min(int(args.val_size), max(10, len(sentences) // 5))
    val = sentences[-val_size:]

    rng = random.Random(int(args.seed))
    noised = [_noise(rng, s, dropout=float(args.dropout)) for s in val]

    model = SentenceTransformer(args.model_dir)
    emb_clean = model.encode(val, normalize_embeddings=True, show_progress_bar=False)
    emb_noised = model.encode(noised, normalize_embeddings=True, show_progress_bar=False)

    cos = (np.asarray(emb_clean) * np.asarray(emb_noised)).sum(axis=1)

    checkpoint = f"{args.model_dir}/model.safetensors"
    print(
        f"VAL mean_cos_clean_vs_noised={cos.mean():.4f} "
        f"n={len(val)} "
        f"checkpoint={checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
