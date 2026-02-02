"""TSDAE embedding fine-tuning for AlloyKG.

This script trains a domain-adapted sentence embedding model using Transformer-based
Denoising Auto-Encoder (TSDAE) from the `sentence-transformers` library.

Why TSDAE?
- No manual labels required: learns from raw domain text.
- Fits RAG well: improves retrieval recall for materials-science terms.

Expected input:
- JSON files produced by `parse_papers.py`, e.g. `data/paper_json/*_content_list.json`.
  Each JSON is a list of blocks with keys like {type: "text", text: "..."}.

Output:
- A local SentenceTransformer model folder (can be used by `ragall.py` via LOCAL_EMBEDDING_MODEL_DIR).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class TrainConfig:
    input_glob: str
    base_model: str
    cache_dir: str
    output_dir: str
    max_sentences: int
    min_chars: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    tie_encoder_decoder: bool
    offline: bool
    noise: str


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|")


def _iter_blocks_from_json(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def _extract_texts_from_blocks(blocks: Iterable[dict]) -> Iterable[str]:
    for b in blocks:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            yield str(b.get("text", ""))
        elif t == "image":
            # EN: Optionally include image captions as training text.
            # CN: 可选：把图片 caption 也当作语料。
            for cap in b.get("image_caption", []) or []:
                yield str(cap)
        elif t == "table":
            # EN: Optionally include table captions.
            # CN: 可选：把表格 caption 也当作语料。
            for cap in b.get("table_caption", []) or []:
                yield str(cap)


def _split_to_sentences(text: str) -> List[str]:
    text = " ".join(str(text).split())
    if not text:
        return []

    # A simple segmenter: split on punctuation boundaries.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def build_corpus(cfg: TrainConfig) -> List[str]:
    paths = sorted(glob.glob(cfg.input_glob, recursive=True))
    if not paths:
        raise FileNotFoundError(
            f"No input files found for --input-glob: {cfg.input_glob}. "
            "Run parse_papers.py first or adjust the glob."
        )

    sentences: List[str] = []
    for path in paths:
        blocks = _iter_blocks_from_json(path)
        for text in _extract_texts_from_blocks(blocks):
            for sent in _split_to_sentences(text):
                if len(sent) >= cfg.min_chars:
                    sentences.append(sent)

    random.Random(cfg.seed).shuffle(sentences)
    if cfg.max_sentences > 0:
        sentences = sentences[: cfg.max_sentences]

    if not sentences:
        raise ValueError(
            "No training sentences after filtering. "
            "Try lowering --min-chars or check your parsed JSON content."
        )

    return sentences


def _resolve_local_snapshot(cache_dir: str, repo_id: str) -> str | None:
    # HuggingFace cache layout: <cache_dir>/models--ORG--NAME/snapshots/<rev_hash>/
    if "/" not in repo_id:
        return None
    org, name = repo_id.split("/", 1)
    model_dir = os.path.join(cache_dir, f"models--{org}--{name}")
    snapshots_dir = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = sorted(
        (
            os.path.join(snapshots_dir, d)
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ),
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def train(cfg: TrainConfig) -> None:
    # Heavy imports only when training actually runs.
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, datasets, losses

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    sentences = build_corpus(cfg)
    print(f"Loaded {len(sentences)} sentences for TSDAE training")

    # EN: Downloading HF models can be flaky on some Windows setups. Use cache + retries.
    # CN: Windows 下模型下载可能不稳定；这里使用 cache 并做简单重试。
    base_model = cfg.base_model
    local_snapshot = _resolve_local_snapshot(cfg.cache_dir, cfg.base_model)
    if local_snapshot:
        # EN: If a cached snapshot exists, use it to avoid any network requests.
        # CN: 如果 cache 里有 snapshot，直接用本地路径，避免联网。
        base_model = local_snapshot
    elif cfg.offline:
        raise RuntimeError(
            "Offline mode requested but no cached model snapshot was found. "
            "Run once with network access to populate --cache-dir, then re-run with --offline."
        )

    last_exc: Exception | None = None
    model = None
    for attempt in range(1, 4):
        try:
            model = SentenceTransformer(base_model, cache_folder=cfg.cache_dir)
            break
        except Exception as e:
            last_exc = e
            print(f"Failed to load base model (attempt {attempt}/3): {e}")
            if attempt < 3:
                time.sleep(5 * attempt)

    if model is None:
        raise RuntimeError(
            "Unable to load base model after retries. "
            "If the download was interrupted, re-run the command; HuggingFace will resume from cache."
        ) from last_exc

    def simple_word_dropout_noise(text: str, dropout: float = 0.3) -> str:
        # EN: A lightweight noise function that avoids NLTK resources.
        # CN: 轻量级加噪方式（按空格分词做随机丢词），避免依赖 NLTK 的 punkt 资源。
        words = [w for w in str(text).split() if w]
        if len(words) <= 2:
            return str(text)

        kept = [w for w in words if random.random() > dropout]
        if len(kept) < 2:
            kept = words[:2]
        return " ".join(kept)

    if cfg.noise == "nltk":
        train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    else:
        train_dataset = datasets.DenoisingAutoEncoderDataset(
            sentences,
            noise_fn=simple_word_dropout_noise,
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # EN: TSDAE uses a decoder to reconstruct the original sentence.
    # CN: TSDAE 通过 decoder 重建原句，逼迫 encoder 学到更好的句向量表示。
    train_loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=base_model,
        tie_encoder_decoder=cfg.tie_encoder_decoder,
    )

    steps_per_epoch = max(1, len(train_dataloader))
    warmup_steps = int(0.1 * steps_per_epoch * cfg.epochs)

    os.makedirs(cfg.output_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": cfg.lr},
        show_progress_bar=True,
        output_path=cfg.output_dir,
    )

    print(f"Saved TSDAE embedding model to: {cfg.output_dir}")
    print(
        "\nNext: set LOCAL_EMBEDDING_MODEL_DIR to this folder when running ragall.py, e.g.\n"
        f"  $env:LOCAL_EMBEDDING_MODEL_DIR=\"{cfg.output_dir}\"\n"
        "  python src/ragall.py\n"
    )


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="TSDAE embedding fine-tuning for AlloyKG")
    p.add_argument(
        "--input-glob",
        default=os.path.join("data", "paper_json", "**", "*_content_list.json"),
        help="Glob for parsed JSON files (from parse_papers.py).",
    )
    p.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base SentenceTransformer model to adapt.",
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.join(".cache", "hf"),
        help="Local cache directory for HuggingFace downloads.",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join("models", "tsdae-embedding"),
        help="Output directory for the trained embedding model.",
    )
    p.add_argument(
        "--max-sentences",
        type=int,
        default=50_000,
        help="Max number of sentences used for training (0 = no limit).",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Filter out very short sentences.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tie-encoder-decoder",
        action="store_true",
        help="Tie encoder and decoder weights. This may trigger additional downloads.",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode (no network). Requires the base model to already exist in cache.",
    )

    p.add_argument(
        "--noise",
        choices=["simple", "nltk"],
        default="simple",
        help=(
            "Noise function for TSDAE. 'simple' avoids NLTK downloads; "
            "'nltk' uses the default NLTK-based tokenizer (may require punkt resources)."
        ),
    )

    args = p.parse_args()
    return TrainConfig(
        input_glob=args.input_glob,
        base_model=args.base_model,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_sentences=args.max_sentences,
        min_chars=args.min_chars,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        tie_encoder_decoder=bool(args.tie_encoder_decoder),
        offline=bool(args.offline),
        noise=str(args.noise),
    )


if __name__ == "__main__":
    train(parse_args())
