"""Offline smoke test for AlloyKG embedding workflow.

Goal: the fastest possible end-to-end sanity check *without any API keys*.

What it tests:
1) Build a tiny corpus from existing parsed JSON files (data/paper_json/**/*_content_list.json).
2) Run a tiny TSDAE training (or skip if model already exists).
3) Load the trained model locally and run a tiny in-memory retrieval demo.

It intentionally does NOT call Zhipu / any external LLM services.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple


def _load_sentences_for_demo(input_glob: str, max_sentences: int, min_chars: int) -> List[str]:
    # Reuse the same corpus builder as the training script for consistency.
    from train_embedding import TrainConfig, build_corpus

    cfg = TrainConfig(
        input_glob=input_glob,
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=os.path.join(".cache", "hf"),
        output_dir=os.path.join("models", "_unused"),
        max_sentences=max_sentences,
        min_chars=min_chars,
        batch_size=2,
        epochs=1,
        lr=2e-5,
        seed=42,
        tie_encoder_decoder=False,
        offline=False,
        noise="simple",
    )
    return build_corpus(cfg)


def _cosine_sim(a, b) -> float:
    # a, b: 1D python lists
    import math

    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0


def _topk(query_vec: List[float], vecs: List[List[float]], texts: List[str], k: int) -> List[Tuple[float, str]]:
    scored = [(_cosine_sim(query_vec, v), t) for v, t in zip(vecs, texts)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def main() -> int:
    p = argparse.ArgumentParser(description="Offline smoke test (no API key)")
    p.add_argument(
        "--input-glob",
        default=os.path.join("data", "paper_json", "**", "*_content_list.json"),
        help="Glob for parsed JSON files.",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join("models", "tsdae-embedding-smoke"),
        help="Where to save/load the tiny trained embedding model.",
    )
    p.add_argument("--cache-dir", default=os.path.join(".cache", "hf"))
    p.add_argument("--max-sentences", type=int, default=50)
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--skip-train", action="store_true", help="Skip training if the model folder already exists")
    p.add_argument(
        "--query",
        default="What is the relationship between alloy composition and mechanical properties?",
        help="Query used for the retrieval demo.",
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--offline", action="store_true", help="Force offline mode (requires HF cache already populated)")
    p.add_argument(
        "--noise",
        choices=["simple", "nltk"],
        default="simple",
        help="Noise function used during TSDAE training (default: simple, no NLTK downloads).",
    )

    args = p.parse_args()

    # 1) Tiny training (optional)
    model_ready = os.path.isdir(args.output_dir) and any(
        os.path.exists(os.path.join(args.output_dir, name))
        for name in ("config.json", "modules.json", "sentence_bert_config.json")
    )

    if args.skip_train and model_ready:
        print(f"[OK] Skip training; found existing model at: {args.output_dir}")
    else:
        from train_embedding import TrainConfig, train

        cfg = TrainConfig(
            input_glob=args.input_glob,
            base_model="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            max_sentences=args.max_sentences,
            min_chars=args.min_chars,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=2e-5,
            seed=42,
            tie_encoder_decoder=False,
            offline=bool(args.offline),
            noise=str(args.noise),
        )
        train(cfg)

    # 2) Load locally + encode + mini retrieval
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.output_dir)

    # Small text pool
    texts = _load_sentences_for_demo(args.input_glob, max_sentences=200, min_chars=max(40, args.min_chars // 2))
    if len(texts) > 200:
        texts = texts[:200]

    vecs = model.encode(texts, normalize_embeddings=True)
    q = model.encode([args.query], normalize_embeddings=True)[0]

    # Convert numpy arrays -> python lists
    try:
        vecs_list = [v.tolist() for v in vecs]
        q_list = q.tolist()
    except Exception:
        vecs_list = [list(v) for v in vecs]
        q_list = list(q)

    hits = _topk(q_list, vecs_list, texts, k=args.top_k)

    print("\n=== Offline retrieval demo (cosine) ===")
    print(f"Query: {args.query}")
    for i, (score, text) in enumerate(hits, 1):
        print(f"{i:02d}. {score:.4f}  {text[:180]}")

    print("\n[OK] Offline smoke test finished.")
    print(f"Tip: To use this model in ragall.py: $env:LOCAL_EMBEDDING_MODEL_DIR=\"{args.output_dir}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
