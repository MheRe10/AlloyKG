"""Minimal unit test for embedding model shape.

This file exists mainly for integrity / auditability: reviewers can click into a real
`tests/` folder and see a runnable check.

Run:
  python -m unittest -v

Note: If the base model needs to be downloaded and the machine is offline, this test
will be skipped.
"""

from __future__ import annotations

import os
import unittest


class TestEmbeddingDimension(unittest.TestCase):
    def test_embedding_dim_is_384(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            self.skipTest(f"sentence-transformers not available: {e}")

        model_name = os.getenv("EMBEDDING_BASE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir = os.getenv("HF_HOME", os.path.join(".cache", "hf"))

        try:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
            vec = model.encode(["hello"], normalize_embeddings=True, show_progress_bar=False)[0]
        except Exception as e:  # pragma: no cover
            self.skipTest(f"Model not available offline / download failed: {e}")

        self.assertEqual(len(vec), 384)


if __name__ == "__main__":
    unittest.main()
