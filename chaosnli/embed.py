"""Regenerate `embeddings_minilm.npz` — the item-embedding matrix used by the
ChaosNLI concentration regression's familiarity features (nearest labelled
neighbour distance and local label density; see `pipeline.build_features`).

The committed artefact is a (3113, 384) float16 matrix plus the matching
`uids`, produced with `sentence-transformers/all-MiniLM-L6-v2` (384-d) run
locally, one embedding per item on the premise/hypothesis pair. This script
regenerates it with the same model and, when the committed file is present,
reports how far the regenerated vectors deviate from it, so the reconstruction
can be checked rather than trusted.

Usage (from repo root; needs `pip install sentence-transformers`):
    python -m chaosnli.embed --data-dir data/chaosNLI_v1.0 \
        [--out chaosnli/embeddings_minilm.npz] [--check]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chaosnli.adapter import load

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE / "embeddings_minilm.npz")
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed file instead of overwriting it",
    )
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # optional dependency

    data = load(args.data_dir)
    texts = [f"{p} {h}" for p, h in zip(data.premises, data.hypotheses)]
    model = SentenceTransformer(MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float16)
    uids = np.asarray(data.uids)

    if args.check:
        ref = np.load(HERE / "embeddings_minilm.npz")
        assert (ref["uids"] == uids).all(), "uid order differs from the committed file"
        a = emb.astype(np.float64)
        b = ref["embeddings"].astype(np.float64)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"cosine to committed: min {cos.min():.4f}, mean {cos.mean():.4f}")
        return

    np.savez_compressed(args.out, embeddings=emb, uids=uids)
    print(f"wrote {args.out}: {emb.shape} {emb.dtype}")


if __name__ == "__main__":
    main()
