"""ChaosNLI data layer — loading, half-splits, and the noise ceiling.

Labels arrive as per-item count vectors (individual annotations are not
ordered or identified), so the half-split is a hypergeometric draw from the
count multiset — valid because labels are exchangeable within an item.

Data: chaosNLI_v1.0 (Nie et al. 2020), files chaosNLI_snli.jsonl and
chaosNLI_mnli_m.jsonl; 3,113 items, exactly 100 labels each, classes e/n/c.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("e", "n", "c")
K = len(CLASSES)
LABELS_PER_ITEM = 100


@dataclass
class ChaosNLI:
    uids: list[str]
    premises: list[str]
    hypotheses: list[str]
    counts: np.ndarray  # (N, 3) int, rows sum to 100
    source: list[str]  # "snli" | "mnli_m" per item


def load(data_dir: Path) -> ChaosNLI:
    uids, premises, hypotheses, counts, source = [], [], [], [], []
    for name in ("snli", "mnli_m"):
        path = data_dir / f"chaosNLI_{name}.jsonl"
        for line in open(path):
            row = json.loads(line)
            vec = [row["label_counter"].get(c, 0) for c in CLASSES]
            total = sum(vec)
            if total != LABELS_PER_ITEM:
                raise ValueError(f"{row['uid']}: {total} labels, expected 100")
            uids.append(row["uid"])
            premises.append(row["example"]["premise"])
            hypotheses.append(row["example"]["hypothesis"])
            counts.append(vec)
            source.append(name)
    return ChaosNLI(
        uids=uids,
        premises=premises,
        hypotheses=hypotheses,
        counts=np.array(counts, dtype=np.int64),
        source=source,
    )


def half_split(
    counts: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split each item's label multiset into two random halves of 50.

    Sequential multivariate hypergeometric draw per item, vectorised over
    items one class at a time.
    """
    n_items = counts.shape[0]
    half_a = np.zeros_like(counts)
    remaining_draw = np.full(n_items, LABELS_PER_ITEM // 2)
    remaining_pool = counts.sum(axis=1).copy()
    for j in range(K):
        take = rng.hypergeometric(
            ngood=counts[:, j],
            nbad=remaining_pool - counts[:, j],
            nsample=remaining_draw,
        )
        half_a[:, j] = take
        remaining_draw -= take
        remaining_pool -= counts[:, j]
    return half_a, counts - half_a


def debiased_squared_error(pred: np.ndarray, counts_eval: np.ndarray) -> np.ndarray:
    """Unbiased estimate of ||pred - p_true||^2 from a k-label empirical
    distribution — the finite-k analytic correction promised in
    experiments_estimation.md §2 / paper_plan.md §6.

    E||pred - p_hat||^2 = ||pred - p_true||^2 + sum_j p_j(1-p_j)/k, and
    (k/(k-1)) sum_j p_hat_j(1-p_hat_j) unbiasedly estimates the sum, so the
    correction term is sum_j p_hat_j(1-p_hat_j)/(k-1). Can be negative for a
    single item (it is an unbiased estimator, not a distance); report means.
    """
    k = counts_eval.sum(axis=1)
    p_hat = counts_eval / k[:, None]
    naive = ((pred - p_hat) ** 2).sum(axis=1)
    correction = (p_hat * (1.0 - p_hat)).sum(axis=1) / (k - 1.0)
    return naive - correction


def manhattan(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Average Manhattan distance per item — M1's categorical lock."""
    return np.abs(p - q).sum(axis=1)


def noise_ceiling(
    counts: np.ndarray, rng: np.random.Generator, n_splits: int = 100
) -> dict:
    """Manhattan distance between the two half-split empirical distributions,
    averaged over random splits: the best any predictor of one half can do
    when scored against the other."""
    per_split = np.empty((n_splits, counts.shape[0]))
    for s in range(n_splits):
        a, b = half_split(counts, rng)
        per_split[s] = manhattan(
            a / a.sum(axis=1, keepdims=True), b / b.sum(axis=1, keepdims=True)
        )
    per_item = per_split.mean(axis=0)
    return {
        "mean": float(per_item.mean()),
        "median": float(np.median(per_item)),
        "p90": float(np.percentile(per_item, 90)),
        "per_item": per_item,
    }
