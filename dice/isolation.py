"""The isolation exhibit — does each estimated uncertainty component track
its own ground truth and ignore the other's? (Discussion 18 Aug: the direct
answer to "have we demonstrated the aleatoric/epistemic split", graded at
the recovery level rather than via decision regret.)

Per world, a 2x2 Spearman matrix:
                      true aleatoric H(p)   true error |post_mean - p|
  estimated aleatoric        [diag]                 [off-diag]
  estimated epistemic      [off-diag]                 [diag]

Isolation = strong diagonal, near-zero off-diagonal. The truth-truth
baseline correlation is reported so a lucky world can't fake it.

Usage (from repo root):
    python -m dice.isolation \
        [--s7b-cache dice/caches/s7b_cache.jsonl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .generator import DiceConfig, corrupt_familiarity, generate
from .llm_judge import TIERS, build_world, load_cache, q_hat_matrix
from .posterior import dirichlet_stats, estimated_posterior, fit_dcm
from .s7b_analysis import confidence_vector


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def isolation_matrix(
    est_aleatoric: np.ndarray,
    est_epistemic: np.ndarray,
    true_aleatoric: np.ndarray,
    true_error: np.ndarray,
) -> dict:
    def rho(a, b):
        return round(float(spearmanr(a, b).statistic), 3)

    return {
        "est_aleatoric": {
            "vs_true_aleatoric": rho(est_aleatoric, true_aleatoric),
            "vs_true_error": rho(est_aleatoric, true_error),
        },
        "est_epistemic": {
            "vs_true_aleatoric": rho(est_epistemic, true_aleatoric),
            "vs_true_error": rho(est_epistemic, true_error),
        },
        "truth_truth_baseline": rho(true_aleatoric, true_error),
    }


def calibrated_world(n_items: int = 4000, seed: int = 42, proxy_r: float = 0.7) -> dict:
    """Estimated condition of the calibrated generator: honest judge, noisy
    proxy inputs."""
    draw = generate(
        DiceConfig(n_items=n_items, kappa=1.0, rho=0.0, f_width=1.0, seed=seed)
    )
    rng = np.random.default_rng(seed)
    f_hat = corrupt_familiarity(draw, proxy_r, rng)
    beta = fit_dcm(f_hat[:, None], draw.q, draw.counts)
    post = estimated_posterior(beta, f_hat[:, None], draw.q, draw.counts)
    st = dirichlet_stats(post)
    err = np.abs(post / post.sum(1, keepdims=True) - draw.p).sum(1)
    return isolation_matrix(st.aleatoric, st.info, entropy(draw.p), err)


def s7b_world(cache_path: Path, n_dice: int = 500, seed: int = 0) -> dict:
    """The real LLM judge (S7b cache), OOD-featured posterior. Also reports
    the incumbent contrast: what total predictive entropy H(q_hat) tracks."""
    world = build_world(n_dice, seed)
    cache = load_cache(cache_path)
    q_hat, spread = q_hat_matrix(world, cache_path)
    conf = confidence_vector(world, cache)
    conf = np.where(np.isnan(conf), np.nanmean(conf), conf)
    spread = np.where(np.isnan(spread), np.nanmean(spread), spread)
    tier_idx = np.array([TIERS.index(t) for t in world.tier])
    z = np.column_stack([np.eye(4)[tier_idx][:, 1:], spread, np.log1p(world.n), conf])
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-9)
    beta = fit_dcm(z, q_hat, world.counts)
    post = estimated_posterior(beta, z, q_hat, world.counts)
    st = dirichlet_stats(post)
    err = np.abs(post / post.sum(1, keepdims=True) - world.p).sum(1)
    out = isolation_matrix(st.aleatoric, st.info, entropy(world.p), err)
    h_q = entropy(q_hat)
    out["incumbent_contrast_H_qhat"] = {
        "vs_true_aleatoric": round(
            float(spearmanr(h_q, entropy(world.p)).statistic), 3
        ),
        "vs_true_error": round(float(spearmanr(h_q, err).statistic), 3),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s7b-cache",
        type=Path,
        default=Path("dice/caches/s7b_cache.jsonl"),
    )
    args = parser.parse_args(argv)
    import json

    report = {"calibrated_world": calibrated_world()}
    if args.s7b_cache.exists():
        report["real_llm_s7b"] = s7b_world(args.s7b_cache)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
