"""S7a — truth-first misspecified-judge sweep (experiments_synthetic_dice.md §8).

The world is drawn truth-first: p_i exists, the judge's prediction q_i is a
noisy observation of it whose fidelity rises with familiarity, and the
judge's STATED confidence is deliberately miscalibrated by a knob m:

  m = 0    honest   — stated confidence == actual observation fidelity
  m = 1    flat     — stated confidence carries no item information
  m > 1    inverted — confident exactly where it shouldn't be (overconfident
                      on unfamiliar items, the realistic direction, past flat)

The judge's stated posterior is NOT the Bayes posterior here, so the
conjugate oracle is undefined; ranking value is scored against the exact
PROPHET oracle — realised error reduction against the true p_i.

Usage (from repo root):
    python -m dice.s7a --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sps

from .generator import ALPHA_CLAMP, K, PEAK, UNIFORM
from .posterior import dirichlet_stats, estimated_posterior, fit_dcm

N_CHOICES = np.array([0, 1, 2, 5, 10])
BUDGETS_PCT = (1, 5, 10, 20)
MISCAL_KNOBS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
KAPPA_SHAPE = 1.0  # contentiousness spread, matching the registered sweeps
FIDELITY_CENTER = 8.0
FIDELITY_RATIO = 256.0  # observation fidelity spans [0.5, 128], as lam did
N_VALUE_DRAWS = 20  # MC draws for the prophet acquisition value

ARMS = (
    "oracle",
    "ours_sqerr",
    "v_only",
    "incumbent",
    "incumbent_prior",
    "practitioner",
    "random",
)


@dataclass
class S7aDraw:
    p: np.ndarray  # (N, K) truth, drawn FIRST
    q: np.ndarray  # judge's noisy prediction of p
    f: np.ndarray  # familiarity (drives real fidelity)
    stated_spread: np.ndarray  # the corrupted confidence feature the estimator sees
    n: np.ndarray
    counts: np.ndarray


def generate(n_items: int, miscal: float, seed: int) -> S7aDraw:
    rng = np.random.default_rng(seed)
    t = sps.beta.ppf(rng.random(n_items), KAPPA_SHAPE, KAPPA_SHAPE)
    perm = np.argsort(rng.random((n_items, K)), axis=1)
    p = (1.0 - t[:, None]) * PEAK[perm] + t[:, None] * UNIFORM

    f = rng.random(n_items)
    fidelity = FIDELITY_CENTER * FIDELITY_RATIO ** (f - 0.5)
    q_alpha = np.maximum(fidelity[:, None] * p, ALPHA_CLAMP)
    gamma = rng.gamma(q_alpha)
    q = gamma / gamma.sum(axis=1, keepdims=True)

    # stated confidence: honest at m=0, flat at m=1, inverted beyond;
    # exponent (1 - m) rotates log-fidelity through zero information
    stated_fidelity = FIDELITY_CENTER * FIDELITY_RATIO ** ((1.0 - miscal) * (f - 0.5))
    stated_spread = 1.0 / np.sqrt(stated_fidelity)
    stated_spread = stated_spread + 0.02 * rng.standard_normal(n_items)

    n = rng.choice(N_CHOICES, size=n_items)
    counts = np.zeros((n_items, K), dtype=np.int64)
    for n_val in np.unique(n):
        if n_val == 0:
            continue
        mask = n == n_val
        counts[mask] = rng.multinomial(int(n_val), p[mask])
    return S7aDraw(p=p, q=q, f=f, stated_spread=stated_spread, n=n, counts=counts)


def prophet_values(
    posterior: np.ndarray, p: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Exact-truth expected one-step value: mean Manhattan-error reduction
    from one more label drawn from the true p, MC over N_VALUE_DRAWS."""
    n_items = p.shape[0]
    post_sum = posterior.sum(axis=1, keepdims=True)
    before = np.abs(posterior / post_sum - p).sum(axis=1)
    value = np.zeros(n_items)
    for _ in range(N_VALUE_DRAWS):
        label = (rng.random(n_items)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
        updated = posterior.copy()
        updated[np.arange(n_items), label] += 1
        after = np.abs(updated / (post_sum + 1.0) - p).sum(axis=1)
        value += before - after
    return value / N_VALUE_DRAWS


def run_cell(
    n_items: int, miscal: float, seed: int, ood_r: float | None = None
) -> list[dict]:
    draw = generate(n_items, miscal, seed)
    rng = np.random.default_rng(seed + 2_000_000)

    cols = [draw.stated_spread, np.log1p(draw.n)]
    if ood_r is not None:
        # judge-independent familiarity proxy (stands in for the embedding /
        # label-density OOD channel): noisy observation of f at quality ood_r
        z_f = sps.norm.ppf(np.clip(draw.f, 1e-6, 1 - 1e-6))
        z_hat = ood_r * z_f + np.sqrt(1 - ood_r**2) * rng.standard_normal(n_items)
        cols.append(sps.norm.cdf(z_hat))
    features = np.column_stack(cols)
    features = (features - features.mean(axis=0)) / np.maximum(
        features.std(axis=0), 1e-9
    )
    beta = fit_dcm(features, draw.q, draw.counts)
    posterior = estimated_posterior(beta, features, draw.q, draw.counts)
    est = dirichlet_stats(posterior)

    values = prophet_values(posterior, draw.p, rng)
    prior_entropy = -np.sum(np.where(draw.q > 0, draw.q * np.log(draw.q), 0.0), axis=1)
    tie = rng.random(n_items) * 1e-9
    scores = {
        "oracle": values,
        "ours_sqerr": est.delta,
        "v_only": est.v,
        "incumbent": est.total,
        "incumbent_prior": prior_entropy + tie,
        "practitioner": -draw.n.astype(float) + tie,
        "random": rng.random(n_items),
    }

    post_mean = posterior / posterior.sum(axis=1, keepdims=True)
    err = np.abs(post_mean - draw.p).sum(axis=1)

    rows = []
    for arm in ARMS:
        order = np.argsort(-scores[arm])
        for budget in BUDGETS_PCT:
            top = order[: max(1, n_items * budget // 100)]
            oracle_top = np.argsort(-values)[: len(top)]
            mask = np.zeros(n_items, dtype=bool)
            mask[top] = True
            rows.append(
                {
                    "miscal": miscal,
                    "seed": seed,
                    "arm": arm,
                    "budget_pct": budget,
                    "regret": float(
                        (values[oracle_top].sum() - values[top].sum())
                        / max(values[oracle_top].sum(), 1e-12)
                    ),
                    "auto_resolved_error": float(err[~mask].mean()),
                    "beta_spread": round(float(beta[1]), 4),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--ood-r", type=float, default=None)
    args = parser.parse_args(argv)
    rows = [
        row
        for miscal in MISCAL_KNOBS
        for seed in range(args.seeds)
        for row in run_cell(args.n_items, miscal, seed, ood_r=args.ood_r)
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    name = "s7a.csv" if args.ood_r is None else f"s7a_ood{args.ood_r}.csv"
    path = args.out / name
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
