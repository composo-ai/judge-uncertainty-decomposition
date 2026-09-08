"""Synthetic dice generator — experiments_synthetic_dice.md §1.

Truth is drawn FROM the judge's belief (p_i ~ Dir(alpha_i)), so the prior is
calibrated by construction. test_dice.py::test_calibration_pit fails if the
conditioning direction is ever reversed back to truth-first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

K = 3
PEAK = np.array([0.90, 0.05, 0.05])
UNIFORM = np.full(K, 1.0 / K)
LAM_CENTER = 8.0  # geometric mean of the concentration range (see DiceConfig)
ALPHA_CLAMP = 0.05
N_CHOICES = np.array([0, 1, 2, 5, 10])


@dataclass
class DiceConfig:
    n_items: int = 2000
    kappa: float = math.inf  # inf = degenerate: t_i == 0.5 exactly (constant shape)
    rho: float = 0.0
    f_width: float = 1.0  # 0 = constant familiarity (the DCM-correct null axis)
    n_fixed: int | None = (
        None  # fix labels/item; None = choice set (the null needs both)
    )
    # lam spans [8/sqrt(ratio), 8*sqrt(ratio)]: geometric mean fixed at 8 so
    # the S6 dose-response varies range width without moving the centre.
    # ratio=256 reproduces the original [0.5, 128].
    lam_ratio: float = 256.0
    seed: int = 0


@dataclass
class DiceDraw:
    config: DiceConfig
    t: np.ndarray
    f: np.ndarray
    z_f: np.ndarray  # normal score of f, kept for proxy corruption
    q: np.ndarray
    lam: np.ndarray
    alpha: np.ndarray  # judge prior, post-clamp
    p: np.ndarray  # truth
    n: np.ndarray
    counts: np.ndarray
    posterior: np.ndarray  # alpha + counts (conjugate)
    n_clamped: int = field(default=0)


def generate(config: DiceConfig) -> DiceDraw:
    rng = np.random.default_rng(config.seed)
    n_items = config.n_items

    cov = np.array([[1.0, config.rho], [config.rho, 1.0]])
    z = rng.multivariate_normal(np.zeros(2), cov, size=n_items)
    z_t, z_f_neg = z[:, 0], z[:, 1]

    if math.isinf(config.kappa):
        t = np.full(n_items, 0.5)
    else:
        t = stats.beta.ppf(stats.norm.cdf(z_t), config.kappa, config.kappa)

    # high t (contentious) pairs with low f (unfamiliar) when rho > 0;
    # f_width shrinks familiarity spread toward 0.5 without touching z_f,
    # so the proxy corruption stays defined even at the null
    z_f = -z_f_neg
    f = 0.5 + config.f_width * (stats.norm.cdf(z_f) - 0.5)

    perm = np.argsort(rng.random((n_items, K)), axis=1)
    peak_permuted = PEAK[perm]
    q = (1.0 - t[:, None]) * peak_permuted + t[:, None] * UNIFORM

    lam = LAM_CENTER * config.lam_ratio ** (f - 0.5)
    alpha_raw = lam[:, None] * q
    alpha = np.maximum(alpha_raw, ALPHA_CLAMP)
    n_clamped = int((alpha_raw < ALPHA_CLAMP).sum())

    gamma = rng.gamma(alpha)
    p = gamma / gamma.sum(axis=1, keepdims=True)

    if config.n_fixed is None:
        n = rng.choice(N_CHOICES, size=n_items)
    else:
        n = np.full(n_items, config.n_fixed)
    counts = np.zeros((n_items, K), dtype=np.int64)
    for n_val in np.unique(n):
        if n_val == 0:
            continue
        mask = n == n_val
        counts[mask] = _multinomial_rows(rng, int(n_val), p[mask])

    return DiceDraw(
        config=config,
        t=t,
        f=f,
        z_f=z_f,
        q=q,
        lam=lam,
        alpha=alpha,
        p=p,
        n=n,
        counts=counts,
        posterior=alpha + counts,
        n_clamped=n_clamped,
    )


def corrupt_familiarity(
    draw: DiceDraw, r: float, rng: np.random.Generator
) -> np.ndarray:
    """Noisy proxy f_hat at informativeness r — spec §3.2."""
    eps = rng.standard_normal(len(draw.z_f))
    z_hat = r * draw.z_f + math.sqrt(max(0.0, 1.0 - r**2)) * eps
    return stats.norm.cdf(z_hat)


def _multinomial_rows(rng: np.random.Generator, n: int, p: np.ndarray) -> np.ndarray:
    # rng.multinomial broadcasts over rows of p when n is scalar
    return rng.multinomial(n, p)
