"""T1–T6 from experiments_synthetic_dice.md §5. Run from repo root:
pytest dice/ -m unit
"""

import math

import numpy as np
import pytest
from scipy import stats as sps

from dice.arms import all_regrets, regret, true_values
from dice.generator import (
    ALPHA_CLAMP,
    DiceConfig,
    corrupt_familiarity,
    generate,
)
from dice.posterior import dirichlet_stats

pytestmark = pytest.mark.unit


# T1 — closed forms vs Monte Carlo
@pytest.mark.parametrize(
    "alpha",
    [
        (2.0, 2.0, 2.0),
        (0.5, 0.5, 0.5),
        (10.0, 1.0, 1.0),
        (ALPHA_CLAMP, 5.0, 5.0),  # clamped near-vertex case
        (120.0, 4.0, 4.0),
    ],
)
def test_closed_forms_match_monte_carlo(alpha):
    rng = np.random.default_rng(7)
    alpha_arr = np.array([alpha])
    draws = rng.dirichlet(alpha, size=200_000)
    got = dirichlet_stats(alpha_arr)

    mc_mean = draws.mean(axis=0)
    mc_total = -np.sum(mc_mean * np.log(mc_mean))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_draws = np.where(draws > 0, np.log(draws), 0.0)
    mc_aleatoric = float(-(draws * log_draws).sum(axis=1).mean())
    mc_v = float(draws.var(axis=0).sum())

    assert got.total[0] == pytest.approx(mc_total, abs=5e-3)
    assert got.aleatoric[0] == pytest.approx(mc_aleatoric, abs=5e-3)
    assert got.info[0] == pytest.approx(mc_total - mc_aleatoric, abs=8e-3)
    assert got.v[0] == pytest.approx(mc_v, rel=0.02)
    # delta: one-step expected reduction in sum_j Var(p_j), via conjugacy
    alpha0 = sum(alpha)
    m = np.array(alpha) / alpha0
    label_probs = m
    post_vs = []
    for j in range(3):
        a_new = np.array(alpha, dtype=float)
        a_new[j] += 1
        m_new = a_new / (alpha0 + 1)
        post_vs.append((1 - (m_new**2).sum()) / (alpha0 + 2))
    mc_delta = got.v[0] - float(np.dot(label_probs, post_vs))
    assert got.delta[0] == pytest.approx(mc_delta, rel=1e-6)


# T2 — the DCM-correct null: constant concentration (f_width=0, n fixed),
# shape varying. Entropy-ranking must be near-optimal here. The original
# spec's null (kappa=inf) was wrong — at constant shape the incumbent is
# WORSE than random (see spec changelog); T2b pins that regime too.
def _null_cell(cfg_kwargs, seeds=5):
    cells = []
    for seed in range(seeds):
        draw = generate(DiceConfig(rho=0.0, seed=seed, **cfg_kwargs))
        rng = np.random.default_rng(seed + 500)
        rows = all_regrets(draw.posterior, draw.posterior, draw.alpha, draw.n, rng)
        cells.append(
            {
                row["arm"]: row["regret"]
                for row in rows
                if row["objective"] == "sqerr" and row["budget_pct"] == 10
            }
        )
    return {arm: float(np.mean([c[arm] for c in cells])) for arm in cells[0]}


def test_null_anchor_advantage_small():
    cell = _null_cell(dict(n_items=2000, kappa=1.0, f_width=0.0, n_fixed=5))
    assert cell["incumbent"] - cell["ours_sqerr"] <= 0.02  # [ratify] spec T2
    assert cell["incumbent_prior"] - cell["ours_sqerr"] <= 0.05


def test_epistemic_only_regime_incumbent_fails():
    cell = _null_cell(dict(n_items=2000, kappa=math.inf, f_width=1.0))
    assert cell["incumbent_prior"] >= 0.5  # entropy is blind to concentration
    assert cell["ours_sqerr"] <= 0.05


# T3 — calibration: fails if the generator reverts to truth-first
def test_calibration_pit():
    draw = generate(DiceConfig(n_items=2000, kappa=2.0, rho=0.0, seed=11))
    a1 = draw.alpha[:, 0]
    rest = draw.alpha[:, 1:].sum(axis=1)
    pit = sps.beta.cdf(draw.p[:, 0], a1, rest)
    assert sps.kstest(pit, "uniform").pvalue > 0.01


def test_posterior_coverage():
    draw = generate(DiceConfig(n_items=2000, kappa=2.0, rho=0.0, seed=13))
    a1 = draw.posterior[:, 0]
    rest = draw.posterior[:, 1:].sum(axis=1)
    lo = sps.beta.ppf(0.05, a1, rest)
    hi = sps.beta.ppf(0.95, a1, rest)
    coverage = np.mean((draw.p[:, 0] >= lo) & (draw.p[:, 0] <= hi))
    assert coverage == pytest.approx(0.90, abs=0.02)


# T4 — no negative regret under matched objectives
def test_no_negative_regret():
    draw = generate(DiceConfig(n_items=2000, kappa=1.0, rho=0.0, seed=17))
    rng = np.random.default_rng(17)
    for row in all_regrets(draw.posterior, draw.posterior, draw.alpha, draw.n, rng):
        if row["arm"] == "oracle":
            assert row["regret"] == pytest.approx(0.0, abs=1e-12)
        else:
            assert row["regret"] >= -1e-12


# T5 — copula sanity
def test_copula_independence_at_rho_zero():
    draw = generate(DiceConfig(n_items=5000, kappa=1.0, rho=0.0, seed=19))
    sp = sps.spearmanr(draw.lam, -np.sum(draw.q**2, axis=1)).statistic
    assert abs(sp) < 0.05


def test_copula_positive_at_rho_high():
    draw = generate(DiceConfig(n_items=5000, kappa=1.0, rho=0.8, seed=19))
    entropy_q = -np.sum(np.where(draw.q > 0, draw.q * np.log(draw.q), 0.0), axis=1)
    sp = sps.spearmanr(entropy_q, -draw.lam).statistic
    assert sp > 0.5


# T6 — proxy monotonicity
def test_proxy_informativeness_monotone():
    draw = generate(DiceConfig(n_items=5000, kappa=1.0, rho=0.0, seed=23))
    rng = np.random.default_rng(23)
    correlations = [
        sps.spearmanr(corrupt_familiarity(draw, r, rng), draw.f).statistic
        for r in (0.0, 0.3, 0.6, 0.9, 1.0)
    ]
    assert abs(correlations[0]) < 0.05
    assert correlations[-1] > 0.999
    assert all(b > a for a, b in zip(correlations, correlations[1:]))


# regret sanity on a hand-built case
def test_regret_hand_case():
    values = np.array([10.0, 5.0, 1.0, 0.0])
    assert regret(values, np.array([0.0, 1.0, 2.0, 3.0]), 25) == pytest.approx(1.0)
    assert regret(values, values, 25) == pytest.approx(0.0)
    assert regret(values, np.array([5.0, 10.0, 0.0, 1.0]), 50) == pytest.approx(0.0)


def test_true_values_objectives_differ():
    draw = generate(DiceConfig(n_items=500, kappa=0.5, rho=0.0, seed=29))
    info = true_values(draw.posterior, "info")
    sqerr = true_values(draw.posterior, "sqerr")
    assert sps.spearmanr(info, sqerr).statistic < 1.0
