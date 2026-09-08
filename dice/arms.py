"""Ranking arms and regret — experiments_synthetic_dice.md §4.

Oracle value and regret must share one objective; mixing them permits
negative regret (the spec's T4 guards this).
"""

from __future__ import annotations

import numpy as np

from .posterior import DirichletStats, dirichlet_stats

OBJECTIVES = ("info", "sqerr")
# incumbent       = entropy of the estimated posterior (labels folded in)
# incumbent_prior = entropy of the judge's prior mean, no labels — what
#                   Trust-or-Escalate-style disagreement escalation actually uses
ARMS = (
    "oracle",
    "ours_info",
    "ours_sqerr",
    "v_only",
    "practitioner",
    "incumbent",
    "incumbent_prior",
    "random",
)
BUDGETS_PCT = (1, 5, 10, 20)


def true_values(true_posterior: np.ndarray, objective: str) -> np.ndarray:
    stats = dirichlet_stats(true_posterior)
    if objective == "info":
        return stats.info
    if objective == "sqerr":
        return stats.delta
    raise ValueError(f"unknown objective {objective!r}")


def arm_scores(
    est: DirichletStats,
    est_prior: DirichletStats,
    n: np.ndarray,
    rng: np.random.Generator,
    oracle_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Escalation priority per arm, higher = escalate first."""
    n_items = len(n)
    tie = rng.random(n_items) * 1e-9
    return {
        "oracle": oracle_values,
        "ours_info": est.info,
        "ours_sqerr": est.delta,
        "v_only": est.v,
        "practitioner": -n.astype(float) + tie,
        "incumbent": est.total,
        "incumbent_prior": est_prior.total + tie,
        "random": rng.random(n_items),
    }


def regret(values: np.ndarray, scores: np.ndarray, budget_pct: int) -> float:
    n_top = max(1, int(len(values) * budget_pct / 100))
    oracle_top = np.argpartition(values, -n_top)[-n_top:]
    arm_top = np.argpartition(scores, -n_top)[-n_top:]
    oracle_sum = values[oracle_top].sum()
    return float((oracle_sum - values[arm_top].sum()) / oracle_sum)


def all_regrets(
    true_posterior: np.ndarray,
    est_posterior: np.ndarray,
    est_prior: np.ndarray,
    n: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """One row per (objective, arm, budget)."""
    est = dirichlet_stats(est_posterior)
    prior = dirichlet_stats(est_prior)
    rows = []
    for objective in OBJECTIVES:
        values = true_values(true_posterior, objective)
        scores = arm_scores(est, prior, n, rng, oracle_values=values)
        for arm in ARMS:
            for budget in BUDGETS_PCT:
                rows.append(
                    {
                        "objective": objective,
                        "arm": arm,
                        "budget_pct": budget,
                        "regret": regret(values, scores[arm], budget),
                    }
                )
    return rows


# NOTE: the original spec's "predicted regret" overlay (incumbent regret from
# n=0 generator quantities) was defined against the pre-revision null and is
# superseded: the theory curve is now the oracle-posterior condition itself,
# compared against the estimated-posterior condition at r=1 (spec changelog).
