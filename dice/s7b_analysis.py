"""S7b analysis — the real LLM judge scored against exact truth
(experiments_synthetic_dice.md §8; the gate before releasing the ChaosNLI
cache run, per discussion 18 Aug).

Sections:
  1. Tier calibration (P1): LLM prediction error, stated confidence and
     variant spread per description-informativeness tier.
  2. Placement on the S7a map: the LLM's effective miscalibration —
     Spearman between its confidence signals and its actual error, located
     between honest (strong +correlation of spread with error), flat (0)
     and inverted (-).
  3. Arm evaluation: DCM posterior from the LLM's q_hat + revealed rolls,
     all ranking arms against the exact prophet oracle.
  4. Diagnostic: LLM self-updating vs conjugate updating on its own prior.

Usage (from repo root):
    python -m dice.s7b_analysis --cache <s7b_cache.jsonl>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats as sps

from .llm_judge import TIERS, build_world, load_cache, q_hat_matrix
from .posterior import dirichlet_stats, estimated_posterior, fit_dcm
from .s7a import BUDGETS_PCT, prophet_values

ARMS = (
    "oracle",
    "ours_sqerr",
    "v_only",
    "incumbent",
    "incumbent_prior",
    "verbalised",
    "practitioner",
    "random",
)


def confidence_vector(world, cache: dict) -> np.ndarray:
    out = np.full(len(world.tier), np.nan)
    acc: dict[int, list[float]] = {}
    for row in cache.values():
        if row.get("confidence") is not None:
            acc.setdefault(row["idx"], []).append(float(row["confidence"]))
    for i in range(len(world.tier)):
        if i in acc:
            out[i] = float(np.mean(acc[i]))
    return out


def diagnostic_vector(world, cache: dict) -> dict[int, np.ndarray]:
    n = len(world.tier)
    out: dict[int, np.ndarray] = {}
    for row in cache.values():
        if (
            row.get("kind") == "diagnostic"
            and row.get("dist") is not None
            and row["idx"] < n
        ):
            out[row["idx"]] = np.array(row["dist"])
    return out


def analyse(cache_path: Path, n_dice: int = 500, seed: int = 0) -> dict:
    world = build_world(n_dice, seed)
    cache = load_cache(cache_path)
    q_hat, spread = q_hat_matrix(world, cache_path)
    confidence = confidence_vector(world, cache)
    err = np.abs(q_hat - world.p).sum(axis=1)

    report: dict = {}

    # 1 — tier calibration (P1)
    tiers = np.array(world.tier)
    tier_rows = {}
    for tier in TIERS:
        mask = tiers == tier
        tier_rows[tier] = {
            "n": int(mask.sum()),
            "llm_error": round(float(err[mask].mean()), 3),
            "stated_confidence": round(float(np.nanmean(confidence[mask])), 1),
            "variant_spread": round(float(np.nanmean(spread[mask])), 4),
        }
    report["tier_calibration"] = tier_rows

    # P1 verdict inputs: is confidence flat/high while error varies by tier?
    conf_ok = ~np.isnan(confidence)
    report["spearman_confidence_vs_error"] = round(
        float(sps.spearmanr(confidence[conf_ok], err[conf_ok]).statistic), 3
    )
    report["spearman_spread_vs_error"] = round(
        float(sps.spearmanr(spread[conf_ok], err[conf_ok]).statistic), 3
    )

    # 2 — effective-m placement: honest => spread tracks error strongly (+),
    # flat => ~0, inverted => negative. Reference points from S7a's design.
    report["s7a_placement"] = {
        "honest_reference": "+0.5..+0.8",
        "flat_reference": "~0",
        "inverted_reference": "negative",
        "llm_spread_signal": report["spearman_spread_vs_error"],
        "llm_confidence_signal": -report["spearman_confidence_vs_error"],
    }

    # 3 — arm evaluation against the prophet oracle, exact truth
    rng = np.random.default_rng(seed + 3_000_000)
    conf_feature = np.where(np.isnan(confidence), np.nanmean(confidence), confidence)
    spread_feature = np.where(np.isnan(spread), np.nanmean(spread), spread)
    features = np.column_stack([spread_feature, np.log1p(world.n), conf_feature])
    features = (features - features.mean(axis=0)) / np.maximum(
        features.std(axis=0), 1e-9
    )
    beta = fit_dcm(features, q_hat, world.counts)
    posterior = estimated_posterior(beta, features, q_hat, world.counts)
    est = dirichlet_stats(posterior)
    values = prophet_values(posterior, world.p, rng)
    post_mean = posterior / posterior.sum(axis=1, keepdims=True)
    post_err = np.abs(post_mean - world.p).sum(axis=1)

    prior_entropy = -np.sum(np.where(q_hat > 0, q_hat * np.log(q_hat), 0.0), axis=1)
    tie = rng.random(n_dice) * 1e-9
    scores = {
        "oracle": values,
        "ours_sqerr": est.delta,
        "v_only": est.v,
        "incumbent": est.total,
        "incumbent_prior": prior_entropy + tie,
        "verbalised": -conf_feature + tie,  # escalate where LLM says unsure
        "practitioner": -world.n.astype(float) + tie,
        "random": rng.random(n_dice),
    }
    arm_rows = {}
    for arm in ARMS:
        order = np.argsort(-scores[arm])
        row = {}
        for budget in BUDGETS_PCT:
            top = order[: max(1, n_dice * budget // 100)]
            oracle_top = np.argsort(-values)[: len(top)]
            row[f"regret_b{budget}"] = round(
                float(
                    (values[oracle_top].sum() - values[top].sum())
                    / max(values[oracle_top].sum(), 1e-12)
                ),
                3,
            )
        arm_rows[arm] = row
    report["arms"] = arm_rows
    report["beta"] = [round(float(b), 3) for b in beta]
    report["mean_posterior_error"] = round(float(post_err.mean()), 3)

    # 4 — diagnostic: LLM self-update vs conjugate update on its own prior
    diag = diagnostic_vector(world, cache)
    if diag:
        idx = np.array(sorted(diag))
        llm_updated = np.stack([diag[i] for i in idx])
        conj = posterior[idx] / posterior[idx].sum(axis=1, keepdims=True)
        report["diagnostic"] = {
            "n_dice": len(idx),
            "llm_self_update_error": round(
                float(np.abs(llm_updated - world.p[idx]).sum(axis=1).mean()), 3
            ),
            "conjugate_update_error": round(
                float(np.abs(conj - world.p[idx]).sum(axis=1).mean()), 3
            ),
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--n-dice", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    report = analyse(args.cache, args.n_dice, args.seed)
    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
