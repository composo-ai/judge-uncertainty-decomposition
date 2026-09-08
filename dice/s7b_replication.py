"""S7b reveal-resampling replication (WRITEUP_BRIEF honesty constraint 4).

The LLM judged descriptions only, so its cached predictions are valid under
any reveal draw; resampling counts ~ Multinomial(n_i, p_i) and refitting the
DCM gives free replication CIs for every downstream statement. Run 18 Aug
(day 6, during the v1 write-up audit). Two findings this module exists to
record:

1. The single-run arm-advantage table in `s7b_analysis.py` does NOT
   replicate: across 200 reveal draws, no between-arm regret difference at
   any budget separates from zero on the 500-die world (e.g. ours minus
   entropy-ranking at B=10%: +6pp, 95% CI −25..+28). Per-budget advantage
   claims from S7b are therefore out of the paper.
2. The isolation matrix, tier calibration, verbalised-vs-random comparison
   and the diagnostic DO replicate, with the CIs printed here.

Usage (from repo root):
    python -m dice.s7b_replication \
        [--cache dice/caches/s7b_cache.jsonl] \
        [--replicates 200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .llm_judge import TIERS, build_world, load_cache, q_hat_matrix
from .posterior import dirichlet_stats, estimated_posterior, fit_dcm
from .s7a import BUDGETS_PCT, prophet_values
from .s7b_analysis import confidence_vector, diagnostic_vector

ARMS = (
    "ours_sqerr",
    "v_only",
    "incumbent",
    "incumbent_prior",
    "verbalised",
    "practitioner",
    "random",
)


def entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def ci(a: list[float]) -> str:
    arr = np.array(a)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return f"{arr.mean():.3f} ({lo:.3f},{hi:.3f})"


def run(cache_path: Path, replicates: int, n_dice: int = 500, seed: int = 0) -> None:
    world = build_world(n_dice, seed)
    cache = load_cache(cache_path)
    q_hat, spread = q_hat_matrix(world, cache_path)
    conf = confidence_vector(world, cache)
    conf_f = np.where(np.isnan(conf), np.nanmean(conf), conf)
    spread_f = np.where(np.isnan(spread), np.nanmean(spread), spread)
    tier_idx = np.array([TIERS.index(t) for t in world.tier])

    def norm(z: np.ndarray) -> np.ndarray:
        return (z - z.mean(0)) / np.maximum(z.std(0), 1e-9)

    feats = {
        "base": norm(np.column_stack([spread_f, np.log1p(world.n), conf_f])),
        "ood": norm(
            np.column_stack(
                [np.eye(4)[tier_idx][:, 1:], spread_f, np.log1p(world.n), conf_f]
            )
        ),
    }
    prior_entropy = entropy(q_hat)
    true_alea = entropy(world.p)

    regrets = {fs: {arm: {b: [] for b in BUDGETS_PCT} for arm in ARMS} for fs in feats}
    iso = {k: [] for k in ("aa", "ae", "ea", "ee", "tt", "inc_a", "inc_e")}
    for r in range(replicates):
        rng = np.random.default_rng(10_000 + r)
        counts = np.zeros_like(world.counts)
        for i in range(n_dice):
            if world.n[i] > 0:
                counts[i] = rng.multinomial(int(world.n[i]), world.p[i])
        for fs, z in feats.items():
            beta = fit_dcm(z, q_hat, counts)
            post = estimated_posterior(beta, z, q_hat, counts)
            est = dirichlet_stats(post)
            values = prophet_values(post, world.p, rng)
            tie = rng.random(n_dice) * 1e-9
            scores = {
                "ours_sqerr": est.delta,
                "v_only": est.v,
                "incumbent": est.total,
                "incumbent_prior": prior_entropy + tie,
                "verbalised": -conf_f + tie,
                "practitioner": -world.n.astype(float) + tie,
                "random": rng.random(n_dice),
            }
            oracle_order = np.argsort(-values)
            for arm in ARMS:
                order = np.argsort(-scores[arm])
                for b in BUDGETS_PCT:
                    k = max(1, n_dice * b // 100)
                    top, otop = order[:k], oracle_order[:k]
                    reg = (values[otop].sum() - values[top].sum()) / max(
                        values[otop].sum(), 1e-12
                    )
                    regrets[fs][arm][b].append(float(reg))
            if fs == "ood":
                err = np.abs(post / post.sum(1, keepdims=True) - world.p).sum(1)
                iso["aa"].append(spearmanr(st := est.aleatoric, true_alea).statistic)
                iso["ae"].append(spearmanr(st, err).statistic)
                iso["ea"].append(spearmanr(est.info, true_alea).statistic)
                iso["ee"].append(spearmanr(est.info, err).statistic)
                iso["tt"].append(spearmanr(true_alea, err).statistic)
                iso["inc_a"].append(spearmanr(prior_entropy, true_alea).statistic)
                iso["inc_e"].append(spearmanr(prior_entropy, err).statistic)

    for fs in feats:
        print(f"\n=== arm regrets, features={fs} (R={replicates}) ===")
        for arm in ARMS:
            print(
                f"  {arm:<16}"
                + "  ".join(f"B={b}%: {ci(regrets[fs][arm][b])}" for b in BUDGETS_PCT)
            )
        for b in BUDGETS_PCT:
            adv = [
                100 * (i - o)
                for i, o in zip(
                    regrets[fs]["incumbent_prior"][b], regrets[fs]["ours_sqerr"][b]
                )
            ]
            arr = np.array(adv)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            print(
                f"  ours-vs-entropy advantage B={b}%: "
                f"{arr.mean():.1f}pp CI ({lo:.1f},{hi:.1f})"
            )
    verb_rand = [
        100 * (v - r)
        for v, r in zip(regrets["ood"]["verbalised"][10], regrets["ood"]["random"][10])
    ]
    arr = np.array(verb_rand)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(
        f"\nverbalised minus random regret at B=10% (paired, ood): "
        f"{arr.mean():.1f}pp CI ({lo:.1f},{hi:.1f})"
    )

    print("\n=== isolation matrix over reveal draws (ood features) ===")
    for key, label in (
        ("aa", "est-aleatoric vs true-aleatoric"),
        ("ae", "est-aleatoric vs true-error"),
        ("ea", "est-epistemic vs true-aleatoric"),
        ("ee", "est-epistemic vs true-error"),
        ("tt", "truth-truth baseline"),
        ("inc_a", "H(q_hat) vs true-aleatoric"),
        ("inc_e", "H(q_hat) vs true-error"),
    ):
        print(f"  {label:<32} {ci(iso[key])}")

    diag = diagnostic_vector(world, cache)
    idx = np.array(sorted(diag))
    llm_updated = np.stack([diag[i] for i in idx])
    print(
        f"\ndiagnostic (n={len(idx)}): "
        f"llm_self_update_error={np.abs(llm_updated - world.p[idx]).sum(1).mean():.3f}"
    )
    for fs, z in feats.items():
        beta = fit_dcm(z, q_hat, world.counts)
        post = estimated_posterior(beta, z, q_hat, world.counts)
        conj = post[idx] / post[idx].sum(1, keepdims=True)
        print(
            f"  conjugate_update_error ({fs}): "
            f"{np.abs(conj - world.p[idx]).sum(1).mean():.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("dice/caches/s7b_cache.jsonl"),
    )
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args(argv)
    run(args.cache, args.replicates)


if __name__ == "__main__":
    main()
