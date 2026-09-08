"""Sweeps S1–S4 — experiments_synthetic_dice.md §6 (as revised after the
day-0 implementation findings; see the spec's changelog).

Primary axis is epistemic (concentration) heterogeneity via f_width — the
DCM-correct null is constant concentration, NOT constant aleatoric entropy.

Usage (from repo root):
    python -m dice.sweep --sweep s1 --out results/
    python -m dice.sweep --sweep smoke --out /tmp/smoke/
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .arms import all_regrets
from .generator import DiceConfig, corrupt_familiarity, generate
from .posterior import dirichlet_stats, estimated_posterior, fit_dcm, predict_lam

F_WIDTHS = tuple(round(0.1 * i, 1) for i in range(11))
KAPPAS_SHAPE = (math.inf, 64, 32, 16, 8, 4, 2, 1, 0.75, 0.6, 0.5)
RHOS_S2 = (-0.8, -0.4, 0.0, 0.4, 0.8)
RS_S3 = tuple(round(0.1 * i, 1) for i in range(11))
F_WIDTHS_S3 = (0.25, 0.5, 1.0)
KAPPA_DEFAULT = 1.0

FIELDS = [
    "sweep",
    "kappa",
    "lam_ratio",
    "f_width",
    "rho",
    "r",
    "seed",
    "condition",
    "objective",
    "arm",
    "budget_pct",
    "regret",
    "sd_aleatoric",
    "sd_epistemic",
    "spearman_aleatoric_info",
    "spearman_fhat_f",
    "n_clamped",
]


def _run_cell(
    sweep: str,
    kappa: float,
    f_width: float,
    rho: float,
    r: float | None,
    seed: int,
    condition: str,
    n_items: int,
    lam_ratio: float = 256.0,
) -> list[dict]:
    draw = generate(
        DiceConfig(
            n_items=n_items,
            kappa=kappa,
            rho=rho,
            f_width=f_width,
            lam_ratio=lam_ratio,
            seed=seed,
        )
    )
    rng = np.random.default_rng(seed + 1_000_000)

    if condition == "oracle":
        est_posterior_alpha = draw.posterior
        est_prior_alpha = draw.alpha
        sp_fhat = ""
    else:
        f_hat = corrupt_familiarity(draw, r if r is not None else 1.0, rng)
        beta = fit_dcm(f_hat[:, None], draw.q, draw.counts)
        est_posterior_alpha = estimated_posterior(
            beta, f_hat[:, None], draw.q, draw.counts
        )
        est_prior_alpha = predict_lam(beta, f_hat[:, None])[:, None] * draw.q
        sp_fhat = round(float(spearmanr(f_hat, draw.f).statistic), 4)

    true_stats = dirichlet_stats(draw.posterior)
    sd_aleatoric = round(float(true_stats.aleatoric.std()), 4)
    sd_epistemic = round(float(true_stats.info.std()), 4)
    sp_ai = round(float(spearmanr(true_stats.aleatoric, true_stats.info).statistic), 4)

    rows = all_regrets(
        draw.posterior, est_posterior_alpha, est_prior_alpha, draw.n, rng
    )
    for row in rows:
        row.update(
            sweep=sweep,
            kappa=kappa,
            lam_ratio=lam_ratio,
            f_width=f_width,
            rho=rho,
            r="" if r is None else r,
            seed=seed,
            condition=condition,
            sd_aleatoric=sd_aleatoric,
            sd_epistemic=sd_epistemic,
            spearman_aleatoric_info=sp_ai,
            spearman_fhat_f=sp_fhat,
            n_clamped=draw.n_clamped,
        )
    return rows


def sweep_s1(n_items: int, seeds: int) -> list[dict]:
    """Primary: epistemic heterogeneity at moderate shape spread."""
    return [
        row
        for f_width in F_WIDTHS
        for seed in range(seeds)
        for row in _run_cell(
            "s1", KAPPA_DEFAULT, f_width, 0.0, None, seed, "oracle", n_items
        )
    ]


def sweep_s1shape(n_items: int, seeds: int) -> list[dict]:
    """Secondary: shape heterogeneity at full epistemic spread."""
    return [
        row
        for kappa in KAPPAS_SHAPE
        for seed in range(seeds)
        for row in _run_cell("s1shape", kappa, 1.0, 0.0, None, seed, "oracle", n_items)
    ]


def sweep_s2(n_items: int, seeds: int) -> list[dict]:
    return [
        row
        for f_width in F_WIDTHS
        for rho in RHOS_S2
        for seed in range(seeds)
        for row in _run_cell(
            "s2", KAPPA_DEFAULT, f_width, rho, None, seed, "oracle", n_items
        )
    ]


def sweep_s3(n_items: int, seeds: int) -> list[dict]:
    return [
        row
        for r in RS_S3
        for f_width in F_WIDTHS_S3
        for seed in range(seeds)
        for row in _run_cell(
            "s3", KAPPA_DEFAULT, f_width, 0.0, r, seed, "estimated", n_items
        )
    ]


def sweep_s4(n_items: int, seeds: int) -> list[dict]:
    # both conditions at the criterion cell; the criterion binds on estimated at r=1
    rows = []
    for seed in range(seeds):
        rows.extend(
            _run_cell("s4", KAPPA_DEFAULT, 1.0, 0.0, None, seed, "oracle", n_items)
        )
        rows.extend(
            _run_cell("s4", KAPPA_DEFAULT, 1.0, 0.0, 1.0, seed, "estimated", n_items)
        )
    return rows


def sweep_s5track(n_items: int, seeds: int) -> list[dict]:
    """Tracking curve for the exit criterion: estimated condition at r=1
    across the full f_width sweep, paired against S1's oracle curve."""
    return [
        row
        for f_width in F_WIDTHS
        for seed in range(seeds)
        for row in _run_cell(
            "s5track", KAPPA_DEFAULT, f_width, 0.0, 1.0, seed, "estimated", n_items
        )
    ]


def sweep_s6dose(n_items: int, seeds: int) -> list[dict]:
    """Dose-response over concentration-range width (anticipated_reviews.md
    R1.4/R4): does the advantage survive when the epistemic range is narrow?
    ratio=256 is the registered regime; 4 is a 2x spread either side of 8."""
    rows = []
    for ratio in (4.0, 16.0, 64.0, 256.0, 1024.0):
        for seed in range(seeds):
            for condition, r in (("oracle", None), ("estimated", 1.0)):
                rows.extend(
                    _run_cell(
                        "s6dose",
                        KAPPA_DEFAULT,
                        1.0,
                        0.0,
                        r,
                        seed,
                        condition,
                        n_items,
                        lam_ratio=ratio,
                    )
                )
    return rows


SWEEPS = {
    "s1": sweep_s1,
    "s1shape": sweep_s1shape,
    "s2": sweep_s2,
    "s3": sweep_s3,
    "s4": sweep_s4,
    "s5track": sweep_s5track,
    "s6dose": sweep_s6dose,
}
DEFAULT_SEEDS = {
    "s1": 200,
    "s1shape": 100,
    "s2": 50,
    "s3": 100,
    "s4": 500,
    "s5track": 100,
    "s6dose": 100,
}


def summarise_s4(rows: list[dict], n_boot: int = 10_000, seed: int = 0) -> dict:
    """Exit-criterion cell (spec §0, revised): bootstrap CI over seeds of the
    incumbent_prior-minus-ours advantage at B=10%, sq-err objective, in the
    ESTIMATED-posterior condition at r=1. Oracle reported alongside."""
    out: dict = {}
    for condition in ("estimated", "oracle"):
        per_seed: dict[int, dict[str, float]] = {}
        for row in rows:
            if (
                row["condition"] == condition
                and row["objective"] == "sqerr"
                and row["budget_pct"] == 10
                and row["arm"] in ("incumbent_prior", "ours_sqerr")
            ):
                per_seed.setdefault(row["seed"], {})[row["arm"]] = row["regret"]
        adv = np.array(
            [cell["incumbent_prior"] - cell["ours_sqerr"] for cell in per_seed.values()]
        )
        rng = np.random.default_rng(seed)
        boot = rng.choice(adv, size=(n_boot, len(adv)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[condition] = {
            "advantage_pp": round(float(adv.mean()) * 100, 2),
            "ci95_pp": (round(float(lo) * 100, 2), round(float(hi) * 100, 2)),
            "n_seeds": len(adv),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=[*SWEEPS, "smoke"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-items", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args(argv)

    if args.sweep == "smoke":
        rows = sweep_s1(200, 2) + sweep_s3(200, 2)
        name = "smoke"
    else:
        seeds = args.seeds or DEFAULT_SEEDS[args.sweep]
        rows = SWEEPS[args.sweep](args.n_items, seeds)
        name = args.sweep

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{name}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}", file=sys.stderr)

    if args.sweep == "s4":
        print(summarise_s4(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
