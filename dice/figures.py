"""Figures from sweep CSVs — experiments_synthetic_dice.md §7 (revised axes).

Usage (from repo root):
    python -m dice.figures --results results/ --out figures/

Reads whichever of s1/s2/s3 CSVs exist; skips figures whose inputs are absent.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HEADLINE = {"objective": "sqerr", "budget_pct": "10"}
PLOT_ARMS = (
    "incumbent_prior",
    "incumbent",
    "ours_sqerr",
    "v_only",
    "practitioner",
    "random",
)


def _load(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _cell_mean(rows: list[dict], keys: tuple[str, ...], value: str = "regret"):
    acc: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["objective"] != HEADLINE["objective"]:
            continue
        if row["budget_pct"] != HEADLINE["budget_pct"]:
            continue
        if row[value] == "":
            continue
        acc[tuple(row[k] for k in keys)].append(float(row[value]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fig_s1(rows: list[dict], out: Path) -> None:
    """Regret per arm vs realised epistemic heterogeneity (f_width sweep)."""
    sd_by_width = _cell_mean(rows, ("f_width",), value="sd_epistemic")
    fig, ax = plt.subplots(figsize=(6, 4))
    for arm in PLOT_ARMS:
        pts = _cell_mean([r for r in rows if r["arm"] == arm], ("f_width",))
        xs = sorted(pts, key=lambda k: sd_by_width[k])
        ax.plot(
            [sd_by_width[k] for k in xs],
            [pts[k] for k in xs],
            marker="o",
            label=arm,
        )
    ax.set_xlabel("realised sd of epistemic term I across items")
    ax.set_ylabel(f"regret at B={HEADLINE['budget_pct']}% ({HEADLINE['objective']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_s1_epistemic_axis.png", dpi=150)
    plt.close(fig)


def fig_phase(rows: list[dict], out: Path) -> None:
    """S2: advantage (incumbent_prior - ours) over (f_width, rho)."""
    inc = _cell_mean(
        [r for r in rows if r["arm"] == "incumbent_prior"], ("f_width", "rho")
    )
    ours = _cell_mean([r for r in rows if r["arm"] == "ours_sqerr"], ("f_width", "rho"))
    widths = sorted({w for w, _ in inc}, key=float)
    rhos = sorted({r for _, r in inc}, key=float)
    grid = np.array(
        [[(inc[(w, r)] - ours[(w, r)]) * 100 for w in widths] for r in rhos]
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdBu_r")
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths, rotation=45, fontsize=7)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels(rhos, fontsize=8)
    ax.set_xlabel("f_width (epistemic heterogeneity)")
    ax.set_ylabel("rho (aleatoric–epistemic correlation)")
    fig.colorbar(im, label="advantage of ours over incumbent_prior (pp)")
    fig.tight_layout()
    fig.savefig(out / "fig1_phase.png", dpi=150)
    plt.close(fig)


def fig_rho_star(rows: list[dict], out: Path) -> None:
    """S3: advantage vs realised proxy informativeness, per f_width."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for f_width in sorted({row["f_width"] for row in rows}, key=float):
        sub = [r for r in rows if r["f_width"] == f_width]
        inc = _cell_mean([r for r in sub if r["arm"] == "incumbent_prior"], ("r",))
        ours = _cell_mean([r for r in sub if r["arm"] == "ours_sqerr"], ("r",))
        sp = _cell_mean(sub, ("r",), value="spearman_fhat_f")
        rs = sorted(inc, key=lambda k: float(k[0]))
        ax.plot(
            [sp[r] for r in rs],
            [(inc[r] - ours[r]) * 100 for r in rs],
            marker="o",
            label=f"f_width={f_width}",
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("realised Spearman(f_hat, f)")
    ax.set_ylabel("advantage over incumbent_prior (pp)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_rho_star.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = {"s1.csv": fig_s1, "s2.csv": fig_phase, "s3.csv": fig_rho_star}
    for name, fn in jobs.items():
        path = args.results / name
        if path.exists():
            fn(_load(path), args.out)


if __name__ == "__main__":
    main()
