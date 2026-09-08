"""Paper figures (v1 draft) — Fig 1 (S1 axis + S6 dose-response) and
Fig 2 (S7a robustness), from sweep CSVs regenerated via:

    python -m dice.sweep --sweep s1 --out <r>
    python -m dice.sweep --sweep s6dose --out <r>
    python -m dice.s7a --out <r>
    python -m dice.s7a --out <r> --ood-r 0.5

Usage (from repo root):
    python -m dice.paper_figures \
        --results <r> --out dice/figures
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


def _load(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _mean_regret(rows: list[dict], arm: str, keys: tuple[str, ...]):
    acc: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm:
            continue
        if row.get("objective", HEADLINE["objective"]) != HEADLINE["objective"]:
            continue
        if row["budget_pct"] != HEADLINE["budget_pct"]:
            continue
        acc[tuple(row[k] for k in keys)].append(float(row["regret"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fig1(results: Path, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    s1 = _load(results / "s1.csv")
    labels = {
        "incumbent_prior": "entropy (judge output)",
        "incumbent": "entropy (posterior)",
        "practitioner": "fewest-labels-first",
        "random": "random",
        "ours_sqerr": "ours (epistemic value)",
    }
    for arm, label in labels.items():
        pts = _mean_regret(s1, arm, ("f_width",))
        xs = sorted(pts, key=lambda k: float(k[0]))
        ax1.plot(
            [float(k[0]) for k in xs],
            [100 * pts[k] for k in xs],
            marker="o",
            markersize=3.5,
            label=label,
        )
    ax1.set_xlabel("epistemic heterogeneity (familiarity spread)")
    ax1.set_ylabel("escalation regret at B=10% (pp)")
    ax1.set_title("(a) S1: regret vs epistemic heterogeneity")
    ax1.legend(fontsize=7)

    s6 = _load(results / "s6dose.csv")
    for cond, label in (
        ("oracle", "oracle condition"),
        ("estimated", "estimated, r=1"),
    ):
        sub = [r for r in s6 if r["condition"] == cond]
        inc = _mean_regret(sub, "incumbent_prior", ("lam_ratio",))
        ours = _mean_regret(sub, "ours_sqerr", ("lam_ratio",))
        xs = sorted(inc, key=lambda k: float(k[0]))
        ax2.plot(
            [float(k[0]) for k in xs],
            [100 * (inc[k] - ours[k]) for k in xs],
            marker="o",
            markersize=3.5,
            label=label,
        )
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("concentration range width (×)")
    ax2.set_ylabel("advantage over entropy-ranking (pp)")
    ax2.set_title("(b) S6: dose–response in range width")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out / "fig1_s1_s6.png", dpi=200)
    plt.close(fig)


def fig2(results: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name, label in (
        ("s7a.csv", "judge features only"),
        ("s7a_ood0.5.csv", "+ judge-independent familiarity (r=0.5)"),
    ):
        rows = _load(results / name)
        inc = _mean_regret(rows, "incumbent_prior", ("miscal",))
        ours = _mean_regret(rows, "ours_sqerr", ("miscal",))
        xs = sorted(inc, key=lambda k: float(k[0]))
        ax.plot(
            [float(k[0]) for k in xs],
            [100 * (inc[k] - ours[k]) for k in xs],
            marker="o",
            markersize=4,
            label=label,
        )
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.annotate(
        "dead spot: uninformative confidence,\nfidelity varying silently",
        xy=(1.0, 4.5),
        xytext=(0.42, 15),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel("stated-confidence miscalibration m (0 honest, 1 flat, 1.5 inverted)")
    ax.set_ylabel("advantage over entropy-ranking (pp)")
    ax.set_title("S7a: robustness to judge miscalibration")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig2_s7a_robustness.png", dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    fig1(args.results, args.out)
    fig2(args.results, args.out)


if __name__ == "__main__":
    main()
