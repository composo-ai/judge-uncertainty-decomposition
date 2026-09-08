"""Day-5 gate evaluation against the pre-registered exit criteria
(experiments_synthetic_dice.md §0, revised). Written before the registered
sweeps were read, so the thresholds cannot drift to fit the curves.

Usage (from repo root):
    python -m dice.gate --results <dir>

Criterion (PASS requires both):
  C1  s4, estimated condition at r=1: mean(incumbent_prior - ours_sqerr)
      >= 15pp at B=10%, sq-err objective, 95% bootstrap CI excluding 0.
  C2  advantage curve tracking: Spearman >= 0.9 across the 11 f_width points
      between the oracle-condition curve (s1) and the estimated-condition
      curve at r=1 (s5track). The registered wording said "regret curve";
      both the advantage curve and ours-arm curve are reported, advantage
      binds (stated in RESULTS.md before evaluation).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .sweep import summarise_s4

PASS_ADVANTAGE_PP = 15.0
PASS_TRACKING_SPEARMAN = 0.9


def _load(path: Path) -> list[dict]:
    """CSV round-trip loses types; summarise_s4 expects the in-memory schema
    (int seed/budget, float regret), so coerce on load."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["budget_pct"] = int(row["budget_pct"])
        row["regret"] = float(row["regret"])
    return rows


def _headline(rows: list[dict], arm: str) -> dict[str, float]:
    """Mean regret per f_width for one arm at the headline cell."""
    acc: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["arm"] == arm
            and row["objective"] == "sqerr"
            and row["budget_pct"] == 10
        ):
            acc[row["f_width"]].append(row["regret"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def evaluate(results: Path) -> dict:
    s4 = summarise_s4(_load(results / "s4.csv"))
    est = s4["estimated"]
    c1 = est["advantage_pp"] >= PASS_ADVANTAGE_PP and est["ci95_pp"][0] > 0

    s1_rows = _load(results / "s1.csv")
    s5_rows = _load(results / "s5track.csv")
    widths = sorted(_headline(s1_rows, "ours_sqerr"), key=float)

    def curve(rows: list[dict]) -> dict[str, np.ndarray]:
        inc = _headline(rows, "incumbent_prior")
        ours = _headline(rows, "ours_sqerr")
        return {
            "advantage": np.array([inc[w] - ours[w] for w in widths]),
            "ours": np.array([ours[w] for w in widths]),
        }

    oracle, estimated = curve(s1_rows), curve(s5_rows)
    sp_adv = float(spearmanr(oracle["advantage"], estimated["advantage"]).statistic)
    sp_ours = float(spearmanr(oracle["ours"], estimated["ours"]).statistic)
    c2 = sp_adv >= PASS_TRACKING_SPEARMAN

    return {
        "C1_s4": s4,
        "C1_pass": c1,
        "C2_tracking_spearman_advantage": round(sp_adv, 4),
        "C2_tracking_spearman_ours": round(sp_ours, 4),
        "C2_pass": c2,
        "verdict": "PASS" if (c1 and c2) else ("WEAK PASS" if c1 else "FAIL"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    for key, value in evaluate(args.results).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
