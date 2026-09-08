"""ChaosNLI registered analysis — executes PREREG.md, nothing else.

Every number is a mean over R reveal-resamples with 95% percentile CIs;
between-arm comparisons are paired within replicate. Two conditions:
full features (P-C1..P-C3) and counts-only (P-C4).

Usage (from repo root):
    python -m chaosnli.analysis \
        --data-dir <chaosNLI_v1.0> \
        [--cache chaosnli/caches/judge_cache.jsonl] \
        [--replicates 200] [--procs 8] [--out <report.json>]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .adapter import load
from .judge import confidence_matrix, q_hat_matrix
from .pipeline import BUDGETS_PCT, run_pipeline

REPO = Path(__file__).resolve().parent
_STATE: dict = {}


def _init(data_dir: str, cache: str) -> None:
    data = load(Path(data_dir))
    q_hat, spread = q_hat_matrix(data, Path(cache))
    conf = confidence_matrix(data, Path(cache))
    emb_path = REPO / "embeddings_minilm.npz"
    embeddings = (
        np.load(emb_path)["embeddings"].astype(np.float64)
        if emb_path.exists()
        else None
    )
    _STATE.update(
        counts=data.counts, q_hat=q_hat, spread=spread, conf=conf, emb=embeddings
    )


def _one(args: tuple[int, bool]) -> dict:
    seed, counts_only = args
    r = run_pipeline(
        _STATE["counts"],
        _STATE["q_hat"],
        _STATE["spread"],
        _STATE["emb"],
        seed,
        confidence_score=_STATE["conf"],
        counts_only=counts_only,
    )
    return {
        "selective": r.selective,
        "acquisition": r.acquisition,
        "diagnostics": r.diagnostics,
    }


def _ci(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {
        "mean": round(float(a.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
    }


def _paired(
    reps: list[dict], table: str, metric: str, arm_a: str, arm_b: str, budget: int
) -> dict:
    """CI of (arm_a - arm_b) on `metric`, paired within replicate."""
    diffs = []
    for rep in reps:
        rows = {(r["arm"], r["budget_pct"]): r[metric] for r in rep[table]}
        diffs.append(rows[(arm_a, budget)] - rows[(arm_b, budget)])
    return _ci(diffs)


def analyse(data_dir: Path, cache: Path, replicates: int, procs: int) -> dict:
    _init(str(data_dir), str(cache))
    report: dict = {"replicates": replicates}

    # Reveal-independent P-C3 inputs: judge raw error vs the full 100-label
    # frequencies; stated confidence and prior entropy against it.
    freq = _STATE["counts"] / _STATE["counts"].sum(1, keepdims=True)
    q = _STATE["q_hat"]
    raw_err = np.abs(q - freq).sum(1)
    prior_entropy = -np.sum(np.where(q > 0, q * np.log(q), 0.0), axis=1)
    stated_conf = -_STATE["conf"]
    report["p_c3_correlations"] = {
        "spearman_confidence_vs_raw_error": round(
            float(spearmanr(stated_conf, raw_err).statistic), 3
        ),
        "spearman_prior_entropy_vs_raw_error": round(
            float(spearmanr(prior_entropy, raw_err).statistic), 3
        ),
        "spearman_spread_vs_raw_error": round(
            float(spearmanr(_STATE["spread"], raw_err).statistic), 3
        ),
        "mean_stated_confidence": round(float(stated_conf.mean()), 1),
        "mean_raw_error": round(float(raw_err.mean()), 3),
    }

    with ProcessPoolExecutor(
        max_workers=procs,
        initializer=_init,
        initargs=(str(data_dir), str(cache)),
    ) as pool:
        full = list(pool.map(_one, [(s, False) for s in range(replicates)]))
        counts_only = list(pool.map(_one, [(s, True) for s in range(replicates)]))

    # P-C1 — regime placement
    report["p_c1_regime"] = {
        "sd_log_lam": _ci([r["diagnostics"]["sd_log_lam"] for r in full]),
        "quadrant_occupancy": _ci(
            [r["diagnostics"]["quadrant_occupancy"] for r in full]
        ),
        "condition_met_sd_ge_0.5": bool(
            np.mean([r["diagnostics"]["sd_log_lam"] for r in full]) >= 0.5
        ),
    }

    # Arm tables, both metrics, all registered budgets
    arms = sorted({row["arm"] for row in full[0]["acquisition"]})
    for table, metric in (
        ("acquisition", "value"),
        ("selective", "auto_resolved_sqerr_debiased"),
        ("selective", "auto_resolved_error"),
    ):
        key = f"{table}_{metric}"
        report[key] = {
            arm: {
                f"b{b}": _ci(
                    [
                        {(r["arm"], r["budget_pct"]): r[metric] for r in rep[table]}[
                            (arm, b)
                        ]
                        for rep in full
                    ]
                )
                for b in BUDGETS_PCT
            }
            for arm in arms
        }

    # P-C2 — the headline paired comparison (+ secondary pairs)
    report["p_c2_headline"] = {
        "ours_minus_incumbent_prior_acqvalue": {
            f"b{b}": _paired(
                full, "acquisition", "value", "ours_sqerr", "incumbent_prior", b
            )
            for b in BUDGETS_PCT
        },
        "ours_minus_incumbent_acqvalue": {
            f"b{b}": _paired(full, "acquisition", "value", "ours_sqerr", "incumbent", b)
            for b in BUDGETS_PCT
        },
    }
    # P-C3 — verbalised arm vs incumbent_prior
    report["p_c3_verbalised_minus_incumbent_prior_acqvalue"] = {
        f"b{b}": _paired(
            full, "acquisition", "value", "verbalised", "incumbent_prior", b
        )
        for b in BUDGETS_PCT
    }
    # P-C4 — counts-only backbone vs entropy-ranking (paired across the two
    # condition runs at the same seed: same reveal draw by construction)
    diffs_c4 = []
    for rep_c, rep_f in zip(counts_only, full):
        rows_c = {(r["arm"], r["budget_pct"]): r["value"] for r in rep_c["acquisition"]}
        rows_f = {(r["arm"], r["budget_pct"]): r["value"] for r in rep_f["acquisition"]}
        diffs_c4.append(rows_c[("ours_sqerr", 10)] - rows_f[("incumbent_prior", 10)])
    report["p_c4_countsonly_minus_incumbent_prior_acqvalue_b10"] = _ci(diffs_c4)

    report["mean_beta_full"] = [
        round(float(b), 3)
        for b in np.mean([r["diagnostics"]["beta"] for r in full], axis=0)
    ]
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "chaosnli/caches/judge_cache.jsonl"
        ),
    )
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--procs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = analyse(args.data_dir, args.cache, args.replicates, args.procs)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
