"""S7b convergence figure (27 Aug — the paper's hero figure).

x: rolls revealed per die, n = 0..10 swept for every die (the deployed
reveal design draws n_i from {0, 1, 2, 5, 10}; the sweep asks what the
posterior's error is at each evidence level). y: mean Manhattan distance
from the posterior mean to the true face distribution over the 500 dice.

Arms:
  ours          Dir(lam_hat * q_hat + counts). lam_hat is the S7b ood fit
                (tier one-hots + variant spread + log1p(n) + stated
                confidence), fitted per reveal draw on the deployed design
                exactly as s7b_replication.py fits it, then held fixed
                while the sweep reveals rolls cumulatively.
  judge prior   m = q_hat, never updated: a flat line at the judge's error.
  counts only   Dir(1, 1, 1) + counts: add-one smoothing, no judge.

Band: 2.5-97.5 percentiles over reveal draws, seeds 10_000 + r as in
s7b_replication.py. Rolls accumulate within a draw (n = 5 extends the same
draw's first 4), so each curve is one world revealing evidence, not eleven
independent worlds.

Verification: before drawing, the script asserts the regenerated world +
cache reproduce the published per-tier judge-prior errors (Table on
stated confidence: 0.53 / 0.54 / 0.41 / 0.007) to within 0.005 — the
published values are 2-significant-figure roundings, and the realised
diffs run up to 0.0042. A failure means the provenance chain is broken;
nothing is written.

Usage (from repo root):
    python -m dice.convergence_figure \
        [--replicates 200] [--out dice/figures]

Writes fig3_s7b_decomposition.png (the paper's two-panel hero: the
decomposition in motion + the Manhattan convergence curves), plus two
archived alternates (fig3_s7b_convergence.png, single panel;
fig3b_s7b_convergence_predicted.png, predicted-spread overlay), and prints
the curve tables for RESULTS.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .llm_judge import TIERS, build_world, load_cache, q_hat_matrix
from .posterior import dirichlet_stats, fit_dcm, predict_lam
from .s7b_analysis import confidence_vector

N_SWEEP = np.arange(0, 11)
PUBLISHED_TIER_ERRORS = {"none": 0.53, "weak": 0.54, "directional": 0.41, "strong": 0.007}


def manhattan(m: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.abs(m - p).sum(axis=1)


def run(cache_path: Path, replicates: int, out_dir: Path, n_dice: int = 500,
        seed: int = 0) -> None:
    world = build_world(n_dice, seed)
    cache = load_cache(cache_path)
    q_hat, spread = q_hat_matrix(world, cache_path)
    conf = confidence_vector(world, cache)
    conf_f = np.where(np.isnan(conf), np.nanmean(conf), conf)
    spread_f = np.where(np.isnan(spread), np.nanmean(spread), spread)
    tier_idx = np.array([TIERS.index(t) for t in world.tier])

    # Provenance gate: the regenerated world + cache must reproduce the
    # published per-tier judge-prior errors before anything is drawn.
    prior_err = manhattan(q_hat, world.p)
    for tier, published in PUBLISHED_TIER_ERRORS.items():
        got = prior_err[tier_idx == TIERS.index(tier)].mean()
        assert abs(got - published) < 5e-3, (
            f"tier {tier}: regenerated prior error {got:.4f} != published {published}"
        )
    print("provenance check: per-tier prior errors match the published table")

    def norm(z: np.ndarray) -> np.ndarray:
        return (z - z.mean(0)) / np.maximum(z.std(0), 1e-9)

    ood = norm(np.column_stack(
        [np.eye(4)[tier_idx][:, 1:], spread_f, np.log1p(world.n), conf_f]))

    arms = ("ours", "counts_only")
    err = {a: np.zeros((replicates, len(N_SWEEP))) for a in arms}
    pred_v = np.zeros((replicates, len(N_SWEEP)))
    est_total = np.zeros((replicates, len(N_SWEEP)))
    est_alea = np.zeros((replicates, len(N_SWEEP)))
    est_info = np.zeros((replicates, len(N_SWEEP)))
    for r in range(replicates):
        rng = np.random.default_rng(10_000 + r)
        # the deployed reveal draw, exactly as s7b_replication.py draws it,
        # is what the trust regression sees
        counts_dep = np.zeros_like(world.counts)
        for i in range(n_dice):
            if world.n[i] > 0:
                counts_dep[i] = rng.multinomial(int(world.n[i]), world.p[i])
        beta = fit_dcm(ood, q_hat, counts_dep)
        lam_hat = predict_lam(beta, ood)

        # sweep: rolls accumulate one at a time for every die
        counts_n = np.zeros((n_dice, 3))
        for j, n in enumerate(N_SWEEP):
            if n > 0:
                for i in range(n_dice):
                    counts_n[i] += rng.multinomial(1, world.p[i])
            post = lam_hat[:, None] * q_hat + counts_n
            m = post / post.sum(axis=1, keepdims=True)
            err["ours"][r, j] = manhattan(m, world.p).mean()
            stats = dirichlet_stats(post)
            pred_v[r, j] = stats.v.mean()
            est_total[r, j] = stats.total.mean()
            est_alea[r, j] = stats.aleatoric.mean()
            est_info[r, j] = stats.info.mean()
            m_c = (1.0 + counts_n) / (3.0 + n)
            err["counts_only"][r, j] = manhattan(m_c, world.p).mean()

    prior_flat = prior_err.mean()
    true_alea = float(
        (-np.where(world.p > 0, world.p * np.log(world.p), 0.0).sum(axis=1)).mean())
    print(f"\njudge prior alone (flat): {prior_flat:.3f}")
    print(f"true mean pool disagreement H(p): {true_alea:.3f} nats")
    print("n         " + "  ".join(f"{n:>6d}" for n in N_SWEEP))
    for a in arms:
        print(f"{a:<10}" + "  ".join(f"{v:6.3f}" for v in err[a].mean(axis=0)))
    print("pred_v    " + "  ".join(f"{v:6.4f}" for v in pred_v.mean(axis=0)))
    print("est_total " + "  ".join(f"{v:6.3f}" for v in est_total.mean(axis=0)))
    print("est_alea  " + "  ".join(f"{v:6.3f}" for v in est_alea.mean(axis=0)))
    print("est_info  " + "  ".join(f"{v:6.3f}" for v in est_info.mean(axis=0)))

    # THE paper figure (lead's approval, 27 Aug): (a) the decomposition in
    # motion, entropy units; (b) the Manhattan convergence curves.
    styles_b = {
        "ours": ("tab:green", "ours: judge prior at fitted trust + rolls"),
        "counts_only": ("tab:orange", "counts only: rolls, no judge"),
    }
    # Single decomposition panel since the lead executed reserve-ladder item 1
    # (28 Aug): the former panel (b) content remains available via the archived
    # fig3_s7b_convergence.png below and in the TAE build's caption.
    fig, axa = plt.subplots(figsize=(4.8, 2.05))
    for arr, colour, label in (
            (est_total, "black", "total uncertainty $H(m)$"),
            (est_alea, "tab:purple", "estimated aleatoric $\\mathbb{E}[H(p)\\mid c]$"),
            (est_info, "tab:red", "estimated epistemic $I$")):
        mean = arr.mean(axis=0)
        lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
        axa.plot(N_SWEEP, mean, "-o", ms=3, color=colour, label=label)
        axa.fill_between(N_SWEEP, lo, hi, color=colour, alpha=0.15, lw=0)
    axa.axhline(true_alea, color="tab:purple", ls="--", lw=1.5,
                label="true pool disagreement $H(p)$")
    axa.set_xlabel("rolls revealed per die")
    axa.set_ylabel("entropy (nats)")
    axa.set_xticks(N_SWEEP)
    axa.set_ylim(bottom=0)
    axa.legend(fontsize=6, loc="center right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_s7b_decomposition.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / 'fig3_s7b_decomposition.png'}")

    styles = styles_b
    for variant_b in (False, True):
        fig, ax = plt.subplots(figsize=(8, 2.2))
        ax.axhline(prior_flat, color="tab:blue", ls="--", lw=1.5,
                   label="judge prior alone (never updated)")
        for a in arms:
            mean = err[a].mean(axis=0)
            lo, hi = np.percentile(err[a], [2.5, 97.5], axis=0)
            colour, label = styles[a]
            ax.plot(N_SWEEP, mean, "-o", ms=4, color=colour, label=label)
            ax.fill_between(N_SWEEP, lo, hi, color=colour, alpha=0.15, lw=0)
        ax.set_xlabel("rolls revealed per die")
        ax.set_ylabel("mean Manhattan error")
        ax.set_xticks(N_SWEEP)
        ax.set_ylim(bottom=0)
        if variant_b:
            ax2 = ax.twinx()
            ax2.plot(N_SWEEP, pred_v.mean(axis=0), ":", color="tab:red", lw=2,
                     label="model's own predicted spread $v$ (right axis)")
            ax2.set_ylabel("mean posterior spread $v$")
            ax2.set_ylim(bottom=0)
            lines, labels = ax.get_legend_handles_labels()
            l2, lb2 = ax2.get_legend_handles_labels()
            ax.legend(lines + l2, labels + lb2, fontsize=8, loc="upper right")
        else:
            ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        name = ("fig3b_s7b_convergence_predicted.png" if variant_b
                else "fig3_s7b_convergence.png")
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        print(f"wrote {out_dir / name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=Path("dice/caches/s7b_cache.jsonl"))
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument(
        "--out", type=Path,
        default=Path("dice/figures"))
    args = parser.parse_args(argv)
    run(args.cache, args.replicates, args.out)


if __name__ == "__main__":
    main()
