"""ChaosNLI evaluation pipeline (paper §4) — judge-cache in, results out.

Consumes only artifacts: the label counts (adapter), a q_hat matrix with
per-item ensemble spread (judge.q_hat_matrix, or a mock), and optionally an
embedding matrix. Everything downstream of the judge cache lives here so
the real cache can be dropped in with zero further engineering.

Protocol per replicate:
  1. half_split: half A supplies revealed labels (and the acquisition pool),
     half B is evaluation truth — D_x is only ever an evaluation quantity.
  2. Reveal n_i ~ {0,1,2,5,10} labels per item from half A.
  3. Fit the DCM concentration regression log lam = beta0 + beta^T z on the
     revealed counts (features modular; see build_features).
  4. Posterior Dir(lam_hat * q_hat + revealed counts) -> arm scores.
  5. Selective evaluation: error (Manhattan to half-B frequencies) of the
     auto-resolved set at each budget, per arm.
  6. Acquisition evaluation: escalated items receive one extra label from
     the UNREVEALED remainder of half A (never half B), and the value of an
     arm is the total error reduction its top-B purchase achieves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dice.posterior import (
    dirichlet_stats,
    estimated_posterior,
    fit_dcm,
    predict_lam,
)
from .adapter import LABELS_PER_ITEM, debiased_squared_error, half_split, manhattan

N_CHOICES = np.array([0, 1, 2, 5, 10])
BUDGETS_PCT = (1, 5, 10, 20)
ARMS = (
    "ours_info",
    "ours_sqerr",
    "v_only",
    "incumbent",
    "incumbent_prior",
    "verbalised",  # requires a confidence vector; skipped when absent (mock runs)
    "practitioner",
    "random",
)


@dataclass
class PipelineResult:
    selective: list[dict]  # rows: arm, budget_pct, auto_resolved_error
    acquisition: list[dict]  # rows: arm, budget_pct, value (error reduction)
    scatter: dict  # per-item aleatoric/info/arm-selection masks for Fig 2
    diagnostics: dict


def reveal(
    half_a: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Reveal n_i labels per item from half A via hypergeometric subsampling."""
    n = rng.choice(N_CHOICES, size=half_a.shape[0])
    revealed = np.zeros_like(half_a)
    remaining_draw = n.copy()
    remaining_pool = half_a.sum(axis=1).copy()
    for j in range(half_a.shape[1]):
        take = rng.hypergeometric(
            ngood=half_a[:, j],
            nbad=np.maximum(remaining_pool - half_a[:, j], 0),
            nsample=np.minimum(remaining_draw, remaining_pool),
        )
        take = np.minimum(take, half_a[:, j])
        revealed[:, j] = take
        remaining_draw -= take
        remaining_pool -= half_a[:, j]
    return revealed, n


def build_features(
    spread: np.ndarray,
    revealed: np.ndarray,
    embeddings: np.ndarray | None,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Standardised feature matrix for the concentration regression.

    Always: ensemble spread, log1p(revealed count). With embeddings: distance
    to the nearest labelled item and local labelled density (the OOD channel,
    modelling.md §2's required component). With confidence: the judge's raw
    stated confidence (S7b feature parity; the fit decides its weight).
    """
    n = revealed.sum(axis=1)
    cols = [spread, np.log1p(n)]
    if confidence is not None:
        cols.append(np.where(np.isnan(confidence), np.nanmean(confidence), confidence))
    if embeddings is not None:
        labelled = n > 0
        if labelled.any():
            from scipy.spatial.distance import cdist

            d = cdist(embeddings, embeddings[labelled], metric="cosine")
            nearest = np.sort(d, axis=1)
            cols.append(nearest[:, 0])
            k = min(10, labelled.sum())
            cols.append(nearest[:, :k].mean(axis=1))
    z = np.column_stack(cols)
    z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-9)
    return z


def _arm_scores(
    posterior: np.ndarray,
    q_hat: np.ndarray,
    n: np.ndarray,
    rng: np.random.Generator,
    verbalised: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    est = dirichlet_stats(posterior)
    prior_entropy = -np.sum(np.where(q_hat > 0, q_hat * np.log(q_hat), 0.0), axis=1)
    tie = rng.random(len(n)) * 1e-9
    scores = {
        "ours_info": est.info,
        "ours_sqerr": est.delta,
        "v_only": est.v,
        "incumbent": est.total,
        "incumbent_prior": prior_entropy + tie,
        "practitioner": -n.astype(float) + tie,
        "random": rng.random(len(n)),
    }
    if verbalised is not None:
        # judge.confidence_matrix output: -mean stated confidence, so higher
        # = judge less confident = escalate first; nan (no reply) -> mean
        scores["verbalised"] = (
            np.where(np.isnan(verbalised), np.nanmean(verbalised), verbalised) + tie
        )
    return scores


def run_pipeline(
    counts: np.ndarray,
    q_hat: np.ndarray,
    spread: np.ndarray,
    embeddings: np.ndarray | None,
    seed: int,
    confidence_score: np.ndarray | None = None,
    counts_only: bool = False,
) -> PipelineResult:
    """confidence_score: judge.confidence_matrix output (-mean stated
    confidence, nan when absent). Enables the verbalised arm and adds raw
    confidence to the DCM features; None (mock runs) skips both.
    counts_only: PREREG P-C4 — the regression sees intercept + log1p(n)
    only (no judge-side or embedding features); everything else identical."""
    rng = np.random.default_rng(seed)
    half_a, half_b = half_split(counts, rng)
    truth = half_b / half_b.sum(axis=1, keepdims=True)

    revealed, n = reveal(half_a, rng)
    conf_raw = None if confidence_score is None else -confidence_score
    if counts_only:
        features = np.log1p(revealed.sum(axis=1))[:, None]
        features = (features - features.mean(0)) / np.maximum(features.std(0), 1e-9)
    else:
        features = build_features(spread, revealed, embeddings, conf_raw)
    beta = fit_dcm(features, q_hat, revealed)
    lam_prior = predict_lam(beta, features)
    prior_entropy = -np.sum(np.where(q_hat > 0, q_hat * np.log(q_hat), 0.0), axis=1)
    posterior = estimated_posterior(beta, features, q_hat, revealed)
    post_mean = posterior / posterior.sum(axis=1, keepdims=True)
    err = manhattan(post_mean, truth)
    err_debiased = debiased_squared_error(post_mean, half_b)

    scores = _arm_scores(posterior, q_hat, n, rng, verbalised=confidence_score)
    n_items = counts.shape[0]

    selective, acquisition = [], []
    scatter_masks = {}
    stats = dirichlet_stats(posterior)
    for arm in (a for a in ARMS if a in scores):
        order = np.argsort(-scores[arm])
        for budget in BUDGETS_PCT:
            top = order[: max(1, n_items * budget // 100)]
            mask = np.zeros(n_items, dtype=bool)
            mask[top] = True
            selective.append(
                {
                    "arm": arm,
                    "budget_pct": budget,
                    "auto_resolved_error": float(err[~mask].mean()),
                    "escalated_error": float(err[mask].mean()),
                    # finite-k corrected companion metric (paper_plan.md §6):
                    # unbiased for the true squared error, unlike Manhattan
                    # against the k-label empirical distribution
                    "auto_resolved_sqerr_debiased": float(err_debiased[~mask].mean()),
                }
            )
            acquisition.append(
                {
                    "arm": arm,
                    "budget_pct": budget,
                    "value": _acquisition_value(
                        top, half_a, revealed, posterior, truth, rng
                    ),
                }
            )
            if budget == 10:
                scatter_masks[arm] = mask
    return PipelineResult(
        selective=selective,
        acquisition=acquisition,
        scatter={
            "aleatoric": stats.aleatoric,
            "info": stats.info,
            "err": err,
            "n": n,
            "masks_b10": scatter_masks,
        },
        diagnostics={
            "beta": beta.tolist(),
            "mean_error_all": float(err.mean()),
            "n_labelled": int((n > 0).sum()),
            # PREREG P-C1 instrumentation: concentration heterogeneity of the
            # PRIOR lambda (before counts) and occupancy of the confidently-
            # predicted-split quadrant (top-half prior lambda AND top-half
            # predicted entropy)
            "sd_log_lam": float(np.std(np.log(lam_prior))),
            "quadrant_occupancy": float(
                (
                    (lam_prior >= np.median(lam_prior))
                    & (prior_entropy >= np.median(prior_entropy))
                ).mean()
            ),
        },
    )


def _acquisition_value(
    top: np.ndarray,
    half_a: np.ndarray,
    revealed: np.ndarray,
    posterior: np.ndarray,
    truth: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Total error reduction from buying one label for each escalated item,
    drawn from half A's unrevealed remainder (half B is never touched)."""
    pool = half_a[top] - revealed[top]
    pool_total = pool.sum(axis=1)
    buyable = pool_total > 0
    if not buyable.any():
        return 0.0
    p = pool[buyable] / pool_total[buyable, None]
    label = (rng.random(len(p))[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
    idx = top[buyable]
    before = manhattan(
        posterior[idx] / posterior[idx].sum(axis=1, keepdims=True), truth[idx]
    )
    updated = posterior[idx].copy()
    updated[np.arange(len(idx)), label] += 1
    after = manhattan(updated / updated.sum(axis=1, keepdims=True), truth[idx])
    return float((before - after).sum())


def mock_judge(
    counts: np.ndarray, quality: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Mock q_hat for pipeline validation before the real cache exists:
    a quality-weighted blend of the true frequencies with Dirichlet noise,
    plus a spread feature anti-correlated with per-item quality."""
    n_items = counts.shape[0]
    freq = counts / counts.sum(axis=1, keepdims=True)
    per_item_quality = np.clip(quality + 0.2 * rng.standard_normal(n_items), 0.05, 0.95)
    noise = rng.dirichlet(np.ones(counts.shape[1]), size=n_items)
    q_hat = per_item_quality[:, None] * freq + (1 - per_item_quality[:, None]) * noise
    q_hat /= q_hat.sum(axis=1, keepdims=True)
    spread = (1 - per_item_quality) + 0.1 * rng.standard_normal(n_items)
    return q_hat, spread
