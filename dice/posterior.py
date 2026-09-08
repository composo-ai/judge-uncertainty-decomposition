"""Dirichlet closed forms and Dirichlet–multinomial regression.

experiments_synthetic_dice.md §2–§3. Deliberately dataset-agnostic: the
ChaosNLI adapter should import this unchanged, swapping only the feature
matrix and the q_hat source (the one-code-path promise in the spec).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, xlogy

# eta = log(lam) is clamped in the regression: outside this range the DCM
# likelihood is flat and BFGS wanders
ETA_MIN, ETA_MAX = np.log(1e-3), np.log(1e6)


@dataclass
class DirichletStats:
    total: np.ndarray  # H(E[p]) — the incumbent's score
    aleatoric: np.ndarray  # E[H(p)]
    info: np.ndarray  # I = total - aleatoric (log-loss acquisition value)
    v: np.ndarray  # posterior spread, sum_j Var(p_j)
    delta: np.ndarray  # squared-error one-step acquisition value


def dirichlet_stats(alpha: np.ndarray) -> DirichletStats:
    alpha0 = alpha.sum(axis=1)
    m = alpha / alpha0[:, None]
    total = -xlogy(m, m).sum(axis=1)
    aleatoric = digamma(alpha0 + 1.0) - (m * digamma(alpha + 1.0)).sum(axis=1)
    g = 1.0 - (m**2).sum(axis=1)
    # delta: E[V_1] = alpha0*G/(alpha0+1)^2 exactly (marginal label prob is m,
    # conjugate update), so the one-step reduction is G/(alpha0+1)^2 — NOT
    # G/((alpha0+1)(alpha0+2)), the error T1 originally caught
    return DirichletStats(
        total=total,
        aleatoric=aleatoric,
        info=total - aleatoric,
        v=g / (alpha0 + 1.0),
        delta=g / (alpha0 + 1.0) ** 2,
    )


def dcm_nll(
    beta: np.ndarray, features: np.ndarray, q: np.ndarray, counts: np.ndarray
) -> float:
    """Negative log-likelihood of counts under Dir(lam(features) * q)-multinomial.

    Rows with zero counts contribute zero and may be pre-filtered by the caller.
    """
    eta = np.clip(beta[0] + features @ beta[1:], ETA_MIN, ETA_MAX)
    lam = np.exp(eta)
    a = lam[:, None] * q
    a0 = a.sum(axis=1)
    n = counts.sum(axis=1)
    ll = gammaln(a0) - gammaln(a0 + n) + (gammaln(a + counts) - gammaln(a)).sum(axis=1)
    return float(-ll.sum())


def fit_dcm(features: np.ndarray, q: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Fit log lam = beta0 + features @ beta[1:] by DCM maximum likelihood."""
    if (q <= 0).any():
        # alpha_j = lam*q_j = 0 makes gammaln NaN and BFGS silently returns
        # its start point (caught on ChaosNLI, 18 Aug). Floor q upstream.
        raise ValueError("q contains zero classes; floor and renormalise q first")
    labelled = counts.sum(axis=1) >= 1
    beta0 = np.zeros(1 + features.shape[1])
    beta0[0] = np.log(8.0)
    result = minimize(
        dcm_nll,
        beta0,
        args=(features[labelled], q[labelled], counts[labelled]),
        method="BFGS",
    )
    return result.x


def predict_lam(beta: np.ndarray, features: np.ndarray) -> np.ndarray:
    eta = np.clip(beta[0] + features @ beta[1:], ETA_MIN, ETA_MAX)
    return np.exp(eta)


def estimated_posterior(
    beta: np.ndarray, features: np.ndarray, q: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Dir(lam_hat * q + counts). Zero-label items get lam_hat from the
    regression alone — the cross-item pooling that mirrors E3."""
    lam_hat = predict_lam(beta, features)
    return lam_hat[:, None] * q + counts
