"""Unit tests for the ChaosNLI data layer. The load test needs the data zip
(kept out of git); it skips cleanly when absent so CI stays green."""

import os
from pathlib import Path

import numpy as np
import pytest

from chaosnli.adapter import (
    LABELS_PER_ITEM,
    half_split,
    load,
    manhattan,
    noise_ceiling,
)

pytestmark = pytest.mark.unit

DATA_DIR = Path(os.environ.get("CHAOSNLI_DIR", "data/chaosNLI_v1.0"))


def test_half_split_partitions_exactly():
    rng = np.random.default_rng(0)
    counts = np.array([[100, 0, 0], [50, 30, 20], [0, 3, 97], [34, 33, 33]])
    a, b = half_split(counts, rng)
    assert (a + b == counts).all()
    assert (a.sum(axis=1) == LABELS_PER_ITEM // 2).all()
    assert (a >= 0).all() and (b >= 0).all()


def test_half_split_is_hypergeometric_not_binomial():
    # a degenerate item must split 50/50 exactly, never stochastically
    rng = np.random.default_rng(1)
    counts = np.array([[100, 0, 0]])
    for _ in range(10):
        a, _ = half_split(counts, rng)
        assert a[0, 0] == 50


def test_manhattan_bounds():
    p = np.array([[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    q = np.array([[0.0, 1.0, 0.0], [1 / 3, 1 / 3, 1 / 3]])
    d = manhattan(p, q)
    assert d[0] == pytest.approx(2.0)
    assert d[1] == pytest.approx(0.0)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="chaosNLI data not downloaded")
def test_load_matches_plan_assumptions():
    data = load(DATA_DIR)
    assert data.counts.shape == (3113, 3)
    assert (data.counts.sum(axis=1) == 100).all()
    assert data.source.count("snli") == 1514
    assert data.source.count("mnli_m") == 1599


@pytest.mark.skipif(not DATA_DIR.exists(), reason="chaosNLI data not downloaded")
def test_noise_ceiling_sane():
    data = load(DATA_DIR)
    ceiling = noise_ceiling(data.counts, np.random.default_rng(0), n_splits=10)
    # k=50 multinomial noise implies a strictly positive floor well below max
    assert 0.02 < ceiling["mean"] < 0.5


def test_debiased_squared_error_is_unbiased():
    from chaosnli.adapter import (
        debiased_squared_error,
    )

    rng = np.random.default_rng(0)
    p_true = np.array([0.6, 0.3, 0.1])
    pred = np.array([[0.5, 0.3, 0.2]])
    true_sq = float(((pred[0] - p_true) ** 2).sum())
    k = 50
    estimates = [
        float(debiased_squared_error(pred, rng.multinomial(k, p_true)[None, :])[0])
        for _ in range(20_000)
    ]
    # unbiased: MC mean within 3 MC-standard-errors of the true squared error
    mc_mean = float(np.mean(estimates))
    mc_se = float(np.std(estimates) / np.sqrt(len(estimates)))
    assert abs(mc_mean - true_sq) < 3 * mc_se + 1e-6
