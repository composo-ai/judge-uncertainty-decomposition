"""Pipeline validation with the mock judge — proves the §4 machinery runs
end-to-end before the real cache exists, and pins the properties that must
hold regardless of judge quality."""

import os
from pathlib import Path

import numpy as np
import pytest

from chaosnli.adapter import load
from chaosnli.pipeline import (
    ARMS,
    BUDGETS_PCT,
    mock_judge,
    reveal,
    run_pipeline,
)

pytestmark = pytest.mark.unit

DATA_DIR = Path(os.environ.get("CHAOSNLI_DIR", "data/chaosNLI_v1.0"))


def _toy_counts(n_items=300, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(3) * 2, size=n_items)
    return np.array([rng.multinomial(100, p[i]) for i in range(n_items)])


def test_reveal_never_exceeds_pool():
    rng = np.random.default_rng(0)
    half_a = np.array([[50, 0, 0], [20, 20, 10], [0, 1, 49], [17, 17, 16]])
    revealed, n = reveal(half_a, rng)
    assert (revealed <= half_a).all()
    assert (revealed.sum(axis=1) == np.minimum(n, half_a.sum(axis=1))).all()


def test_pipeline_end_to_end_on_toy():
    counts = _toy_counts()
    rng = np.random.default_rng(1)
    q_hat, spread = mock_judge(counts, quality=0.7, rng=rng)
    # no confidence vector -> the verbalised arm is skipped
    result = run_pipeline(counts, q_hat, spread, embeddings=None, seed=1)
    assert len(result.selective) == (len(ARMS) - 1) * len(BUDGETS_PCT)
    for row in result.selective:
        assert np.isfinite(row["auto_resolved_error"])
    assert np.isfinite(result.scatter["info"]).all()
    assert result.diagnostics["n_labelled"] > 0


def test_pipeline_verbalised_arm_with_confidence():
    counts = _toy_counts()
    rng = np.random.default_rng(1)
    q_hat, spread = mock_judge(counts, quality=0.7, rng=rng)
    conf = -rng.uniform(0, 100, counts.shape[0])
    conf[:10] = np.nan  # missing replies stay handleable
    result = run_pipeline(
        counts, q_hat, spread, embeddings=None, seed=1, confidence_score=conf
    )
    arms = {row["arm"] for row in result.selective}
    assert arms == set(ARMS)
    assert len(result.selective) == len(ARMS) * len(BUDGETS_PCT)


def test_better_mock_judge_lower_error():
    counts = _toy_counts(seed=2)
    errors = []
    for quality in (0.2, 0.9):
        rng = np.random.default_rng(3)
        q_hat, spread = mock_judge(counts, quality=quality, rng=rng)
        result = run_pipeline(counts, q_hat, spread, embeddings=None, seed=3)
        errors.append(result.diagnostics["mean_error_all"])
    assert errors[1] < errors[0]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="chaosNLI data not downloaded")
def test_pipeline_runs_on_real_counts_with_mock_judge():
    data = load(DATA_DIR)
    rng = np.random.default_rng(4)
    q_hat, spread = mock_judge(data.counts, quality=0.6, rng=rng)
    result = run_pipeline(data.counts, q_hat, spread, embeddings=None, seed=4)
    rows = {
        (r["arm"], r["budget_pct"]): r["auto_resolved_error"] for r in result.selective
    }
    assert len(rows) == (len(ARMS) - 1) * len(BUDGETS_PCT)
    assert all(np.isfinite(v) for v in rows.values())
