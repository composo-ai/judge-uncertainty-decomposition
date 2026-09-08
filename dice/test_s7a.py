"""S7a unit tests: the truth-first world behaves as specified, and the
pre-registered predictions are checkable (not yet checked — that happens
on the registered run, this pins the machinery)."""

import numpy as np
import pytest
from scipy import stats as sps

from dice.s7a import (
    generate,
    prophet_values,
    run_cell,
)

pytestmark = pytest.mark.unit


def test_truth_first_judge_fidelity_tracks_familiarity():
    draw = generate(5000, miscal=0.0, seed=0)
    err = np.abs(draw.q - draw.p).sum(axis=1)
    # judge prediction error falls with familiarity
    assert sps.spearmanr(draw.f, err).statistic < -0.5


def test_stated_confidence_honest_at_m0_flat_at_m1():
    honest = generate(5000, miscal=0.0, seed=1)
    flat = generate(5000, miscal=1.0, seed=1)
    # honest: stated spread anti-correlates with familiarity (confident where familiar)
    assert sps.spearmanr(honest.f, honest.stated_spread).statistic < -0.9
    # flat: stated spread carries ~no familiarity information
    assert abs(sps.spearmanr(flat.f, flat.stated_spread).statistic) < 0.1


def test_stated_confidence_inverted_beyond_m1():
    inv = generate(5000, miscal=1.5, seed=2)
    # inverted: MORE confident (lower spread) where LESS familiar
    assert sps.spearmanr(inv.f, inv.stated_spread).statistic > 0.5


def test_prophet_values_nonnegative_in_expectation_and_finite():
    draw = generate(500, miscal=0.5, seed=3)
    rng = np.random.default_rng(3)
    posterior = np.maximum(8.0 * draw.q, 0.05) + draw.counts
    values = prophet_values(posterior, draw.p, rng)
    assert np.isfinite(values).all()
    # one more true-label draw improves error on average across items
    assert values.mean() > 0


def test_run_cell_schema_and_oracle_zero_regret():
    rows = run_cell(500, miscal=0.5, seed=4)
    oracle_rows = [r for r in rows if r["arm"] == "oracle"]
    assert all(r["regret"] == pytest.approx(0.0, abs=1e-9) for r in oracle_rows)
    assert all(np.isfinite(r["auto_resolved_error"]) for r in rows)
