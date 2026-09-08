"""S7b machinery tests — everything except the API call, which is the same
caller used by the ChaosNLI cache."""

import numpy as np
import pytest

from dice.llm_judge import (
    TIERS,
    build_world,
    parse_distribution,
    run,
)

pytestmark = pytest.mark.unit


def test_world_is_deterministic_given_seed():
    a = build_world(100, seed=7)
    b = build_world(100, seed=7)
    assert np.allclose(a.p, b.p)
    assert a.descriptions == b.descriptions
    assert (a.counts == b.counts).all()


def test_tiers_cycle_and_descriptions_render():
    world = build_world(40, seed=0)
    assert set(world.tier) == set(TIERS)
    for description in world.descriptions:
        assert "{" not in description  # all format fields filled


def test_strong_tier_states_true_top_face():
    world = build_world(200, seed=1)
    for i, tier in enumerate(world.tier):
        if tier == "strong":
            top_face = ("red", "green", "blue")[int(np.argmax(world.p[i]))]
            assert top_face in world.descriptions[i]


def test_parse_distribution():
    assert np.allclose(parse_distribution("red=70 green=20 blue=10"), [0.7, 0.2, 0.1])
    assert parse_distribution("no numbers here") is None


def test_dry_run_counts_jobs(tmp_path):
    world = build_world(10, seed=0)
    # 10 dice x (3 variants + 1 confidence) = 40 jobs, one provider
    run(world, tmp_path / "cache.jsonl", limit=None, dry_run=True, diagnostic=False)
