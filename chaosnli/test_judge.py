"""Network-free unit tests for the judge script: prompt render, answer
parsing, cache consumption. Live calls are exercised only where keys exist."""

import json

import numpy as np
import pytest

from chaosnli.adapter import ChaosNLI
from chaosnli.judge import (
    VARIANTS,
    build_prompt,
    parse_answer,
    q_hat_matrix,
)

pytestmark = pytest.mark.unit


def test_prompt_renders_all_variants():
    for key in VARIANTS:
        prompt = build_prompt(key, "A dog runs.", "An animal moves.")
        assert "A dog runs." in prompt
        assert "e=<int> n=<int> c=<int>" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("e=70 n=20 c=10", [0.7, 0.2, 0.1]),
        ("Sure! My answer:\ne = 100 n = 0 c = 0", [1.0, 0.0, 0.0]),
        ("e=33 n=33 c=34", [0.33, 0.33, 0.34]),
    ],
)
def test_parse_answer_valid(text, expected):
    assert parse_answer(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["I think entailment", "e=0 n=0 c=0", ""])
def test_parse_answer_invalid(text):
    assert parse_answer(text) is None


def test_q_hat_matrix_from_cache(tmp_path):
    data = ChaosNLI(
        uids=["u1", "u2"],
        premises=["p", "p"],
        hypotheses=["h", "h"],
        counts=np.array([[50, 30, 20], [10, 80, 10]]),
        source=["snli", "snli"],
    )
    cache = tmp_path / "cache.jsonl"
    rows = [
        {
            "uid": "u1",
            "model": "m",
            "variant": "neutral",
            "raw": "",
            "dist": [0.6, 0.3, 0.1],
        },
        {
            "uid": "u1",
            "model": "m",
            "variant": "strict",
            "raw": "",
            "dist": [0.8, 0.1, 0.1],
        },
        {"uid": "u1", "model": "m", "variant": "intuitive", "raw": "", "dist": None},
    ]
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    q_hat, spread = q_hat_matrix(data, cache)
    assert q_hat[0] == pytest.approx([0.7, 0.2, 0.1])
    assert spread[0] > 0
    # u2 has no members: uniform fallback, nan spread
    assert q_hat[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert np.isnan(spread[1])
