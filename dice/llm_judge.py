"""S7b — real LLM judge on dice with controlled truth
(experiments_synthetic_dice.md §8). Cache-first and standalone, same
pattern as chaosnli/judge.py: runs anywhere keys exist, resumable, the
pipeline consumes only the cache.

Each die has a known bias p_i and a textual description whose
informativeness tier we control — description informativeness IS
familiarity, made real. The LLM reads the description ONLY (rolls are
revealed downstream, where the DCM does the updating, as in production).
An optional diagnostic arm also shows the rolls, to compare the LLM's
self-updating against conjugate updating on its own stated prior.

Usage:
    python -m dice.llm_judge \
        --cache s7b_cache.jsonl [--n-dice 500] [--limit 20] [--dry-run] [--diagnostic]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sps

from chaosnli.judge import (
    KEY_VARS,
    MODELS,
    available_providers,
    call_with_backoff,
)
from .generator import K, PEAK, UNIFORM
from .s7a import KAPPA_SHAPE

FACES = ("red", "green", "blue")  # colour faces avoid digit-face confusion in prompts

# Informativeness tiers — the real-world analogue of familiarity. Cycled
# deterministically over dice so every tier sees every contentiousness level.
TIERS = ("none", "weak", "directional", "strong")

DESCRIPTIONS = {
    "none": "A three-sided die with faces red, green and blue.",
    "weak": (
        "A three-sided die (faces red, green, blue) from a workshop that "
        "produces a mix of fair and weighted dice."
    ),
    "directional": (
        "A three-sided die (faces red, green, blue) that testing suggests "
        "somewhat favours {top} over the other faces."
    ),
    "strong": (
        "A three-sided die (faces red, green, blue) weighted toward {top}: "
        "in a long calibration run it landed {top} roughly {pct} percent of "
        "the time."
    ),
}

VARIANTS = {
    "neutral": "You are estimating the behaviour of a physical die.",
    "sceptical": (
        "You are estimating the behaviour of a physical die. Descriptions "
        "can be unreliable; weigh them accordingly."
    ),
    "literal": (
        "You are estimating the behaviour of a physical die. Take the "
        "description at face value."
    ),
}

PROMPT = """{variant}

{description}

If this die were rolled 100 times, predict how the outcomes would be
distributed. Answer with only three integers summing to 100, formatted
exactly as:
red=<int> green=<int> blue=<int>"""

CONFIDENCE_PROMPT = """{description}

If this die were rolled 100 times, how confident are you that you can
predict the outcome distribution to within 10 rolls per face? Answer with
only:
confidence=<0-100>"""

DIAGNOSTIC_PROMPT = """{variant}

{description}

The die has now been rolled {n} times with these results: {rolls}.

Combining the description and the observed rolls, predict how 100 further
rolls would be distributed. Answer with only three integers summing to 100,
formatted exactly as:
red=<int> green=<int> blue=<int>"""

ANSWER_RE = re.compile(r"red\s*=\s*(\d+)\s+green\s*=\s*(\d+)\s+blue\s*=\s*(\d+)")
CONFIDENCE_RE = re.compile(r"confidence\s*=\s*(\d+)")


@dataclass
class DiceWorld:
    p: np.ndarray  # (N, K) truth
    tier: list[str]
    descriptions: list[str]
    counts: np.ndarray  # revealed rolls (for the DCM and the diagnostic arm)
    n: np.ndarray


def build_world(n_dice: int = 500, seed: int = 0) -> DiceWorld:
    """Deterministic given seed — the world is part of the experiment spec,
    so the same seed must regenerate it exactly on any machine."""
    rng = np.random.default_rng(seed)
    t = sps.beta.ppf(rng.random(n_dice), KAPPA_SHAPE, KAPPA_SHAPE)
    perm = np.argsort(rng.random((n_dice, K)), axis=1)
    p = (1.0 - t[:, None]) * PEAK[perm] + t[:, None] * UNIFORM

    tier = [TIERS[i % len(TIERS)] for i in range(n_dice)]
    descriptions = []
    for i in range(n_dice):
        top = FACES[int(np.argmax(p[i]))]
        pct = int(round(100 * p[i].max()))
        descriptions.append(DESCRIPTIONS[tier[i]].format(top=top, pct=pct))

    n = rng.choice(np.array([0, 1, 2, 5, 10]), size=n_dice)
    counts = np.zeros((n_dice, K), dtype=np.int64)
    for n_val in np.unique(n):
        if n_val == 0:
            continue
        mask = n == n_val
        counts[mask] = rng.multinomial(int(n_val), p[mask])
    return DiceWorld(p=p, tier=tier, descriptions=descriptions, counts=counts, n=n)


def parse_distribution(text: str) -> np.ndarray | None:
    match = ANSWER_RE.search(text)
    if not match:
        return None
    counts = np.array([int(g) for g in match.groups()], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else None


def _rolls_text(counts: np.ndarray) -> str:
    return ", ".join(f"{FACES[j]}: {int(counts[j])}" for j in range(K))


def cache_key(idx: int, model: str, kind: str, variant: str) -> str:
    return f"die{idx}|{model}|{kind}|{variant}"


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in open(path):
            row = json.loads(line)
            cache[cache_key(row["idx"], row["model"], row["kind"], row["variant"])] = (
                row
            )
    return cache


def run(
    world: DiceWorld,
    cache_path: Path,
    limit: int | None,
    dry_run: bool,
    diagnostic: bool,
    workers: int = 8,
) -> None:
    cache = load_cache(cache_path)
    providers = available_providers()
    if not providers and not dry_run:
        sys.exit(
            "No judge API keys in environment "
            f"({', '.join(KEY_VARS.values())}); run with --dry-run to preview."
        )
    # one model per discussion 17 Aug: first available provider only
    providers = providers[:1] if providers else list(MODELS)[:1]
    n_dice = len(world.tier) if limit is None else min(limit, len(world.tier))
    todo = done = 0
    jobs: list[tuple[int, str, str, str]] = []
    for i in range(n_dice):
        for variant in VARIANTS:
            jobs.append(
                (
                    i,
                    "dist",
                    variant,
                    PROMPT.format(
                        variant=VARIANTS[variant], description=world.descriptions[i]
                    ),
                )
            )
        jobs.append(
            (
                i,
                "confidence",
                "confidence",
                CONFIDENCE_PROMPT.format(description=world.descriptions[i]),
            )
        )
        if diagnostic and world.n[i] > 0:
            jobs.append(
                (
                    i,
                    "diagnostic",
                    "neutral",
                    DIAGNOSTIC_PROMPT.format(
                        variant=VARIANTS["neutral"],
                        description=world.descriptions[i],
                        n=int(world.n[i]),
                        rolls=_rolls_text(world.counts[i]),
                    ),
                )
            )
    provider = providers[0]
    model = MODELS[provider]
    pending = []
    for idx, kind, variant, prompt in jobs:
        if cache_key(idx, model, kind, variant) in cache:
            done += 1
        else:
            pending.append((idx, kind, variant, prompt))
    todo = len(pending)
    print(f"cached={done} todo={todo} providers={providers}", flush=True)
    if dry_run or not pending:
        return

    def do_job(job):
        idx, kind, variant, prompt = job
        text = call_with_backoff(provider, model, prompt)
        row: dict = {
            "idx": idx,
            "model": model,
            "kind": kind,
            "variant": variant,
            "raw": text,
        }
        if kind == "confidence":
            match = CONFIDENCE_RE.search(text or "")
            row["confidence"] = int(match.group(1)) if match else None
        else:
            dist = parse_distribution(text or "")
            row["dist"] = None if dist is None else dist.tolist()
        return row

    written = skipped_empty = 0
    lock = threading.Lock()
    with open(cache_path, "a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(do_job, pending):
                # empty reply = transient blip; leave uncached so reruns retry
                if not (row["raw"] or "").strip():
                    skipped_empty += 1
                    continue
                with lock:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    written += 1
                    if written % 200 == 0:
                        print(f"progress: {written}/{todo}", flush=True)
    print(f"written={written} skipped_empty={skipped_empty}", flush=True)


def q_hat_matrix(world: DiceWorld, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-die ensemble mean q_hat and variant spread, distribution kind only."""
    cache = load_cache(cache_path)
    by_idx: dict[int, list[np.ndarray]] = {}
    for row in cache.values():
        if row["kind"] == "dist" and row.get("dist") is not None:
            by_idx.setdefault(row["idx"], []).append(np.array(row["dist"]))
    n_dice = len(world.tier)
    q_hat = np.full((n_dice, K), 1.0 / K)
    spread = np.full(n_dice, np.nan)
    for i in range(n_dice):
        members = by_idx.get(i)
        if members:
            stack = np.stack(members)
            q_hat[i] = stack.mean(axis=0)
            spread[i] = float(stack.std(axis=0).mean())
    return q_hat, spread


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--n-dice", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    world = build_world(args.n_dice, args.seed)
    run(world, args.cache, args.limit, args.dry_run, args.diagnostic, args.workers)


if __name__ == "__main__":
    main()
