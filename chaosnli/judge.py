"""Judge ensemble for ChaosNLI — standalone and cache-first.

This environment carries no LLM API keys, so this script is built to run
anywhere: it reads the data zip, writes one JSONL cache line per
(uid, model, variant) as calls complete, and is safely re-runnable — cached
entries are never re-queried. The downstream pipeline consumes only the
cache file, so runs can happen on any machine with keys and the artifact
committed or copied back.

Design (paper_plan.md §5): q_hat_x = mean predicted label distribution over
the ensemble; ensemble spread becomes a feature for the DCM concentration
regression. Judges are asked for a distribution over annotators — "of 100
people, how many would say e/n/c" — NOT their own single answer, since the
target is the pool, not the judge's opinion. Three prompt variants supply
within-model spread; two models supply cross-model spread.

Usage:
    python -m chaosnli.judge \
        --data-dir <chaosNLI_v1.0> --cache judge_cache.jsonl [--limit 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .adapter import CLASSES, ChaosNLI, load

MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.2",
    # Azure deployment NAME (not model id); the account/model version behind
    # it must be recorded for the paper at run time
    "azure": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-terra"),
}

# Three readings of the task; disagreement between them is rubric-reading
# spread (modelling.md §3's prompted-basis idea in its cheapest form).
VARIANTS = {
    "neutral": (
        "100 crowd annotators were shown this premise and hypothesis and each "
        "chose one label: entailment (e), neutral (n), or contradiction (c)."
    ),
    "strict": (
        "100 careful annotators labelled this premise/hypothesis pair as "
        "entailment (e) only if the hypothesis must be true given the premise, "
        "contradiction (c) only if it cannot be true, and neutral (n) otherwise."
    ),
    "intuitive": (
        "100 ordinary people read this premise and hypothesis quickly and gave "
        "their gut answer: does it follow (e), contradict (c), or neither (n)?"
    ),
}

PROMPT = """{variant}

Premise: {premise}
Hypothesis: {hypothesis}

Predict how the 100 annotators' labels are distributed. Answer with only
three integers summing to 100, formatted exactly as:
e=<int> n=<int> c=<int>"""

# Verbalised-confidence baseline (u(x) feature 9 — the deployed cheap
# baseline reviewers expect; included to be tested, not endorsed). Must be
# in the SAME cache run: adding it after the fact doubles API spend.
CONFIDENCE_VARIANT = "confidence"
CONFIDENCE_PROMPT = """Premise: {premise}
Hypothesis: {hypothesis}

Choose the single best label for this pair — entailment (e), neutral (n),
or contradiction (c) — and state how confident you are that a panel of 100
human annotators would agree with your label on average.
Answer with only:
label=<e|n|c> confidence=<0-100>"""

ANSWER_RE = re.compile(r"e\s*=\s*(\d+)\s+n\s*=\s*(\d+)\s+c\s*=\s*(\d+)")
CONFIDENCE_RE = re.compile(r"label\s*=\s*([enc])\s+confidence\s*=\s*(\d+)")


def build_prompt(variant_key: str, premise: str, hypothesis: str) -> str:
    return PROMPT.format(
        variant=VARIANTS[variant_key], premise=premise, hypothesis=hypothesis
    )


def parse_answer(text: str) -> np.ndarray | None:
    """Extract the e/n/c counts; None if unparseable (caller logs and skips)."""
    match = ANSWER_RE.search(text)
    if not match:
        return None
    counts = np.array([int(g) for g in match.groups()], dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    return counts / total


def cache_key(uid: str, model: str, variant: str) -> str:
    return f"{uid}|{model}|{variant}"


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in open(path):
            row = json.loads(line)
            cache[cache_key(row["uid"], row["model"], row["variant"])] = row
    return cache


def _call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_openai(model: str, prompt: str) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model, max_tokens=64, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


_AZURE_CLIENT = None


def _azure_client():
    global _AZURE_CLIENT
    if _AZURE_CLIENT is None:
        import openai

        # one shared client: openai clients are thread-safe, and per-call
        # construction pays a TLS handshake every request
        _AZURE_CLIENT = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
        )
    return _AZURE_CLIENT


def _call_azure(model: str, prompt: str) -> str:
    import openai

    client = _azure_client()
    messages = [{"role": "user", "content": prompt}]
    try:
        # reasoning-family models: cap includes reasoning tokens, so leave
        # headroom and ask for minimal reasoning where supported
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2000,
            reasoning_effort="minimal",
            messages=messages,
        )
    except openai.BadRequestError:
        response = client.chat.completions.create(
            model=model, max_completion_tokens=2000, messages=messages
        )
    return response.choices[0].message.content


CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai, "azure": _call_azure}
KEY_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}


def available_providers() -> list[str]:
    out = [p for p, var in KEY_VARS.items() if os.environ.get(var)]
    # azure needs the endpoint as well as the key
    if "azure" in out and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        out.remove("azure")
    return out


def call_with_backoff(provider: str, model: str, prompt: str, attempts: int = 5) -> str:
    import openai

    for attempt in range(attempts):
        try:
            return CALLERS[provider](model, prompt)
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APIStatusError,
        ):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt + random.random())
    raise RuntimeError("unreachable")


def run(
    data: ChaosNLI,
    cache_path: Path,
    limit: int | None,
    dry_run: bool,
    workers: int = 8,
) -> None:
    cache = load_cache(cache_path)
    providers = available_providers()
    if not providers and not dry_run:
        sys.exit(
            "No judge API keys in environment "
            f"({', '.join(KEY_VARS.values())}); run with --dry-run to preview."
        )
    n_items = len(data.uids) if limit is None else min(limit, len(data.uids))
    jobs = []
    done = 0
    for i in range(n_items):
        for provider in providers or list(MODELS):
            model = MODELS[provider]
            for variant in [*VARIANTS, CONFIDENCE_VARIANT]:
                if cache_key(data.uids[i], model, variant) in cache:
                    done += 1
                else:
                    jobs.append((i, provider, model, variant))
    print(
        f"cached={done} todo={len(jobs)} providers={providers or 'none (dry run)'}",
        flush=True,
    )
    if dry_run or not jobs:
        return

    def do_job(job):
        i, provider, model, variant = job
        if variant == CONFIDENCE_VARIANT:
            prompt = CONFIDENCE_PROMPT.format(
                premise=data.premises[i], hypothesis=data.hypotheses[i]
            )
            text = call_with_backoff(provider, model, prompt)
            match = CONFIDENCE_RE.search(text or "")
            return {
                "uid": data.uids[i],
                "model": model,
                "variant": variant,
                "raw": text,
                "dist": None,
                "label": match.group(1) if match else None,
                "confidence": int(match.group(2)) if match else None,
            }
        prompt = build_prompt(variant, data.premises[i], data.hypotheses[i])
        text = call_with_backoff(provider, model, prompt)
        dist = parse_answer(text or "")
        return {
            "uid": data.uids[i],
            "model": model,
            "variant": variant,
            "raw": text,
            "dist": None if dist is None else dist.tolist(),
        }

    written = skipped_empty = 0
    lock = threading.Lock()
    with open(cache_path, "a") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(do_job, jobs):
                # empty reply = transient upstream blip; leave uncached so the
                # next run retries it
                if not (row["raw"] or "").strip():
                    skipped_empty += 1
                    continue
                with lock:
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    written += 1
                    if written % 200 == 0:
                        print(f"progress: {written}/{len(jobs)}", flush=True)
    print(f"written={written} skipped_empty={skipped_empty}", flush=True)


def q_hat_matrix(data: ChaosNLI, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Consume the cache: per-item ensemble mean q_hat (N,3) and ensemble
    spread (N,) — sd of member distributions, the first DCM feature.
    Items with no parsed members get uniform q_hat and nan spread."""
    cache = load_cache(cache_path)
    by_uid: dict[str, list[np.ndarray]] = {}
    for row in cache.values():
        if row["dist"] is not None:
            by_uid.setdefault(row["uid"], []).append(np.array(row["dist"]))
    q_hat = np.full((len(data.uids), len(CLASSES)), 1.0 / len(CLASSES))
    spread = np.full(len(data.uids), np.nan)
    for i, uid in enumerate(data.uids):
        members = by_uid.get(uid)
        if members:
            stack = np.stack(members)
            q_hat[i] = stack.mean(axis=0)
            spread[i] = float(stack.std(axis=0).mean())
    # A reported 0/100 means "<0.5 in 100", not literally zero — and an exact
    # zero class makes the DCM likelihood undefined (PREREG deviation D1;
    # 14.4% of items had one). Floor at half a count and renormalise.
    q_hat = np.maximum(q_hat, 0.005)
    q_hat /= q_hat.sum(axis=1, keepdims=True)
    return q_hat, spread


def confidence_matrix(data: ChaosNLI, cache_path: Path) -> np.ndarray:
    """Verbalised-confidence baseline score per item: mean stated confidence
    across models, sign-flipped so higher = escalate first. nan when absent."""
    cache = load_cache(cache_path)
    by_uid: dict[str, list[float]] = {}
    for row in cache.values():
        if row.get("confidence") is not None:
            by_uid.setdefault(row["uid"], []).append(float(row["confidence"]))
    out = np.full(len(data.uids), np.nan)
    for i, uid in enumerate(data.uids):
        if uid in by_uid:
            out[i] = -float(np.mean(by_uid[uid]))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    run(load(args.data_dir), args.cache, args.limit, args.dry_run, args.workers)


if __name__ == "__main__":
    main()
