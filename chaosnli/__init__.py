"""ChaosNLI adapter (SNLI + MNLI-matched only — aNLI is binary and out of
scope, see paper_plan.md §8). Data layer is judge-free; the judge ensemble
plugs in via dice/posterior.py, which is dataset-agnostic by design."""
