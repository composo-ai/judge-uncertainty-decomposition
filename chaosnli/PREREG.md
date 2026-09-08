# ChaosNLI Pre-Registration

Committed **before** the full judge run's results are read (run released 18 Aug, day 6,
immediately after this file's commit; go decision by Ryan in-session). Analysis follows this
document; deviations get documented as deviations.

**What has been seen already, honestly:** the 432-row smoke cache
(`caches/judge_cache.jsonl`) was inspected qualitatively during S7b teething checks — we know
the judge parses cleanly, refuses rarely, and readily predicts split distributions (e.g.
12/72/16). No arm evaluation, regression fit, or per-item error against labels has been run on
real q_hat. The judge-free label facts (noise ceiling 0.158, entropy sd 0.23) were computed on
day 3 and are in `dice/RESULTS.md`.

## Protocol (frozen)

- Judge: Azure deployment `gpt-5.6-terra` (2024-12-01-preview), 3 distribution variants +
  1 confidence elicitation per item, 3,113 items — the cache design frozen at smoke time.
- Pipeline: `pipeline.run_pipeline` at this commit — half-split (labels from half A only,
  half B evaluation-only), reveal n_i ∈ {0,1,2,5,10}, DCM features = ensemble spread,
  log1p(n), raw stated confidence, embedding nearest-distance and density
  (`embeddings_minilm.npz`). Arms as in `pipeline.ARMS` incl. `verbalised`.
- **Replication from the start (the S7b lesson):** every reported number is a mean over
  R = 200 reveal-resamples (seeds 0..199) with 95% percentile CIs. No single-draw numbers.
- Headline cell: **acquisition value, B = 10%**, ours_sqerr vs incumbent_prior. Selective
  (auto-resolved debiased squared error vs the 0.158 noise ceiling) is secondary. Other
  registered budgets {1, 5, 20}% reported with the same CIs, no cherry-picking.

## Registered predictions

- **P-C1 (regime placement — descriptive, no pass/fail).** Measure sd(log λ̂) across items and
  the occupancy of the confidently-predicted-split quadrant (items in the top half of fitted
  concentration AND top half of predicted entropy). These place real data on the S1/S6 axes and
  are reported regardless of outcome.
- **P-C2 (the decision claim — conditional, the honest form after S7b).** IF the judge
  produces heterogeneous concentration on real items (registered condition: sd(log λ̂) ≥ 0.5)
  THEN ours_sqerr beats incumbent_prior on B=10% acquisition value with the 95% CI excluding
  zero. If the condition fails, the registered prediction is **no separation**, and the paper
  reports the regime placement as the finding: the incumbent is safe exactly when concentration
  is homogeneous — the proposition's boundary, measured on real data.
- **P-C3 (verbalised confidence).** From S7b: stated confidence will be at best weakly
  informative — |Spearman(confidence, per-item error)| < Spearman(prior entropy, error) — and
  the verbalised arm will not beat incumbent_prior at any registered budget.
- **P-C4 (counts backbone).** Dropping all judge-side features (counts-only λ regression,
  intercept + log1p(n)) retains an advantage over incumbent_prior at B=10% acquisition value
  (CI excluding zero) — the S1 f_width=0 mechanism on real data. This is the
  deployment-critical claim: it needs no judge quality at all.

## Deviations

- **D1 (18 Aug, before any registered result was readable).** 14.4% of items had an exact-zero
  class in q̂ (all three variants answered 0/100), which makes the DCM likelihood NaN — BFGS was
  silently returning its initialisation, so every "result" was an init-point artifact and none
  was read as a finding. Fix: q̂ floored at 0.005 (a reported 0/100 means "<0.5 in 100") and
  renormalised at consume time; `fit_dcm` now raises on zero classes instead of failing
  silently. Dice results unaffected (no zero classes in the S7b cache; verified by re-running
  the dice unit suite and the committed S7b fit).

## Interpretation rules (frozen before results)

- Between-arm comparisons are paired within replicate; CIs on the *difference*.
- The B=1% cell (~15 items) is reported but pre-declared noise-prone; no claims rest on it.
- Manhattan errors are read against the 0.158 ceiling; squared-error claims use the debiased
  estimator only.
- A null P-C2 with the condition met is reported as a refutation of the transfer claim,
  prominently, per the refutations-as-findings policy.
