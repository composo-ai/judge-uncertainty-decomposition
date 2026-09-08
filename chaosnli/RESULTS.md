# ChaosNLI Registered Results — PREREG.md scored

Run 18 Aug (day 6 late): full judge cache (12,452 rows, `gpt-5.6-terra`), 200
reveal-resamples, paired CIs, per `analysis.py`. Raw numbers in
`registered_report.json`. Deviation D1 (q̂ zero-class floor) applied before any
result was read.

## Verdicts, scored as registered

- **P-C1 (regime placement).** sd(log λ̂) = **0.216** (CI 0.17–0.30) — the P-C2 condition
  (≥ 0.5) is **NOT met**. Confidently-predicted-split quadrant occupancy 0.23. Real data with
  this judge sits in the **low trust-heterogeneity regime**: the fitted GLM assigns nearly the
  same credibility everywhere (mean β: intercept 1.62 → λ ≈ 5; all feature coefficients
  |β| ≤ 0.2).
- **P-C2 (decision claim — conditional).** Condition unmet → the registered prediction was
  **no separation**, and that is what we observe at the headline cell: ours − entropy-ranking
  acquisition value at B=10% = +4.5 (CI **−2.2 to +9.4**). B=1/5% likewise cross zero. B=20%
  is marginally positive (+10.2, CI 0.5–18.9) — one registered budget, reported, not
  headlined. **The conditional prediction is CONFIRMED in its condition-unmet branch**: the
  λ̂-spread diagnostic, computable before any escalation decision, correctly forecast that
  entropy-ranking would not be badly wrong on this dataset.
- **P-C3 (verbalised confidence).** Arm clause **confirmed**: escalating on low stated
  confidence is indistinguishable from entropy-ranking and from random at every budget (all
  paired CIs cross zero). Correlation clause **refuted in letter**: |ρ(conf, raw err)| = 0.131
  is not smaller than ρ(entropy, raw err) = 0.110 — but both are near-noise, and the blanket
  confidence pathology reproduces (mean stated confidence **90.3** at mean raw error 0.458 vs
  the 0.158 ceiling). Sign note: on real items confidence points the *right* way (−0.13),
  unlike dice (+0.22) — weakly informative, still useless as a ranking.
- **P-C4 (counts-only backbone).** +6.9pp-equivalent over entropy-ranking at B=10% but CI
  (**−0.4** to +14.7) — **REFUTED as registered** (CI must exclude zero). Just barely, and the
  direction is right; at this N and reveal variance the claim is not established on real data.

## The unregistered but visible structure (exploratory, label as such)

Arm means at B=10% acquisition value split into two clusters: count-aware arms
(ours_info 15.0, practitioner 14.3, ours_sqerr 12.7, v_only 12.1) vs count-blind arms
(random 8.3, entropy 8.2, verbalised 8.1, posterior-entropy 6.9). The pattern matches theory
for a low-λ̂-heterogeneity regime — when judge-side trust is homogeneous, epistemic ranking
reduces to label-count ranking, so fewest-labels-first is nearly optimal and everything
count-blind clusters together. Between-replicate variance keeps most pairwise CIs from
excluding zero; treat the clustering as descriptive.

## What this means for the paper

The honest three-part story survives intact and gains its real-data panel:

1. Mechanism and cost: calibrated simulation (large, pre-registered, generator-caveated).
2. The split is estimable on a real LLM: dice isolation exhibit.
3. On real data, the **diagnostic works**: fit the small GLM, read sd(log λ̂); here it is 0.22,
   the proposition says entropy-ranking is then approximately safe, and it is. The regime the
   incumbent should fear (heterogeneous trust) is exactly the regime any partially-deployed
   pipeline with variable familiarity enters — the simulation measures the cost there.

No claim of a real-data decision-level win is available, and none goes in the draft. The
refutations (P-C4; P-C3's correlation clause) are reported as findings.

## CI-semantics note (21 Aug, from the second-reviewer pass)

The registered percentile intervals are per-draw sensitivity intervals over the evaluation
protocol's randomness (half-split, reveal, purchase draws) on the fixed dataset and judge.
The registered verdicts stand exactly as scored against that standard. As clearly-labelled
exploratory quantities, the *expected* effects are positive and precisely estimated: P-C2
headline mean +4.5 at MC SE ≈ 0.21 (~20 SEs above zero; arm means 12.7 vs 8.2 ≈ +55% relative
in expected acquisition value); P-C4 mean +6.9 at MC SE ≈ 0.27. Neither interval addresses
generalisation to new items (an item-level bootstrap would; not run). Draft v1.2 §3.4 carries
this distinction; the dice-side gate CIs are unaffected (each seed there generates a fresh
world, so seed-percentiles are a legitimate generalisation interval over the generator).

## Correction and exploratory extension (25 Aug, day 13)

Two errors in the reporting above were found while drafting §5.3, both in what was
*computed*, not in what was run. No new judge calls; the recomputation reproduces the
registered headline to four decimals (`exploratory_paired_arms.py`, output committed).

1. **`ours_info` was omitted from the paper's Table 3.** It is the top arm at B=10% (15.0,
   above practitioner's 14.3) and is the mutual-information score the paper's model section
   derives as *the* epistemic estimate. Its level was always in this file; the draft table
   dropped it.
2. **Paired intervals existed for one comparison only.** `analysis.py` scored the frozen
   headline (`ours_sqerr` vs `incumbent_prior`) and nothing else. Recomputing paired
   differences for every arm at every budget gives, at B=10% vs `incumbent_prior`:

   | arm | advantage | per-draw 95% |
   | --- | --- | --- |
   | ours_info | **+6.82** | **+1.66, +11.92** |
   | practitioner | +6.13 | −0.01, +12.38 |
   | ours_sqerr *(registered)* | +4.46 | −2.19, +9.44 |
   | v_only | +3.94 | −1.71, +10.66 |
   | random | +0.13 | −5.48, +5.63 |
   | verbalised | −0.12 | −5.85, +5.14 |
   | incumbent | −1.25 | −5.95, +3.33 |

   `ours_info` also separates at B=20% (+12.81, [+4.29, +20.35]); B=1% and B=5% do not.
   `ours_info` minus `practitioner` is +0.69 [−6.29, +7.09] — indistinguishable.

**Status.** The registered verdicts above stand exactly as scored: P-C2's headline arm does
not separate, P-C4 is refuted. The `ours_info` separation is **exploratory** — the arm ran
under the frozen protocol but was not the named headline, and it is one of seven comparisons
against the baseline.

**Why the registered condition missed it.** P-C2 conditioned on sd(log λ̂) ≥ 0.5, i.e. on
judge-side trust heterogeneity. The §3 characterisation is about total evidence
α₀ = λ + n. Reveal counts n ∈ {0,1,2,5,10} make α₀ heterogeneous here regardless of λ, which
is why every count-aware arm sits above every count-blind one. The condition tested the wrong
term; that is recorded as a mis-specification, not reinterpreted into a confirmation.
