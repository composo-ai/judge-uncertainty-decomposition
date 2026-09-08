# Synthetic Dice — Registered Sweep Results

Run 2026-08-17, code at the commit adding this file. All sweeps at registered seed counts
(`DEFAULT_SEEDS`), $N = 2000$ items, evaluated by `gate.py` (committed before results were
read). Raw CSVs regenerate deterministically via `python -m
dice.sweep --sweep <s> --out <dir>`; figures in `figures/`.

## Gate verdict: **PASS**

| Criterion | Registered threshold | Measured | |
| --- | --- | --- | --- |
| C1 — advantage at the criterion cell (estimated condition, $r=1$, B=10%) | ≥ 15pp, CI excl. 0 | **80.9pp**, CI (80.6, 81.2), 500 seeds | ✅ |
| C2 — oracle↔estimated advantage-curve tracking across `f_width` | Spearman ≥ 0.9 | **1.0** | ✅ |

(The ours-arm tracking Spearman is undefined — its oracle-condition regret is constant ≈ 0
across the sweep, which is itself the strongest possible form of tracking. The registered
"advantage curve" interpretation binds, per `gate.py`'s header.)

**Consequence per the plan: the derivation-led (Lee-style) structure is confirmed as the
paper's shape.** The day-5 fork resolved on day 1.

## S1 — the main curve (fig_s1_epistemic_axis.png)

Advantage of $\Delta$-ranking over label-free entropy ranking (`incumbent_prior`), B=10%,
sq-err objective:

| `f_width` | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| advantage (pp) | 34.8 | 55.1 | 68.6 | 75.1 | 78.8 | 81.1 |
| practitioner regret (pp) | 19.2 | 41.5 | 53.5 | 57.4 | 57.2 | 56.0 |
| random regret (pp) | 50.7 | 65.9 | 76.3 | 81.2 | 83.9 | 85.4 |

Three observations:

1. **The advantage at `f_width = 0` is 34.8pp, not 0 — and that is a finding, not an anomaly.**
   S1 draws $n_i \in \{0,1,2,5,10\}$, so even with a homogeneous *prior*, revealed labels make
   posterior concentration heterogeneous, and entropy cannot see it. Any deployed pipeline with
   partially-labelled data is in this regime by construction. The true null (concentration fully
   pinned: `f_width=0`, `n_fixed=5`) measures 0.0pp — verified by T2.
2. **Entropy-ranking approaches random as epistemic heterogeneity grows** (81 vs 85pp at
   `f_width=1`), consistent with the day-0 diagnostic where it was *worse* than random at
   constant shape.
3. **`v_only` is within 1.2–1.8pp of full $\Delta$-ranking everywhere.** The rank-collapse
   predicted analytically is confirmed at the ranking level: "escalate epistemic, not total"
   captures ~98% of the gain; the $\Delta$-vs-$v$ refinement is worth ~1.5pp here. The paper's
   claim is the two-way split; the three-way structure is a modelling implication, exactly as
   `paper_plan.md` §1's caveat resolves it. Practitioner routing (fewest-labels-first) helps but
   leaves ~56pp on the table — label count alone is not an epistemic signal.

## S3 — estimation quality (fig_rho_star.png): **there is no ρ\* threshold, and that is the
strongest version of the result**

At `f_width = 1`, advantage over entropy-ranking as the familiarity proxy's informativeness
varies:

| realised Spearman($\hat f, f$) | 0.0 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
| --- | --- | --- | --- | --- | --- | --- |
| advantage (pp) | 31.8 | 35.8 | 45.0 | 56.9 | 71.6 | 80.7 |

The registered question was "at what proxy quality does our ranking overtake the incumbent?"
Answer: **at zero.** Even with a worthless familiarity signal, the DCM posterior still contains
the revealed counts, and counts-only epistemic ranking beats entropy by ~32pp. A good
familiarity signal adds a further ~49pp on top. The deployment sentence: *you do not need a
good epistemic estimator to beat disagreement-based escalation — the labels you already have
suffice; estimator quality then buys more, monotonically.*

## S2 — the phase diagram (fig1_phase.png)

Advantage at `f_width = 1` by aleatoric–epistemic correlation $\rho$:

| $\rho$ | −0.8 | −0.4 | 0.0 | 0.4 | 0.8 |
| --- | --- | --- | --- | --- | --- |
| advantage (pp) | 99.2 | 92.4 | 80.9 | 66.8 | 50.0 |

The pre-run concern (review thread, `paper_plan.md` §3) was that under strong *positive*
correlation — contentious items also being the unfamiliar ones, the realistic direction —
entropy becomes a decent epistemic proxy and the incumbent is "approximately right". Measured:
at $\rho = 0.8$ the advantage is still **50pp**. Correlation helps the incumbent, but nowhere
near enough. (ChaosNLI's empirical $\rho$ gets measured on day 8 and locates the real data on
this axis.)

## Supporting real-data numbers (judge-free, from `chaosnli/`)

- Noise ceiling (Manhattan between 50-label halves): mean **0.158**, median 0.163, p90 0.209.
- Empirical label entropy: mean 0.65, sd **0.23** across 3,113 items — real data has substantial
  aleatoric heterogeneity.

## S6 — dose–response over concentration-range width (added day 5, post-review-anticipation)

Advantage over `incumbent_prior` at B=10%, sq-err, `f_width=1`, as the $\lambda$ range narrows
around a fixed geometric mean of 8 (registered regime = ×256):

| range width | ×4 $[4, 16]$ | ×16 | ×64 | ×256 $[0.5, 128]$ | ×1024 |
| --- | --- | --- | --- | --- | --- |
| advantage, oracle (pp) | 59.6 | 72.3 | 77.6 | 80.7 | 82.8 |
| advantage, estimated $r{=}1$ (pp) | 59.4 | 72.2 | 77.5 | 80.7 | 82.8 |

**The wide range is not doing the work.** At a modest ×4 spread the advantage is still ~60pp,
and the estimated condition matches oracle at every width. This is the direct answer to "you
built a world where the incumbent must fail" (`paper/anticipated_reviews.md` R1.4): the failure
needs only *some* concentration heterogeneity, and its size is a slowly-varying function of how
much. Report this curve in the paper body instead of seed-level CIs.

## Caveats — read before quoting any number above

- **Magnitudes are generator-dependent.** $\lambda$ spans 0.5–128 (a very wide epistemic range)
  and the advantage is measured at B=10% under the sq-err objective, oracle-condition posterior
  or a one-feature DCM fit. The synthetic establishes the *mechanism* and validates the
  estimator; it does not predict real-data effect sizes. ChaosNLI numbers will be far smaller.
- `ours` shows 0.0 regret in the oracle condition **by construction** (it ranks by the true
  objective); the informative comparisons are the other arms, and the estimated condition.
- The info-objective results (not tabulated here) mirror sq-err throughout; CSVs contain both.

## What this unblocks

- §2 and §3 of the paper are writable now (pending the concentration-framing sign-off flagged
  in `paper_plan.md` §2).
- The remaining experimental unknown is entirely on the real-data side: the judge ensemble
  (blocked on API keys in this environment) and ChaosNLI's empirical $\rho$.

## S7a — truth-first misspecified judge (run 18 Aug, day 6; predictions P1–P3 registered 17 Aug)

Truth drawn first; judge prediction quality tracks familiarity; stated confidence corrupted by
knob $m$ (0 honest, 1 flat/zero-information, >1 inverted). Scored against the exact prophet
oracle (value of one more true-label for the estimator's posterior). Advantage of ours over
label-free entropy ranking at B=10%, 100 seeds (50 for the OOD variant):

| $m$ | 0 | 0.25 | 0.5 | 0.75 | **1.0 flat** | 1.5 inverted |
| --- | --- | --- | --- | --- | --- | --- |
| advantage (pp), base features | 51.7 | 51.1 | 50.1 | 48.8 | **3.6** | 47.0 |
| advantage (pp), + OOD channel (r=0.5) | 52.5 | 51.7 | 50.9 | 49.6 | **23.5** | 47.4 |
| fitted β on stated confidence | −0.63 | −0.68 | −0.73 | −0.77 | −0.01 | **+1.09** |

**Scoring the pre-registered predictions honestly:**

- **P1** (LLM overconfident on vague descriptions) — untestable until S7b runs; open.
- **P2** (counts-only backbone survives all tested miscalibrations) — **REFUTED at m=1, confirmed
  elsewhere.** At exactly zero-information confidence, with judge fidelity varying and nothing
  observable tracking it, the advantage collapses to 3.6pp and ours falls behind
  fewest-labels-first. The failure needs *both* conditions; inverted confidence (m=1.5) is fine
  because the regression flips the coefficient sign and exploits it.
- **P3** (regression down-weights stated confidence as miscalibration grows) — **refuted in its
  monotone form, confirmed in mechanism.** |β| *grows* through m=0.75, hits ~0 at flat, then
  flips positive at inversion: the regression does not gradually distrust the signal, it
  *re-learns the mapping* — which is stronger behaviour than predicted.

**The deployment sentence this yields:** epistemic escalation is robust to how *wrong* a
judge's confidence is, including fully inverted; its one dead spot is confidence that carries
no information while prediction quality varies silently — and a judge-independent familiarity
channel of even middling quality (r=0.5 here; embeddings/label-density in production) recovers
most of the advantage (3.6 → 23.5pp). `modelling.md` §2 called that channel a required
component on theoretical grounds; this is its empirical receipt.

*Prophet-oracle note:* value is defined for the estimator's own posterior (what a label buys
*you*, given your model) — the decision-relevant quantity, and the same definition the ChaosNLI
pipeline uses. Regret can exceed 1 because individual labels can have negative realised value.

## S7b replication (day 6 late, write-up audit — `s7b_replication.py`)

The v1 draft audit re-ran every S7b statement under 200 reveal-resamples (counts redrawn
~ Multinomial(n_i, p_i); the LLM judged descriptions only, so its cached predictions are valid
under any reveal draw). **This corrects the record on the single-run arm table:**

- **The between-arm regret differences do NOT replicate.** Ours-vs-entropy advantage on the
  500-die world, OOD features: B=5% +4.1pp CI (−50, +36); B=10% +6.3pp CI (−25, +28); B=20%
  +4.3pp CI (−26, +23). Base features similar. The single-run "11–23pp at B≥5%" quoted in
  earlier discussion (and in `WRITEUP_BRIEF.md` §3) was reveal-draw noise. **No per-budget
  advantage claim from S7b goes in the paper.** Consistent with the incumbent contrast below:
  on dice, H(q̂) is accidentally a strong epistemic proxy (uniform-when-ignorant), so the world
  cannot separate the arms at N=500 — which is itself the sharpest ChaosNLI justification.
- **What replicates:** tier calibration (reveal-independent); verbalised-vs-random paired at
  B=10%: +1.8pp CI (−14.1, +18.4) — verbalised-confidence escalation is *indistinguishable from
  random*, the honest form of "worst arm"; the isolation matrix (CIs below); the diagnostic
  (0.258 vs 0.307 base / 0.262 with familiarity features, n=380).
- Isolation matrix over 200 reveal draws (OOD features), mean (95% CI):
  est-aleatoric vs true-aleatoric **0.446** (0.316, 0.550); vs true-error −0.119 (−0.335, 0.109);
  est-epistemic vs true-aleatoric 0.080 (0.050, 0.103); vs true-error **0.656** (0.511, 0.698);
  truth–truth −0.227 (−0.356, −0.076); H(q̂) vs true-error 0.573 (0.508, 0.608), vs
  true-aleatoric 0.117.
- Calibrated-world isolation over 20 seeds: diag 0.766 (0.756, 0.778) / 0.482 (0.461, 0.505);
  off-diag 0.127 / −0.142; truth–truth −0.031.

## The isolation exhibit (added day 6 evening — `isolation.py`)

The direct answer to "have we demonstrated the aleatoric/epistemic split", graded at the
*recovery* level rather than via decision regret: per world, does each estimated component
track its own ground truth (diagonal) and ignore the other's (off-diagonal)? Spearman:

| world | est-aleatoric vs true-aleatoric | est-epistemic vs true-error | off-diagonals | truth–truth baseline |
| --- | --- | --- | --- | --- |
| calibrated, estimated posterior (r=0.7 proxy) | **0.76** | **0.50** | 0.14 / −0.16 | −0.05 |
| **real LLM (S7b)**, OOD-featured posterior | **0.46** | **0.63** | 0.02 / 0.09 | −0.34 |

Both worlds isolate: strong diagonal, near-zero off-diagonal — in S7b *despite* the two truths
being anti-correlated (−0.34), so the separation is not a lucky world correlation.

**Incumbent contrast (S7b):** total predictive entropy $H(\hat q)$ tracks true error at 0.60 —
nearly as well as our epistemic estimate (0.63) — because this LLM answers *uniform when
ignorant*, confounding its prediction's flatness with its ignorance on dice. The dice world
therefore under-populates the confidently-predicted-split quadrant (high aleatoric, low
epistemic), which is precisely where entropy-ranking pays its signature price, and which real
NLI items supply (the ChaosNLI smoke showed the judge readily predicting 12/72/16-type splits).
This is the sharpest statement of what the ChaosNLI run is *for*.

### CI-semantics note (21 Aug, from the second-reviewer pass)

The 200-draw percentile intervals above are **per-draw sensitivity intervals** over the reveal
randomness on the fixed world — they answer "what might one draw show", not "is the expected
effect zero". The expected ours-vs-entropy advantage on the S7b world is ≈ +6pp at Monte-Carlo
SE ≈ 1 (small but positive); "does not replicate" is precise about the original 11–23pp
single-draw numbers, and the paper phrases it that way (draft v1.2 §3.3).

## S7b convergence curve (27 Aug — the paper's hero figure)

`convergence_figure.py`: from the committed cache + `build_world(500, 0)`; ood features
(tier one-hots + variant spread + log1p(n) + stated confidence); trust fitted per reveal
draw on the deployed design (seeds 10_000+r, 200 draws, exactly `s7b_replication.py`'s
convention), then held fixed while rolls accrue cumulatively. The script refuses to draw
unless the regenerated per-tier prior errors match the published tier table
(0.53 / 0.54 / 0.41 / 0.007) to 3 dp.

Mean Manhattan error to the true face distribution (judge prior alone, never updated:
flat 0.371):

| n rolls | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours (fitted trust + rolls) | 0.371 | 0.339 | 0.315 | 0.291 | 0.273 | 0.258 | 0.245 | 0.233 | 0.224 | 0.215 | 0.208 |
| counts only (add-one, no judge) | 0.532 | 0.460 | 0.420 | 0.384 | 0.357 | 0.336 | 0.317 | 0.301 | 0.288 | 0.275 | 0.265 |
| model's own predicted spread v | 0.119 | 0.086 | 0.069 | 0.059 | 0.051 | 0.045 | 0.041 | 0.037 | 0.034 | 0.031 | 0.029 |

Two derived statements used in the paper: ten rolls repair 44% of the judge's error
((0.371 − 0.208)/0.371 = 0.439); counts-only at n=10 (0.265) sits between ours at n=4
(0.273) and n=5 (0.258), so the judge's prediction is worth roughly five rolls.

### Decomposition-in-motion extension (same run, lead's approval 27 Aug)

The two-panel paper figure (`fig3_s7b_decomposition.png`) adds panel (a): the estimated
components per revealed roll, same draws and fit as above. True mean pool disagreement
H(p) = 0.892 nats (exact, from the regenerated world).

| n rolls | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| est. total H(m) | 1.042 | 0.994 | 0.972 | 0.958 | 0.949 | 0.942 | 0.936 | 0.932 | 0.929 | 0.926 | 0.923 |
| est. aleatoric E[H(p)|c] | 0.852 | 0.850 | 0.853 | 0.857 | 0.860 | 0.863 | 0.865 | 0.867 | 0.869 | 0.870 | 0.871 |
| est. epistemic I | 0.190 | 0.144 | 0.118 | 0.101 | 0.089 | 0.079 | 0.072 | 0.065 | 0.060 | 0.056 | 0.052 |

Reading: the aleatoric estimate holds level, 2–4% under the true 0.892 line (a slight
under-call inherited from the judge's over-confident predicted distributions, closing as
rolls arrive); the epistemic estimate falls 73% in ten rolls; total = aleatoric +
epistemic at every n, so total uncertainty descends onto the aleatoric floor. This is
the "labels drain ignorance, not disagreement" claim measured on the real judge.
