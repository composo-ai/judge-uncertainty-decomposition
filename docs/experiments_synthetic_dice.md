# E1/E2 Spec — Synthetic Dice Experiment

Implements days 1–5 of `paper_plan.md` §3 and §5. This file closes every decision the plan left
open so the generator can be written tomorrow morning without design choices happening at the
keyboard. Anything marked **[ratify]** is a pre-registration value that Seb/Luke should confirm
or amend *before the sweep runs* — after seeing curves it is no longer pre-registered.

> ## Changelog — implementation findings (day 0, code in `dice/`)
>
> Building the harness falsified two things in the original version of this spec. Both are
> corrected below; the diagnostic numbers are from `dice/test_dice.py` and small runs at
> $N = 2000$, 5 seeds.
>
> **1. The Δ closed form was wrong.** The one-step reduction in $\sum_j \mathrm{Var}(p_j)$ is
> $\Delta = G/(\alpha_0+1)^2$, not $G/((\alpha_0+1)(\alpha_0+2))$. Caught by T1's exact conjugate
> computation (0.013605 vs 0.011905 at $\alpha = (2,2,2)$). Fixed in `posterior.py`; the same
> error is corrected in `paper_plan.md` §5.
>
> **2. The null axis was the wrong axis.** The original null ($\kappa = \infty$: constant shape)
> is not a null at all: aleatoric entropy still varies with sd 0.218 through $\lambda$, and the
> incumbent's regret there is **0.93 — worse than random (0.82)** — because at constant shape,
> entropy is blind to concentration and mildly anti-correlated with value through counts.
> Structurally: **in the Dirichlet family, at fixed concentration, acquisition value
> $G/(\alpha_0+1)^2$ INCREASES with contentiousness** — the Gaussian "split pools are worthless
> to label" intuition does not transfer verbatim, because DCM has no separate noise channel.
> The DCM-correct proposition is:
>
> > *Entropy-ranking is near-optimal **iff concentration $\alpha_0$ is homogeneous across
> > items.** The inversion is concentration-driven, not contentiousness-driven.*
>
> Verified: at the corrected null (constant $\lambda$ and $n$, shape varying) incumbent regret
> is **0.000**; in the epistemic-only regime (shape constant, $\lambda$ varying) it is **0.91**
> vs random 0.85, ours ≈ 0. The primary sweep axis is therefore **epistemic heterogeneity**
> (`f_width`), with shape heterogeneity ($\kappa$) demoted to a secondary sweep. §§1.4, 5, 6
> below are the revised versions.
>
> **Consequence for the paper (needs Seb/Luke sign-off, day 1):** §2's Gaussian counterexample
> is fine as a Gaussian statement, but the categorical section must state the proposition in
> concentration terms, and "escalate ignorance, not disagreement" becomes *more* literal — in
> the categorical model, disagreement-escalation fails precisely and only because it cannot see
> ignorance. An added arm `incumbent_prior` (entropy of the judge's label-free prediction — what
> Trust-or-Escalate-style escalation actually ranks by) covers the deployed practice; the
> posterior-entropy incumbent is kept alongside it.

## 0. Purpose and exit criteria

Go/no-go gate for the workshop paper. Two questions:

1. Does ranking by an epistemic score beat ranking by predictive entropy, by the margin theory
   predicts, when aleatoric heterogeneity is present?
2. How good must the epistemic estimate be (the $\rho^\ast$ threshold) before that advantage
   survives estimation error?

**Pre-registered exit criteria [ratify]** *(axes revised per changelog; the diagnostic runs that
motivated the revision touched neither the S4 criterion cell at its registered seed count nor
the estimated condition, so pre-registration of the criterion itself stands)*:

- **Pass:** at `f_width = 1`, $\rho = 0$, budget $B = 10\%$:
  $\text{regret}_{\text{incumbent\_prior}} - \text{regret}_{\text{ours}} \geq 15$ percentage
  points with a 95% bootstrap CI excluding 0 **in the estimated-posterior condition at
  $r = 1$**, and the estimated-condition regret curve tracks the oracle-condition curve across
  the `f_width` sweep with Spearman $\geq 0.9$ (theory = oracle condition; measurement =
  estimated condition).
- **Weak pass:** advantage significant but either the margin $< 15$pp or the tracking Spearman
  $< 0.9$ → empirical-finding framing, drop the theory-overlay claim.
- **Fail:** CI includes 0 → the negative-result framing; measure ChaosNLI's $\rho$ anyway.

## 1. Generator

$K = 3$ throughout. All draws vectorised numpy; single `default_rng(seed)` per replicate.

### 1.1 Latent item variables, with the $\rho$ copula

Two scalars per item: contentiousness $t_i \in [0,1]$ (aleatoric knob) and familiarity
$f_i \in [0,1]$ (epistemic knob). To impose correlation $\rho$ between aleatoric and epistemic:

```
(z1, z2) ~ BivariateNormal(0, [[1, rho], [rho, 1]])
t_i = BetaInvCDF( Phi(z1); a=kappa, b=kappa )      # mean 0.5, spread set by kappa
f_i = Phi(-z2)                                     # so high t (contentious) pairs with low f
                                                   # (unfamiliar) when rho > 0
```

Report the **realised** Spearman between $\mathbb{E}[H(p)]$ and $I$ per config — that, not the
nominal $\rho$, is the axis value plotted.

### 1.2 Judge belief

```
q_i     = (1 - t_i) * perm_i(0.90, 0.05, 0.05) + t_i * (1/3, 1/3, 1/3)
lam_i   = 0.5 * 256**f_i                           # log-uniform on [0.5, 128]
alpha_i = clip(lam_i * q_i, 0.05, None)            # clamp per plan; spiky-draw guard
```

`perm_i` is a fresh random permutation per item so no class is globally privileged.

### 1.3 Truth, labels, posterior

```
p_i        ~ Dirichlet(alpha_i)                    # truth drawn FROM belief (calibrated)
n_i        ~ UniformChoice({0, 1, 2, 5, 10})
counts_i   ~ Multinomial(n_i, p_i)
post_i     = alpha_i + counts_i                    # conjugate posterior parameters
```

### 1.4 Heterogeneity knobs and the null *(revised — see changelog)*

Two knobs, and the primary one is **epistemic**:

- **`f_width` ∈ {0, 0.1, …, 1.0}** — familiarity spread: $f_i = 0.5 + \text{f\_width} \cdot
  (\Phi(z_f) - 0.5)$. At 0, every item has $\lambda = 8$; at 1, log-uniform on $[0.5, 128]$.
  **This is the axis the proposition lives on.**
- **$\kappa$** — shape spread as before, demoted to a secondary robustness sweep.

**The null** is constant concentration: `f_width = 0` **and** `n_fixed = 5` (both $\lambda$ and
$n$ enter $\alpha_0$, so the null needs both pinned). There, entropy-ranking is near-optimal
(measured incumbent regret 0.000) because value $\propto G(m)$ and entropy $H(m)$ are
rank-aligned Schur-concave functionals of the same posterior mean. The original constant-shape
"null" is retained as the **epistemic-only regime** — the paper's most striking cell, where the
incumbent is worse than random.

## 2. Closed forms

For $\mathrm{Dir}(\alpha)$ with $\alpha_0 = \sum_j \alpha_j$, $m_j = \alpha_j/\alpha_0$,
$G = 1 - \sum_j m_j^2$, $\psi$ = digamma:

| Quantity | Formula |
| --- | --- |
| Total (incumbent's score) | $H(m) = -\sum_j m_j \log m_j$ |
| Aleatoric | $\mathbb{E}[H(p)] = \psi(\alpha_0 + 1) - \sum_j m_j\, \psi(\alpha_j + 1)$ |
| Epistemic / log-loss acquisition | $I = H(m) - \mathbb{E}[H(p)]$ |
| Posterior spread | $v = G / (\alpha_0 + 1)$ |
| Squared-error acquisition | $\Delta = G / (\alpha_0+1)^2$ *(corrected — see changelog)* |

All already verified numerically in the scratchpad (`glm.py`). Unit-test against Monte Carlo
once anyway (§5, T1).

## 3. Conditions and the estimator

### 3.1 Oracle-posterior

Scores computed from the true generating posterior `post_i`. Isolates the decision rule.

### 3.2 Estimated-posterior

The estimator receives, per item: $q_i$ (the judge's mean prediction — observable in
production), `counts_i`, $n_i$, and a **noisy proxy** $\hat f_i$. It never sees $\lambda_i$,
$f_i$, $t_i$, or $p_i$.

Proxy corruption on normal scores, informativeness $r$ swept over
$\{0, 0.1, \ldots, 1.0\}$ (11 points):

```
zf_hat = r * z_f + sqrt(1 - r^2) * eps,   eps ~ N(0,1);   f_hat = Phi(zf_hat)
```

Report realised Spearman$(\hat f, f)$ per cell.

**Estimator = the same Dirichlet–multinomial regression as the real-data method** (one feature
here instead of several):

$$\log \hat\lambda_i = \beta_0 + \beta_1 \hat f_i$$

$\beta$ fitted by minimising the DCM negative log-likelihood (formula in `paper_plan.md` §5)
over items with $n_i \geq 1$, via `scipy.optimize.minimize` (BFGS, init $\beta = (log 8, 0)$).
Estimated posterior $= \mathrm{Dir}(\hat\lambda_i q_i + \text{counts}_i)$. Items with $n_i = 0$
get $\hat\lambda_i$ from the regression — this *is* the cross-item pooling that gives zero-label
items an epistemic estimate, mirroring E3.

## 4. Arms, objectives, regret

Six arms, exactly the `paper_plan.md` §3 table:

| Arm | Score (descending = escalate first) |
| --- | --- |
| Oracle | true one-step value under **the objective being scored**, from `post_i` |
| Ours (info) | $\hat I$ |
| Ours (sq-err) | $\hat\Delta$ |
| Selective ($v$-only) | $\hat v$ |
| Practitioner | $-n_i$ (fewest labels first; ties broken randomly) |
| Incumbent | $H(\hat m)$ of the estimated posterior (labels folded in) |
| Incumbent-prior | $H$ of the judge's label-free prediction — what Trust-or-Escalate-style disagreement escalation actually ranks by; the primary comparison arm |
| Random | uniform permutation |

Two objectives, each with its own oracle: **info** (true one-step expected information gain
$= I$ of `post_i`) and **sq-err** (true $\Delta$ of `post_i`). Every arm's ranking is scored
under both. Never mix — the negative-regret bug from review.

$$\text{regret}(B) = \frac{\sum_{\text{oracle-top-}B} \text{val}_i - \sum_{\text{arm-top-}B} \text{val}_i}{\sum_{\text{oracle-top-}B} \text{val}_i}, \qquad B \in \{1, 5, 10, 20\}\%$$

Headline cell: $B = 10\%$. **Theory overlay** *(revised — see changelog)*: the original
"predicted regret from $n{=}0$ generator quantities" was defined against the pre-revision null
and is superseded. Theory = the oracle-posterior condition (perfect posterior, decision rule
isolated); measurement = the estimated-posterior condition at $r = 1$. "The gap is the size
theory says" now means the estimated curve tracks the oracle curve across the `f_width` sweep.

## 5. Tests (pytest, `@pytest.mark.unit`, co-located per repo convention)

- **T1 — closed forms vs Monte Carlo.** Each §2 formula within MC error of 200k-draw estimates,
  across a grid of $(\alpha)$ shapes including a clamped near-vertex case.
- **T2 — null anchor** *(revised)*. At the DCM-correct null — `f_width = 0`, `n_fixed = 5`,
  $\kappa = 1$, $\rho = 0$, oracle condition — incumbent-vs-ours advantage $\leq 2$pp at
  $B = 10\%$ (measured: 0.0pp). **[ratify** the 2pp**]**
- **T2b — epistemic-only regime.** At $\kappa = \infty$, `f_width = 1`: incumbent-prior regret
  $\geq 50\%$ and ours $\leq 5\%$ — pins the regime where entropy is blind to concentration
  (measured: 84% vs 0.0%).
- **T3 — calibration (the conditioning-direction test).** Truth is drawn from belief, so the
  PIT of $p_{i,1}$ under $\mathrm{Dir}(\alpha_i)$ must be uniform across items (KS $p > 0.01$),
  and central 90% posterior credible intervals must cover at $0.90 \pm$ MC error after
  conditioning on counts. This is the test that fails if anyone reintroduces the old
  truth-first generator.
- **T4 — no negative regret.** Under matched objectives, oracle regret $= 0$ and every arm's
  regret $\geq -\epsilon_{\text{MC}}$. A violation means objectives got mixed again.
- **T5 — copula sanity.** At $\rho = 0$: realised Spearman$(\lambda_i, H(q_i))$ within
  $\pm 0.05$. At $\rho = 0.8$: realised Spearman positive and $> 0.5$.
- **T6 — proxy monotonicity.** Realised Spearman$(\hat f, f)$ increases in $r$ and hits
  $\approx 0$ and $\approx 1$ at the endpoints.

## 6. Sweep matrix and compute

| Sweep | Grid | Seeds | Condition |
| --- | --- | --- | --- |
| S1 — epistemic heterogeneity *(primary)* | 11 `f_width` × $\kappa = 1$ × $\rho = 0$ | 200 | oracle |
| S1shape — shape heterogeneity *(secondary)* | 11 $\kappa$ × `f_width` $= 1$ × $\rho = 0$ | 100 | oracle |
| S2 — phase diagram | 11 `f_width` × $\rho \in \{-0.8,-0.4,0,0.4,0.8\}$ | 50 | oracle |
| S3 — $\rho^\ast$ threshold | 11 $r$ × `f_width` $\in \{0.25, 0.5, 1.0\}$ × $\rho = 0$ | 100 | estimated |
| S4 — headline CI | the single exit-criterion cell | 500 (bootstrap) | both |

$N = 2000$ items per replicate. Everything numpy; S2 is the largest at ~5.5M items and should
run in well under a minute. If it doesn't, something is unvectorised.

**Log every dropped or clamped configuration** — silent truncation reads as full coverage.

## 7. Outputs

```
experiments/epistemic_uncertainty/dice/
  generator.py      # §1
  posterior.py      # §2 closed forms + DCM fit (§3.2) — shared with the ChaosNLI adapter later
  arms.py           # §4
  sweep.py          # §6; writes results/*.csv, one row per (config, seed, arm, objective, B)
  figures.py        # fig1_phase.png, fig_rho_star.png, fig_null_anchor.png
  test_dice.py      # §5
```

`posterior.py` is deliberately dataset-agnostic: the day 6–7 ChaosNLI work should import it
unchanged, swapping only the feature vector and $\hat q$ source. That is the "one code path"
promise made concrete.

## 8. S7 — misspecified and real judges (added day 5, designed with Ryan; not yet built)

The calibrated generator makes "can we isolate the two uncertainties" true by conjugacy — the
world *is* the model. S7 removes that assumption in two steps, and turns the two biggest
conceded objections (`anticipated_reviews.md` R1.2 circularity, R3.1 confidently-wrong judge)
into measurements. Design settled in discussion 17 Aug; **build on day 6, run S7a immediately,
S7b when keys arrive.**

### S7a — simulated misspecification sweep (numpy, free)

Truth-first generator variant: draw $p_i$ first; the judge's $\hat q_i$ is a noisy function of
$p_i$ with noise increasing as familiarity falls, and the judge's *stated* confidence is
deliberately miscalibrated (knob: overconfidence on unfamiliar items — the realistic
direction). The judge's stated posterior is no longer the Bayes posterior, so the conjugate
oracle dies — but the true $p_i$ is known, so the **prophet oracle** (realised one-step value
against truth, as in the ChaosNLI pipeline's acquisition evaluation) is exact. Sweep
miscalibration strength × familiarity-noise coupling; reuse `chaosnli/pipeline.py` with truth
$= p_i$ instead of half-B frequencies.

### S7b — real LLM judge on dice with controlled truth (the layer between §3 and §4)

~500 dice with known $p_i$, each carrying a **textual description whose informativeness we
control** — precise ("weighted ~4-to-1 toward face 1"), vague ("from a batch where some dice
are weighted"), near-empty ("a three-sided die"). Description informativeness *is* familiarity,
made real. One LLM judge (keep it to one, per discussion) reads the description **only** and
states a distribution via the same elicitation family as the ChaosNLI prompts (+ variants for
spread, + verbalised confidence). Rolls are revealed as simulated expert labels; the DCM does
the updating, as in production. Oracle exact, judge behaviour real.

*Optional diagnostic arm:* the LLM also sees the rolls and self-updates; compare against
conjugate updating on its own stated prior. Tests "let the judge read, let Bayes update".

~2,500 calls (500 × 1 model × (3 variants + confidence) + diagnostic subset). **Same key
queue as the ChaosNLI cache.** Cache-first, same runner pattern as `chaosnli/judge.py`.

### Pre-registered predictions (fixed before either runs)

- **P1.** The LLM is overconfident specifically on vague descriptions — miscalibration
  correlated with familiarity.
- **P2.** The counts-only backbone survives all tested miscalibrations: revealed labels beat
  entropy-ranking regardless of judge pathology, because labels are the one signal a
  miscalibrated judge cannot corrupt.
- **P3.** The DCM spread feature partially corrects stated confidence (fitted weight on stated
  confidence shrinks as miscalibration grows in S7a; on S7b the regression down-weights it
  relative to the calibrated-world fit).

### The three evidence layers (paper §3 closing panel)

| Layer | Judge | Truth | Establishes |
| --- | --- | --- | --- |
| S1–S6 | simulated, calibrated | exact | mechanism; decision rule wins when uncertainty is right |
| S7a/b | simulated-miscalibrated / **real LLM** | exact | isolation survives judge pathology; the real LLM located as a *point* on S7a's swept phase map |
| ChaosNLI (§4) | real LLM | ~100 human labels | external validity on real items |

Paper cost: one paragraph + one figure panel, folded into §3 — not a new section.

## 9. What this spec deliberately does not decide

- The final paper framing (two-way vs three-way) — deferred to the day-8 ChaosNLI scatter per
  `paper_plan.md` §1.
- ChaosNLI feature set for the multi-feature DCM regression — day 6, informed by S3's $\rho^\ast$.
- Whether S2's inverted-regime panel ($\rho < 0$ cells) makes the paper — presentation, day 5.
