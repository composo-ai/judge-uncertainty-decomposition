"""Protected-content gate for the 6-page build (CUT_PLAN v2.6 execution).

Re-runnable check that every invariant, verbatim, and structural protection survives in
main.tex and the rendered PDF. Run after any edit and before any commit:

    python3 gate.py            # exits non-zero on any failure

Committed at reviewer request (PR #1573 round 1): after the v2.4/v2.5 episode,
verification lives in the repo, not in session transcripts.
"""
import re
import subprocess
import sys

# Optional variant run: `python3 gate.py main_tae 8` checks the TAE build
# against the same pins with its own page target. Default: main, 4.
BASE = sys.argv[1] if len(sys.argv) > 1 else "main"
PAGE = sys.argv[2] if len(sys.argv) > 2 else "4"

tex = open(BASE + ".tex").read()
# Phrase pins match against whitespace-collapsed source: the pins are word
# sequences, and fit passes rewrap lines constantly (three separate wrap
# breakages before this normalisation). Raw `tex` remains for checks where
# bytes matter (TAB corruption).
flat = re.sub(r"\s+", " ", tex)
pdf = subprocess.run(["pdftotext", BASE + ".pdf", "-"],
                     capture_output=True, text=True).stdout
try:
    log = open(BASE + ".log").read()
    aux = open(BASE + ".aux").read()
except FileNotFoundError:  # clean checkout: build intermediates are gitignored
    log = aux = None

checks = [
    # S5.1 plain-language ruling (lead, 28 Aug) then the full P-label purge
    # (lead, 29 Aug: "I think they should all go!"): no P-numbers anywhere in
    # either build; App A's predictions-as-scored record keeps both
    # refutations in plain words. Same move as the Delta null
    # (EDITORIAL_RULES 16-17, 19-20).
    ("misspecified predictions scored honestly in App A",
     "was refuted at $\\gamma=1$ and confirmed elsewhere" in flat
     and "refuted in its monotone form" in flat),
    ("'as the §3 claim requires' cross-ref",
     "as the \\secref{sec:model} claim requires" in flat),
    # Body-purge ruling (lead, 28 Aug, after a third vocabulary query: the
    # disclosure apparatus "is not required... negative results can live in the
    # appendices"). The Delta null left S5.3 and the E2 caption; the record is
    # the appendix (App D in main; the in-body table block in the TAE build):
    # named inconclusive, interval intact.
    ("Delta-null recorded with interval (appendix record, not body)",
     "inconclusive" in flat and "$+4.5$ $[-2.2, +9.4]$" in flat),
    # >=3 -> >=1 under the same body-purge ruling: the planned/exploratory
    # labelling lives in the escalation-table record, not the body.
    ("exploratory labelling present in the record", tex.count("exploratory") >= 1),
    # "arms" -> "rules" per the lead's jargon query (28 Aug); same disclosure.
    ("seven-comparisons caveat", "Seven rules are compared" in flat),
    ("mis-specification owned", "not about $\\lambda$ alone" in flat),
    ("fewest-labels-first not beaten w/ interval", "$+0.7$ $[-6.3, +7.1]$" in flat),
    # >=3 -> >=2 per the lead's D7 ruling (28 Aug): S1's site dropped at 4 pages;
    # abstract + S5.3 remain. The disclosure escorts travel with both.
    ("83% headline at least 2 sites (D7)", pdf.count("83%") >= 2),
    ("Dubois verbatim",
     "theirs separates grader severity offsets from residual disagreement" in flat),
    ("mechanism-credit", "The mechanism is theirs" in flat),
    ("Kotelevskii same-object", "same object, not rivals" in flat),
    ("half-split protocol",
     "evaluation truth is never touched by training" in flat),
    ("tier-is-familiarity-feature",
     "each die's tier as its familiarity feature" in flat),
    # The full when-safe paragraph moved S3 -> App B in main (lead's space
    # ruling, 28 Aug, to pay for the S5.1 explanation); S3 keeps a one-sentence
    # bridge with the same phrase. main_tae keeps the paragraph in S3.
    ("when-safe inversion claim", "up to full inversion" in flat),
    # The displays live in App B since the lead's 28 Aug estimator-to-appendix
    # ruling (EDITORIAL_RULES Part 6.7); these checks grep the whole file by design.
    ("belief-family three displays intact",
     flat.count("p_x \\sim \\mathrm{Dir}(\\lambda_x \\hat q_x)") == 1
     and "c_x \\sim \\mathrm{Multinomial}" in flat
     and "\\log \\lambda_x = \\beta_0" in flat),
    ("posterior + alpha_0 display", "\\alpha_{0,x} = \\lambda_x + n_x" in flat),
    # \textbf -> \emph here reflects the lead's 27 Aug de-bold ruling; content unchanged.
    ("S8 aleatoric grounding", "is the \\emph{aleatoric} component" in flat),
    ("judge named + query date", "gpt-5.6-terra" in flat and "18 August 2026" in flat),
    ("Lail cited in the intro and present in references (lead's ruling, 27 Aug)",
     "lail2026" in tex and "Lail" in pdf),
    ("worked example fully gone, no orphans",
     "Item A" not in flat and "2.8\\times" not in flat),
    ("gamma knob consistent, posterior mean untouched",
     "$\\gamma=1$" in flat and "m_x = \\alpha_x / \\alpha_{0,x}" in flat),
    ("no 'model calls' residue", "model calls" not in pdf),
    # A-D for main; the TAE variant has A-C (App D's table moved into S5.3).
    ("appendices present (A-D main, A-C tae)",
     all(x in tex for x in ["app:sims", "app:model", "app:exp1"])
     and (BASE != "main" or "app:exp2" in tex)),
    # hero-swap, lead's ruling 27 Aug: the convergence figure is the one
    # main-body float; the escalation table lives in App D, still referenced
    # from S5.3 with all its numbers in the prose.
    # main keeps the table in App D; the TAE variant promotes it into S5.3
    # and drops the then-empty App D, so app:exp2 is main-only.
    ("escalation table present (App D for main)",
     "tab:escalation" in tex and (BASE != "main" or "app:exp2" in tex)),
    # Single-panel since 28 Aug: the lead executed reserve-ladder item 1 (drop
    # panel (b)) to pay for the round-4 exhibit tables. The worth-five-rolls
    # numbers (0.371, 0.208, 44%) leave the 4-pager; they remain in the TAE
    # build, App A's record, and dice/RESULTS.md.
    ("hero decomposition figure in main (single-panel, ladder reserve 1)",
     "fig3_s7b_decomposition" in tex and "fig:convergence" in tex),
    ("hero figure caption numbers (floor 0.892, I 0.190->0.052)",
     all(x in tex for x in ["0.892", "0.052"])
     and "labels drain ignorance, not disagreement" in flat),
    # The round-6 settlement pin (the abstract's registered-miss sentence) was
    # RETIRED by the lead's live-review ruling, 28 Aug ("It's not required"):
    # the abstract keeps the fewest-labels tie clause; the registered-miss
    # disclosure lives in S5.3 (gate: 'registered null leads S5.3'), whose
    # architecture is untouched. EDITORIAL_RULES Part 6 item 11.
    # 4P-plan review round 1: this clause was silently lost to page fit once
    # (restored 27 Aug); the settlement holds only while the S7 summary carries
    # it. It must survive any fold of S7 into S6.
    ("S7 fewest-labels tie clause (settlement rider)",
     "though not more than a fewest-labels-first rule" in flat),
    # Round-4 exhibits (lead's ruling, 28 Aug): the results beats carry small
    # in-body tables. main-only; main_tae's full tables already sit in-body.
    ("S5.2 isolation exhibit in body (E1)",
     BASE != "main" or ("\\label{tab:iso}" in tex and "\\ref{tab:iso}" in tex)),
    ("S5.3 escalation exhibit in body (E2)",
     BASE != "main" or ("\\label{tab:esc}" in tex and "\\ref{tab:esc}" in tex)),
    ("no TAB corruption", "\t" not in tex),
]

# Build-health rows need the current build's log/aux, which are gitignored; on a
# clean checkout they are skipped with a warning, not failed (reviewer round 6).
if log is None:
    print("WARN  main.log/main.aux absent (clean checkout?) -- build-health "
          "checks skipped; run make first for the full gate")
else:
    checks += [
        ("zero overfull boxes (log)", len(re.findall(r"Overfull \\hbox", log)) == 0),
        ("zero undefined refs/cites (log)", "undefined" not in log.lower()),
        # 4 for main (CUT_PLAN_4P.md); the TAE variant passes its own target.
        (f"references begin by p.{PAGE}",
         bool(re.search(r"\\newlabel\{sec:refs\}\{\{[^}]*\}\{(\d+)\}", aux))
         and int(re.search(r"\\newlabel\{sec:refs\}\{\{[^}]*\}\{(\d+)\}",
                           aux).group(1)) <= int(PAGE)),
    ]

fails = 0
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
    fails += (not ok)
print(f"\n{len(checks) - fails}/{len(checks)} checks passed")
sys.exit(1 if fails else 0)
