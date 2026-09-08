"""Page map for the workshop paper: where each section lands, what each costs, and how
far over the 4-page content budget we are.

Per NeurIPS formatting instructions the references do not count towards the limit, so
"content" runs from page 1 to the page the bibliography starts on (the body occupies part
of that page, so the figure is a slight over-count of what must be cut, not an under-count).
"""
import re
import subprocess
import sys

LIMIT = 4  # JUDGe poster track, ruled 27-28 Aug (CUT_PLAN_4P.md); the frozen 6-page
# version lives in main_6p.tex/pdf

aux = open("main.aux").read()
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux):
    labels[m.group(1)] = (m.group(2), int(m.group(3)))

log = open("main.log").read()
total = int(re.search(r"Output written on main\.pdf \((\d+) pages", log).group(1))
refs_page = labels.get("sec:refs", (None, total))[1]

ORDER = [
    ("sec:intro", r"\\section\{Introduction\}", "1. Introduction"),
    ("sec:related", r"\\section\{Related work\}", "2. Related work"),
    ("sec:model", r"\\section\{Model\}", "3. Model"),
    ("sec:setup", r"\\section\{Experimental setup\}", "4. Experimental setup"),
    ("sec:results", r"\\section\{Results\}", "5. Results"),
    ("sec:sims", r"\\subsection\{Simulations", "  5.1 Simulations"),
    ("sec:exp1", r"\\subsection\{Experiment 1", "  5.2 Experiment 1"),
    ("sec:exp2", r"\\subsection\{Experiment 2", "  5.3 Experiment 2"),
    # S7 folded into S6's end per the lead's D2 ruling (28 Aug, CUT_PLAN_4P.md)
    ("sec:limitations", r"\\section\{Limitations", "6. Limits + conclusion"),
    ("sec:refs", r"\\label\{sec:refs\}", "References (exempt)"),
]

tex = open("main.tex").read()
bounds = []
for key, pat, name in ORDER:
    m = re.search(pat, tex)
    bounds.append((key, name, m.start() if m else None))

abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)


def words(chunk):
    chunk = re.sub(r"^%.*$", " ", chunk, flags=re.M)
    chunk = re.sub(r"\\(begin|end)\{[^}]*\}", " ", chunk)
    chunk = re.sub(r"\\[a-zA-Z]+\*?", " ", chunk)
    chunk = re.sub(r"[{}$&_^~\\]", " ", chunk)
    return len(chunk.split())


rows = []
if abstract:
    rows.append(("Abstract", "", words(abstract.group(1))))
for i, (key, name, pos) in enumerate(bounds[:-1]):
    nxt = bounds[i + 1][2]
    if pos is None or nxt is None:
        continue
    page = labels.get(key, ("", 0))[1]
    rows.append((name, page, words(tex[pos:nxt])))

body_words = sum(w for _, _, w in rows)
# S5 is a bare heading; its cost is its three subsections, shown indented beneath it.
sub_total = sum(w for n, _, w in rows if n.startswith("  "))
print(f"{'section':<26}{'starts p.':>10}{'words':>8}{'share':>8}")
print("-" * 52)
for name, page, w in rows:
    if name.startswith("5. Results"):
        w = w + sub_total
    share = "" if name.startswith("  ") else f"{100 * w / body_words:.0f}%"
    print(f"{name:<26}{str(page):>10}{w:>8}{share:>8}")
print("-" * 52)
print(f"{'body total':<26}{'':>10}{body_words:>8}")

floats = [("fig:convergence", "Fig 1 (hero)"), ("fig:sims", "Figure 2"),
          ("fig:robust", "Figure 3"), ("tab:isolation", "Table 1"),
          ("tab:confidence", "Table 2"), ("tab:escalation", "Table 3")]
print("\nfloats:", ", ".join(
    f"{n} p.{labels[k][1]}" for k, n in floats if k in labels))

overfull = len(re.findall(r"Overfull \\hbox", log))
print(f"overfull hboxes: {overfull}")

print(f"\ntotal pages    {total}")
print(f"content pages  {refs_page}   (body runs to p.{refs_page}, "
      f"references start partway down it)")
print(f"limit          {LIMIT}")
over = refs_page - LIMIT
if over > 0:
    print(f"OVER BY        {over} pages -- must cut roughly "
          f"{100 * over / refs_page:.0f}% of the body")
else:
    print("WITHIN BUDGET")
sys.exit(0)
