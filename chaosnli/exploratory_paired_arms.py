"""EXPLORATORY (unregistered): paired per-replicate differences of every arm
vs incumbent_prior at all budgets. PREREG froze only ours_sqerr vs
incumbent_prior as the headline; this recomputes the same 200 replicates and
reports the rest. No API calls — reads the committed judge cache.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chaosnli import analysis as A
from chaosnli.pipeline import BUDGETS_PCT, ARMS

DATA = os.environ.get("CHAOSNLI_DIR", "")  # unpacked chaosNLI_v1.0
CACHE = str(Path(__file__).resolve().parent / "caches" / "judge_cache.jsonl")
R = 200

if __name__ == "__main__":
    with ProcessPoolExecutor(8, initializer=A._init, initargs=(DATA, CACHE)) as ex:
        res = list(ex.map(A._one, [(s, False) for s in range(R)]))

    # per replicate: arm -> budget -> value
    per = []
    for r in res:
        d = {}
        for row in r["acquisition"]:
            d.setdefault(row["arm"], {})[row["budget_pct"]] = row["value"]
        per.append(d)

    def ci(v):
        a = np.array(v, float)
        lo, hi = np.percentile(a, [2.5, 97.5])
        return dict(mean=round(float(a.mean()), 3),
                    mc_se=round(float(a.std(ddof=1) / np.sqrt(len(a))), 3),
                    ci95=[round(float(lo), 3), round(float(hi), 3)])

    out = {"replicates": R, "levels": {}, "paired_vs_incumbent_prior": {}}
    for arm in ARMS:
        if arm not in per[0]:
            continue
        out["levels"][arm] = {f"b{b}": ci([p[arm][b] for p in per]) for b in BUDGETS_PCT}
        out["paired_vs_incumbent_prior"][arm] = {
            f"b{b}": ci([p[arm][b] - p["incumbent_prior"][b] for p in per])
            for b in BUDGETS_PCT
        }
    # head-to-head that matters for the fewest-labels-first question
    out["paired_ours_info_minus_practitioner"] = {
        f"b{b}": ci([p["ours_info"][b] - p["practitioner"][b] for p in per])
        for b in BUDGETS_PCT
    }
    p = Path(__file__).resolve().parent / "exploratory_paired_arms.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["paired_vs_incumbent_prior"], indent=1))
