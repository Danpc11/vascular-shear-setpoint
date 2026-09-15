#!/usr/bin/env python3
"""Print the one table that decides whether domain growth helps.

    python3 read_growth.py results_growth

Rows are the number of growth shells, columns the domain size. Each cell is the
excess dissipation over the best loop-free network found, averaged over domains.
The question is whether "growth, no regrowth" sits below "dense" everywhere: if
it does, enlarging the domain helps on its own; if the sign moves with size or
with the number of shells, the effect is regrowth of pruned vessels.
"""
import sys
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "results_growth"
d = pd.read_csv(f"{out}/summary.tsv", sep="\t")
d = d[d.get("converged", True).fillna(True).astype(bool)]
pct = lambda s: 100 * (s - 1)

print("\ndense start, fixed domain (the baseline):")
print(pct(d[d.exp == "dense"].groupby("n_sinks").D_ratio.mean()).round(1).to_string())

g = d[(d.exp == "growth") & (d.growth_mode == "peripheral")]
for react, lab in ((True, "growth, regrowth ON"), (False, "growth, regrowth OFF")):
    h = g[g.reactivate == react]
    if "r_seed" in h:
        h = h[h.r_seed == h.r_seed.min()]
    if not len(h):
        continue
    print(f"\n{lab}  (excess %, rows = shells, cols = sinks)")
    print(pct(h.pivot_table(index="stages", columns="n_sinks", values="D_ratio",
                            aggfunc="mean")).round(1).to_string())

s = g[g.get("r_seed", 0) == g.get("r_seed", pd.Series([0])).max()]
if len(s) and s.r_seed.iloc[0] != g.r_seed.min():
    print(f"\nseeding control, r_seed = {s.r_seed.iloc[0]}")
    print(pct(s.pivot_table(index="stages", columns="n_sinks", values="D_ratio",
                            aggfunc="mean")).round(1).to_string())

iso = d[(d.exp == "growth") & (d.growth_mode == "isotropic")]
if len(iso):
    print("\nisotropic growth (the negative control):")
    print(pct(iso.groupby("n_sinks").D_ratio.mean()).round(1).to_string())

print("\nverdict: compare 'regrowth OFF' against 'dense' cell by cell.")
