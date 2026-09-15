#!/usr/bin/env python3
"""Figures from results/summary.tsv (run scripts/collect.py first)."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
df = pd.read_csv(os.path.join(out, "summary.tsv"), sep="\t"); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = ["#2F4365", "#915C3E", "#578E6E", "#BE86AC"]
# 1. excess dissipation by protocol. The controls are separate series: growth with and
#    without reactivation answer different questions, and averaging them hides the fact
#    that most of the growth benefit is regrowth of pruned vessels.
fig, ax = plt.subplots(figsize=(6.5, 3))
series = [("starts", dict(), "local rule, random start", C[1], "o"),
          ("dense", dict(), "local rule, dense start", C[3], "s"),
          ("opt", dict(), "tree search", C[2], "D"),
          ("growth", dict(growth_mode="peripheral", reactivate=True),
           "growth, reactivation on", C[0], "^"),
          ("growth", dict(growth_mode="peripheral", reactivate=False),
           "growth, reactivation off", C[0], "v")]
for k, (exp, sel, lab, col, mk) in enumerate(series):
    g = df[(df.exp == exp) & (df.get("beta", 0) == 0)]
    for key, val in sel.items():
        if key in g:
            g = g[g[key] == val]
    if not len(g):
        continue
    st = g.groupby("b")["D_ratio"].agg(["mean", "std"]).reset_index()
    ax.errorbar(st.b + 0.004 * (k - 2), 100 * (st["mean"] - 1), yerr=100 * st["std"].fillna(0),
                fmt=mk, color=col, ms=4, capsize=2, mfc="white" if mk == "v" else col,
                label=lab)
ax.axhline(0, color="0.6", lw=0.6)
ax.set(xlabel="metabolic exponent b", ylabel=r"excess dissipation (%)")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "excess_dissipation.png"), dpi=250)

# 2. growth against the number of shells, the two reactivation protocols apart
fig, ax = plt.subplots(figsize=(4.6, 3))
g = df[(df.exp == "growth") & (df.growth_mode == "peripheral")]
for k, b in enumerate(sorted(g.b.unique())):
    for react, ls, mk in ((True, "-", "o"), (False, "--", "v")):
        h = g[(g.b == b) & (g.get("reactivate", True) == react)]
        if not len(h):
            continue
        s = h.groupby("stages")["D_ratio"].agg(["mean", "std"]).reset_index()
        ax.errorbar(s.stages, 100 * (s["mean"] - 1), yerr=100 * s["std"].fillna(0),
                    fmt=mk + ls, color=C[k], ms=4, mfc="white" if not react else C[k],
                    capsize=2, lw=1.1,
                    label=rf"$b={b:g}$" + ("" if react else ", no regrowth"))
ax.set(xscale="log", xlabel="growth shells", ylabel=r"excess dissipation (%)")
ax.set_xticks([3, 6, 12, 24]); ax.set_xticklabels(["3", "6", "12", "24"]); ax.minorticks_off()
ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "growth_stages.png"), dpi=250)

# 3. loops under fluctuating demand, fixed tau0 and matched budget apart
fig, ax = plt.subplots(figsize=(4.6, 3))
g = df[df.exp == "fluct"]
for k, b in enumerate(sorted(g.b.unique())):
    for mb, ls, mk in ((False, "-", "o"), (True, "--", "v")):
        h = g[(g.b == b) & (g.get("match_budget", False) == mb)]
        if not len(h):
            continue
        s = h.groupby("sigma")["beta"].agg(["mean", "std"]).reset_index()
        ax.errorbar(s.sigma, s["mean"], yerr=s["std"].fillna(0), fmt=mk + ls, color=C[k],
                    ms=4, mfc="white" if mb else C[k], capsize=2, lw=1.1,
                    label=rf"$b={b:g}$" + (", matched budget" if mb else ""))
ax.set(xlabel=r"demand fluctuation $\sigma$", ylabel=r"surviving loops $\beta$")
ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "loops_vs_sigma.png"), dpi=250)
print("figures in", os.path.join(out, "figures"))
