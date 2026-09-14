#!/usr/bin/env python3
"""Figures from results/summary.tsv (run scripts/collect.py first)."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
df = pd.read_csv(os.path.join(out, "summary.tsv"), sep="\t"); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = ["#2F4365", "#915C3E", "#578E6E", "#BE86AC"]
# 1. excess dissipation by protocol
fig, ax = plt.subplots(figsize=(6.5, 3))
labels = {"starts": "local adaptation, random start", "dense": "local adaptation, dense start",
          "opt": "tree search heuristic", "growth": "growth + adaptation"}
for k, (exp, lab) in enumerate(labels.items()):
    g = df[(df.exp == exp) & (df.beta == 0)]
    if exp == "growth": g = g[g.growth_mode == "peripheral"]
    st = g.groupby("b")["D_ratio"].agg(["mean", "std", "count"]).reset_index()
    ax.errorbar(st.b + 0.006 * (k - 1.5), st["mean"] - 1, yerr=st["std"].fillna(0), fmt="o", color=C[k], ms=4, capsize=2, label=lab)
ax.axhline(0, color="0.6", lw=0.6); ax.set(xlabel="metabolic exponent b", ylabel=r"$D/D_{\rm best}-1$")
ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "excess_dissipation.png"), dpi=250)
# 2. loops under fluctuating demand
fig, ax = plt.subplots(figsize=(4.2, 3)); g = df[df.exp == "fluct"]
for k, b in enumerate(sorted(g.b.unique())):
    h = g[g.b == b].groupby("sigma")["beta"].agg(["mean", "std"]).reset_index()
    ax.errorbar(h.sigma, h["mean"], yerr=h["std"].fillna(0), fmt="o-", color=C[k], ms=4, label=rf"$b={b:g}$")
ax.set(xlabel=r"demand fluctuation $\sigma$", ylabel=r"surviving loops $\beta$"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "loops_vs_sigma.png"), dpi=250)
# 3. growth stages
fig, ax = plt.subplots(figsize=(4.2, 3)); g = df[df.exp == "growth"]
for k, b in enumerate(sorted(g.b.unique())):
    for mode, ls in (("peripheral", "-"), ("isotropic", "--")):
        h = g[(g.b == b) & (g.growth_mode == mode)].groupby("stages")["D_ratio"].mean().reset_index()
        if len(h): ax.plot(h.stages, h.D_ratio - 1, ls, marker="o", color=C[k], ms=4, label=rf"$b={b:g}$, {mode}")
ax.axhline(0, color="0.6", lw=0.6); ax.set(xscale="log", xlabel="growth stages", ylabel=r"$D/D_{\rm best}-1$"); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "growth_stages.png"), dpi=250)
print("figures in", os.path.join(out, "figures"))
