#!/usr/bin/env python3
"""Figures from results/summary.tsv (run scripts/collect.py first)."""
import os, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
df = pd.read_csv(os.path.join(out, "summary.tsv"), sep="\t"); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def palette(n):
    """A colour per exponent, for any number of them.

    The campaign default is five exponents, so a fixed four-colour list would
    raise IndexError on the fifth. Sampling a blue-to-red ramp also keeps the
    ordering in b legible.
    """
    base = ["#1B3A5F", "#3E6C96", "#7E8CA3", "#A9705F", "#B3312C"]
    if n <= len(base):
        step = (len(base) - 1) / max(n - 1, 1)
        return [base[int(round(i * step))] for i in range(n)]
    cm = plt.get_cmap("coolwarm")
    return [cm(i / (n - 1)) for i in range(n)]


def usable(g, label=""):
    """Drop runs that did not converge, and say how many were dropped."""
    if "converged" not in g:
        return g, 0
    ok = g[g["converged"].fillna(True).astype(bool)]
    lost = len(g) - len(ok)
    if lost:
        print(f"  dropped {lost} non-converged run(s) from {label or 'a group'}")
    return ok, lost


def blabel(b):
    """Readable exponent labels: 2/3 must not print as 0.666667."""
    for num, den, s in ((2, 3, "2/3"), (3, 4, "3/4"), (1, 2, "1/2"), (1, 4, "1/4"), (1, 1, "1")):
        if abs(b - num / den) < 1e-6:
            return rf"$b={s}$"
    return rf"$b={b:g}$"


C = palette(5)
# 1. excess dissipation by protocol. The controls are separate series: growth with and
#    without reactivation answer different questions, and averaging them hides the fact
#    that most of the growth benefit is regrowth of pruned vessels.
fig, ax = plt.subplots(figsize=(6.5, 3))
# Growth depends strongly on the number of shells, so averaging shell counts into one
# point would mix protocols. The panel shows the deepest growth available, stated in the
# label; the shell dependence itself is the second figure.
gmax = df[df.exp == "growth"]["stages"].max() if "stages" in df else None
series = [("starts", dict(), "local rule, random start", "o"),
          ("dense", dict(), "local rule, dense start", "s"),
          ("opt", dict(), "tree search", "D"),
          ("growth", dict(growth_mode="peripheral", reactivate=True, stages=gmax),
           f"growth, {int(gmax)} shells, regrowth on" if gmax else "growth", "^"),
          ("growth", dict(growth_mode="peripheral", reactivate=False, stages=gmax),
           f"growth, {int(gmax)} shells, regrowth off" if gmax else "growth, no regrowth", "v")]
# b is already the x axis, so colour distinguishes protocols here, not exponents
pc = ["#B3312C", "#A9705F", "#1B3A5F", "#3E6C96", "#3E6C96"]
fig, ax = plt.subplots(figsize=(6.6, 3.1))
for k, (exp, sel, lab, mk) in enumerate(series):
    g = df[(df.exp == exp) & (df.get("beta", 0) == 0)]
    for key, val in sel.items():
        if key in g and val is not None:
            g = g[g[key] == val]
    g, _ = usable(g, lab)
    if not len(g):
        continue
    st = g.groupby("b")["D_ratio"].agg(["mean", "std", "count"]).reset_index()
    ax.errorbar(st.b, 100 * (st["mean"] - 1), yerr=100 * st["std"].fillna(0),
                fmt=mk + "-", color=pc[k], ms=4.5, capsize=2, lw=1.0,
                mfc="white" if mk == "v" else pc[k],
                label=f"{lab}  (n={int(st['count'].min())}–{int(st['count'].max())})")
ax.axhline(0, color="0.6", lw=0.6)
ax.set(xlabel="metabolic exponent $b$", ylabel=r"excess dissipation (%)")
ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2, columnspacing=1.0)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "excess_dissipation.png"), dpi=250)

# 2. growth against the number of shells, the two reactivation protocols apart
fig, ax = plt.subplots(figsize=(4.6, 3))
g = df[(df.exp == "growth") & (df.growth_mode == "peripheral")]
bs = sorted(g.b.unique())
C = palette(len(bs))
for k, b in enumerate(bs):
    for react, ls, mk in ((True, "-", "o"), (False, "--", "v")):
        h = g[(g.b == b) & (g.get("reactivate", True) == react)]
        if not len(h):
            continue
        h, _ = usable(h, f"growth b={b}, reactivate={react}")
        if not len(h):
            continue
        s = h.groupby("stages")["D_ratio"].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(s.stages, 100 * (s["mean"] - 1), yerr=100 * s["std"].fillna(0),
                    fmt=mk + ls, color=C[k], ms=4, mfc="white" if not react else C[k],
                    capsize=2, lw=1.1,
                    label=blabel(b) + ("" if react else ", no regrowth"))
ax.set(xscale="log", xlabel="growth shells", ylabel=r"excess dissipation (%)")
ax.set_xticks([3, 6, 12, 24]); ax.set_xticklabels(["3", "6", "12", "24"]); ax.minorticks_off()
ax.legend(fontsize=5.5, frameon=False, ncol=2, columnspacing=0.8, handlelength=1.4)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "growth_stages.png"), dpi=250)

# 3. loops under fluctuating demand, fixed tau0 and matched budget apart
fig, ax = plt.subplots(figsize=(4.6, 3))
g = df[df.exp == "fluct"]
bs = sorted(g.b.unique())
C = palette(len(bs))
for k, b in enumerate(bs):
    for mb, ls, mk in ((False, "-", "o"), (True, "--", "v")):
        h = g[(g.b == b) & (g.get("match_budget", False) == mb)]
        if not len(h):
            continue
        h, _ = usable(h, f"fluct b={b}, match_budget={mb}")
        if not len(h):
            continue
        s = h.groupby("sigma")["beta"].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(s.sigma, s["mean"], yerr=s["std"].fillna(0), fmt=mk + ls, color=C[k],
                    ms=4, mfc="white" if mb else C[k], capsize=2, lw=1.1,
                    label=blabel(b) + (", matched budget" if mb else ""))
ax.set(xlabel=r"demand fluctuation $\sigma$", ylabel=r"surviving loops $\beta$")
ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(out, "figures", "loops_vs_sigma.png"), dpi=250)
# every point's realisation count, so a curve built from fewer runs is visible
for exp in sorted(df.exp.unique()):
    g = df[df.exp == exp]
    nbad = int((~g.get("converged", True).fillna(True).astype(bool)).sum()) if "converged" in g else 0
    print(f"  {exp:8} {len(g):4d} runs, {nbad} not converged")
print("figures in", os.path.join(out, "figures"))
