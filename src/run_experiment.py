#!/usr/bin/env python3
"""One experiment -> one row in results/<exp>.tsv plus an .npz with the network.

  opt     tree search (reweighted SPT, optional edge swaps)
  starts  local adaptation from random radii, steady demand
  dense   local adaptation from the full graph with uniform radii
  growth  peripheral or isotropic growth during adaptation
  fluct   fluctuating demand of relative amplitude sigma, from the dense start
  shear   shear-law check on the best saved tree

Every row stores D_norm = D * C^{1/alpha}, which is budget-invariant, so runs
can be compared afterwards with scripts/collect.py against the best tree found.
"""
import argparse, os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain import make_domain
from flows import Net, shear_stats, tau0_from_tree
from treesearch import reweighted_spt, edge_swap
from remodel import adapt, grow_peripheral, grow_isotropic, summarize

p = argparse.ArgumentParser()
p.add_argument("exp", choices=["opt", "starts", "dense", "growth", "fluct", "shear"])
p.add_argument("--b", type=float, required=True)
p.add_argument("--n-sinks", type=int, default=600)
p.add_argument("--domain-seed", type=int, default=3)
p.add_argument("--kind", default="delaunay", choices=["delaunay", "knn"])
p.add_argument("--seed", type=int, default=0, help="run seed (starts, fluct MC)")
p.add_argument("--kappa", type=float, default=0.2)
p.add_argument("--clip", type=float, default=0.02,
               help="cap on |d ln r| per Euler step; 0.15 is NOT converged, see run_checks C11")
p.add_argument("--max-steps", type=int, default=20000)
p.add_argument("--tol", type=float, default=1e-6)
p.add_argument("--prune", type=float, default=1e-3)
p.add_argument("--stages", type=int, default=6)
p.add_argument("--growth-mode", default="peripheral", choices=["peripheral", "isotropic"])
p.add_argument("--reactivate", dest="reactivate", action="store_true", default=True,
               help="growth may reseed edges pruned in an earlier shell (default)")
p.add_argument("--no-reactivate", dest="reactivate", action="store_false",
               help="growth seeds only edges never active: isolates domain growth from regrowth")
p.add_argument("--sigma", type=float, default=0.0)
p.add_argument("--mc-samples", type=int, default=0, help=">0: Monte Carlo instead of exact covariance")
p.add_argument("--swap-sweeps", type=int, default=0)
p.add_argument("--out", default="results", help="output directory")
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
tag = f"b{a.b:.4f}_n{a.n_sinks}_d{a.domain_seed}_{a.kind}"
# The opt reference is cached under `tag`; swap_sweeps is deliberately not in the
# name so that one reference serves a whole campaign, but that means two campaigns
# with different --swap-sweeps must not share an --out directory.
pts, E, L = make_domain(a.n_sinks, a.domain_seed, a.kind); net = Net(pts, E, L)
t0 = time.time()

def tau0_ref():
    """set-point constant from the reference tree (computed and cached per tag)."""
    f = os.path.join(a.out, f"opt_{tag}.json")
    if not os.path.exists(f):
        sol = reweighted_spt(net, a.b); t, sp = tau0_from_tree(sol, a.b)
        json.dump(dict(tau0=t, spread=sp, D_norm=sol["D_norm"], edges=sol["edges"]), open(f, "w"))
    return json.load(open(f))

def emit(row, r=None, active=None):
    row.update(exp=a.exp, b=a.b, n_sinks=a.n_sinks, domain_seed=a.domain_seed, kind=a.kind,
               seed=a.seed, kappa=a.kappa, clip=a.clip, tol=a.tol, prune=a.prune, seconds=round(time.time() - t0, 1))
    f = os.path.join(a.out, f"{a.exp}.tsv")
    pd.DataFrame([row]).to_csv(f, sep="\t", index=False, mode="a", header=not os.path.exists(f))
    if r is not None:
        np.savez_compressed(os.path.join(a.out, f"{a.exp}_{tag}_s{a.seed}_sig{a.sigma}_K{a.stages}{a.growth_mode[0]}"
                            f"{'' if a.reactivate else '_noreact'}.npz"),
                            r=r, active=active, edges=np.array(net.E))
    print(json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in row.items()}))

if a.exp == "opt":
    sol = reweighted_spt(net, a.b, seed=a.seed)
    if a.swap_sweeps: sol = edge_swap(net, sol["edges"], a.b, sweeps=a.swap_sweeps)
    t, sp = tau0_from_tree(sol, a.b)
    json.dump(dict(tau0=t, spread=sp, D_norm=sol["D_norm"], edges=sol["edges"]), open(os.path.join(a.out, f"opt_{tag}.json"), "w"))
    emit(dict(D_norm=sol["D_norm"], edges=len(sol["edges"]), beta=0, tau0=t, tau0_spread=sp), sol["r_full"], sol["active"])

elif a.exp in ("starts", "dense", "fluct"):
    ref = tau0_ref(); rng = np.random.default_rng(a.seed)
    rbar = float(net.allocate([tuple(e) for e in ref["edges"]], a.b)["r"].mean())   # radius scale of the reference tree
    r0 = rbar * np.exp(0.5 * rng.standard_normal(net.m)) if a.exp == "starts" else np.full(net.m, rbar)
    hist = []
    r, active, steps, err = adapt(net, a.b, r0, ref["tau0"], sigma=a.sigma if a.exp == "fluct" else 0.0,
                                  kappa=a.kappa, clip=a.clip, max_steps=a.max_steps, tol=a.tol, prune=a.prune,
                                  mc_samples=a.mc_samples, seed=a.seed, history=hist)
    S = summarize(net, r, active, a.b, sigma=a.sigma if a.exp == "fluct" else 0.0)
    S.update(steps=steps, final_err=err, converged=bool(err < a.tol), sigma=a.sigma, mc_samples=a.mc_samples,
             jaccard_vs_opt=len({net.E[k] for k in np.flatnonzero(active)} & {tuple(e) for e in ref["edges"]}) /
                            len({net.E[k] for k in np.flatnonzero(active)} | {tuple(e) for e in ref["edges"]}))
    emit(S, r, active)

elif a.exp == "growth":
    ref = tau0_ref(); kw = dict(kappa=a.kappa, clip=a.clip, max_steps=a.max_steps,
                                tol=a.tol, prune=a.prune)
    if a.growth_mode == "peripheral":
        r, active, diag = grow_peripheral(net, a.b, ref["tau0"], a.stages, r_seed=0.05,
                                          reactivate=a.reactivate, **kw)
    else:
        r, active, diag = grow_isotropic(net, a.b, ref["tau0"], a.stages, r_seed=0.05, **kw)
    S = summarize(net, r, active, a.b)
    S.update(stages=a.stages, growth_mode=a.growth_mode, reactivate=a.reactivate,
             steps=int(sum(d["steps"] for d in diag)),
             final_err=max(d["final_err"] for d in diag),
             converged=bool(all(d["converged"] for d in diag)),
             shells_converged=sum(d["converged"] for d in diag),
             jaccard_vs_opt=len({net.E[k] for k in np.flatnonzero(active)} & {tuple(e) for e in ref["edges"]}) /
                            len({net.E[k] for k in np.flatnonzero(active)} | {tuple(e) for e in ref["edges"]}))
    json.dump(diag, open(os.path.join(a.out, f"growth_shells_{tag}_K{a.stages}{a.growth_mode[0]}"
                                      f"{'' if a.reactivate else '_noreact'}.json"), "w"), indent=1)
    emit(S, r, active)

elif a.exp == "shear":
    ref = tau0_ref(); sol = net.allocate([tuple(e) for e in ref["edges"]], a.b)
    emit(shear_stats(sol, a.b))
