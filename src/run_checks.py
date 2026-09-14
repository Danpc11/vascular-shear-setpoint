#!/usr/bin/env python3
"""Run the verification suite and write results/checks.tsv.

    python3 src/run_checks.py --out results                 # everything
    python3 src/run_checks.py --only C3 C4 C7 --n-sinks 200 # a subset, faster
    python3 src/run_checks.py --list                        # what each check does

Exit status is 1 if any check fails, so this can gate a release of the data.
"""
import argparse, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain import make_domain
from flows import Net
import checks as K

DESC = {
 "C1": "gradient of the dissipation against central differences",
 "C2": "the closed-form tree allocation is the constrained minimum",
 "C3": "set-point constancy on an optimal tree",
 "C4": "the invariant D/c and D/m^b are uniform",
 "C5": "Murray recovery at b=1; non-uniform tau for b<1",
 "C6": "joint regression recovers both exponents; r-only fit is biased",
 "C7": "the Lyapunov identity reproduces the measured dF/dt",
 "C8": "D_norm is invariant under rescaling all radii",
 "C9": "optimal and adapted trees are both rest points yet differ",
 "C10": "exact certificate by enumerating all spanning trees (small domain)",
 "C11": "robustness to kappa, prune cut, clip and tolerance",
 "C12": "robustness to domain seed and candidate-graph type",
 "C13": "convergence of the fluctuating-demand runs",
 "C14": "the endpoint is converged in the integration step",
}

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--out", default="results")
p.add_argument("--b", type=float, nargs="+", default=[1.0, 0.75, 2/3])
p.add_argument("--n-sinks", type=int, default=600, help="domain size for C1-C9, C11, C13")
p.add_argument("--domain-seed", type=int, default=3)
p.add_argument("--kind", default="delaunay", choices=["delaunay", "knn"])
p.add_argument("--cert-sinks", type=int, default=7, help="domain size for the C10 enumeration")
p.add_argument("--cert-knn", type=int, default=3)
p.add_argument("--c12-sinks", type=int, default=200)
p.add_argument("--only", nargs="*", default=None, help="subset, e.g. --only C3 C4")
p.add_argument("--list", action="store_true")
a = p.parse_args()
if a.list:
    for k, v in DESC.items(): print(f"  {k:4} {v}")
    sys.exit()

want = set(a.only) if a.only else set(DESC)
os.makedirs(a.out, exist_ok=True)
pts, E, L = make_domain(a.n_sinks, a.domain_seed, a.kind); net = Net(pts, E, L)
print(f"domain: {net.n} nodes, {net.m} candidate edges, kind={a.kind}, seed={a.domain_seed}\n")

rows, t0 = [], time.time()
def add(x):
    for r in (x if isinstance(x, list) else [x]):
        r.setdefault("b", np.nan); rows.append(r)
        flag = "PASS" if r["passed"] else "FAIL"
        dev = r.get("deviation", np.nan)
        print(f"  [{flag}] {r['check']:46} measured={r['measured']:.6g}  dev={dev:.3g}")

for b in a.b:
    print(f"--- b = {b:.4f} ---")
    if "C1"  in want: add(K.c1_gradient(net, b))
    if "C2"  in want: add(K.c2_allocation(net, b))
    if want & {"C3", "C4", "C5"}: add([r for r in K.c3_c4_c5(net, b) if r["check"][:2] in want])
    if "C6"  in want: add(K.c6_regression(net, b))
    if "C7"  in want: add(K.c7_lyapunov(net, b))
    if "C8"  in want: add(K.c8_budget_invariance(net, b))
    if "C9"  in want: add(K.c9_rest_points(net, b))
    if "C10" in want: add(K.c10_certificate(b, a.cert_sinks, kind="knn", knn=a.cert_knn))
    if "C11" in want: add(K.c11_numerics(net, b))
    if "C12" in want: add(K.c12_domains(b, a.c12_sinks))
    if "C13" in want: add(K.c13_fluctuation(net, b))
    if "C14" in want: add(K.c14_step_convergence(net, b))

df = pd.DataFrame(rows)
df["n_sinks"] = a.n_sinks; df["domain_seed"] = a.domain_seed; df["kind"] = a.kind
f = os.path.join(a.out, "checks.tsv"); df.to_csv(f, sep="\t", index=False)
nfail = int((~df.passed).sum())
print(f"\n{len(df)-nfail}/{len(df)} passed in {time.time()-t0:.0f} s -> {f}")
if nfail:
    print("\nFAILED:"); print(df[~df.passed][["check", "b", "measured", "required", "deviation", "tol"]].to_string(index=False))
sys.exit(1 if nfail else 0)
