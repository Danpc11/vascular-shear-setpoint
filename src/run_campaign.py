#!/usr/bin/env python3
"""Run the full campaign as independent subprocesses with a fixed number of workers.

All settings are flags. The `opt` jobs run first because they write the tau0
reference that every other job reads; the remaining jobs run in parallel after.
"""
import argparse, itertools, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--procs", type=int, default=os.cpu_count() or 4, help="parallel workers")
p.add_argument("--b", type=float, nargs="+", default=[1.0, 0.75, 2/3], help="metabolic exponents")
p.add_argument("--domain-seeds", type=int, nargs="+", default=[3, 4, 5])
p.add_argument("--n-sinks", type=int, default=600)
p.add_argument("--kind", default="delaunay", choices=["delaunay", "knn"])
p.add_argument("--starts", type=int, default=8, help="random-start runs per (b, domain)")
p.add_argument("--growth-stages", type=int, nargs="+", default=[3, 6, 12, 24])
p.add_argument("--isotropic-stages", type=int, nargs="+", default=[6, 12])
p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
p.add_argument("--max-steps", type=int, default=20000)
p.add_argument("--tol", type=float, default=1e-6)
p.add_argument("--fluct-tol", type=float, default=1e-5)
p.add_argument("--mc-samples", type=int, default=0, help=">0 uses Monte Carlo for fluct jobs")
p.add_argument("--swap-sweeps", type=int, default=2)
p.add_argument("--kappa", type=float, default=0.2)
p.add_argument("--out", default="results")
p.add_argument("--skip", nargs="*", default=[], choices=["opt", "starts", "dense", "growth", "fluct", "shear"])
p.add_argument("--checks", action="store_true", help="also run the verification suite at the end")
p.add_argument("--dry-run", action="store_true", help="print the job list and exit")
a = p.parse_args()

def job(exp, b, d, *extra):
    return [sys.executable, os.path.join(here, "run_experiment.py"), exp, "--b", f"{b:.6f}", "--domain-seed", str(d),
            "--n-sinks", str(a.n_sinks), "--kind", a.kind, "--kappa", str(a.kappa), "--out", a.out, *map(str, extra)]

first, rest = [], []
for b, d in itertools.product(a.b, a.domain_seeds):
    if "opt" not in a.skip: first.append(job("opt", b, d, "--swap-sweeps", a.swap_sweeps))
    if "starts" not in a.skip:
        rest += [job("starts", b, d, "--seed", s, "--max-steps", a.max_steps, "--tol", a.tol) for s in range(a.starts)]
    if "dense" not in a.skip: rest.append(job("dense", b, d, "--max-steps", a.max_steps, "--tol", a.tol))
    if "growth" not in a.skip:
        rest += [job("growth", b, d, "--stages", K, "--growth-mode", "peripheral", "--max-steps", a.max_steps, "--tol", a.tol) for K in a.growth_stages]
        rest += [job("growth", b, d, "--stages", K, "--growth-mode", "isotropic", "--max-steps", a.max_steps, "--tol", a.tol) for K in a.isotropic_stages]
    if "fluct" not in a.skip:
        rest += [job("fluct", b, d, "--sigma", sg, "--max-steps", a.max_steps, "--tol", a.fluct_tol, "--mc-samples", a.mc_samples) for sg in a.sigmas]
    if "shear" not in a.skip: rest.append(job("shear", b, d))

if a.dry_run:
    for j in first + rest: print(" ".join(j[1:]))
    print(f"{len(first) + len(rest)} jobs"); sys.exit()

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    tag = " ".join(cmd[2:8]); print(("ok   " if r.returncode == 0 else "FAIL ") + tag, flush=True)
    if r.returncode: print(r.stderr[-800:], file=sys.stderr)
    return r.returncode

os.makedirs(a.out, exist_ok=True)
with ThreadPoolExecutor(a.procs) as ex: fails = sum(ex.map(run, first))
with ThreadPoolExecutor(a.procs) as ex: fails += sum(ex.map(run, rest))
print(f"done: {len(first) + len(rest) - fails} ok, {fails} failed")
subprocess.run([sys.executable, os.path.join(here, "collect.py"), "--out", a.out])
if a.checks:
    subprocess.run([sys.executable, os.path.join(here, "run_checks.py"), "--out", a.out,
                    "--b", *[f"{x:.6f}" for x in a.b], "--n-sinks", str(a.n_sinks),
                    "--domain-seed", str(a.domain_seeds[0]), "--kind", a.kind])
