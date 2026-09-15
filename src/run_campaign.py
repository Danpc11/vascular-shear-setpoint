#!/usr/bin/env python3
"""Run the full campaign as independent subprocesses with a fixed number of workers.

Thread control. Each worker is a separate Python process, and NumPy/SciPy link
against a threaded BLAS that by default opens one thread per core *inside every
process*. With `--procs 24` on a 24-core machine that is 24 x 24 threads, which
oversubscribes the machine and runs slower than one thread per process. This
script therefore exports OMP_NUM_THREADS and its siblings to `--threads`
(default 1) in the child environment, so `--procs N --threads T` uses exactly
N x T threads. The work here is many small independent solves, so T = 1 is
almost always the right choice; raise it only if you lower --procs by the same
factor for a few very large domains.

The `opt` jobs run first because they write the tau0 reference every other job
reads; the rest run in parallel afterwards.
"""
import argparse
import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--procs", type=int, default=os.cpu_count() or 4,
               help="worker processes run at once")
p.add_argument("--threads", type=int, default=1,
               help="BLAS/OpenMP threads inside each process; total load is procs x threads")
p.add_argument("--b", type=float, nargs="+", default=[1.0, 0.75, 2 / 3, 0.5, 0.25],
               help="metabolic exponents. 1 is the volume cost, 3/4 and 2/3 are Kleiber and\n"
                    "Rubner, and 1/2 and 1/4 continue the sequence toward the b -> 0 limit in\n"
                    "which the cost stops weighing conductance and only counts channels")
p.add_argument("--domain-seeds", type=int, nargs="+", default=[3, 4, 5])
p.add_argument("--n-sinks", type=int, nargs="+", default=[600],
               help="domain sizes. The excess dissipation of every protocol grows with size\n"
                    "and is not converged at 600, and the sign of the domain-growth effect\n"
                    "changes between 120 and 300, so a single size is not a result: give a\n"
                    "list and report the trend")
p.add_argument("--kind", default="delaunay", choices=["delaunay", "knn"])
p.add_argument("--starts", type=int, default=8, help="random-start runs per (b, domain)")
p.add_argument("--growth-stages", type=int, nargs="+", default=[3, 6, 12, 24])
p.add_argument("--isotropic-stages", type=int, nargs="+", default=[6, 12])
p.add_argument("--seed-control", action="store_true",
               help="also run peripheral growth at a second seeding scale, the control for the\n"
                    "free seeding parameter of the growth protocol")
p.add_argument("--match-budget-control", action="store_true",
               help="also run the fluctuating sweep with tau0 bisected to a common budget")
p.add_argument("--no-reactivate-control", action="store_true",
               help="also run peripheral growth with reactivation disabled, the control that\n"
                    "separates domain growth from regrowth of pruned vessels")
p.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
p.add_argument("--max-steps", type=int, default=20000)
p.add_argument("--tol", type=float, default=1e-6)
p.add_argument("--fluct-tol", type=float, default=1e-5)
p.add_argument("--mc-samples", type=int, default=0,
               help=">0 uses Monte Carlo for the fluctuating runs; use above ~1000 sinks")
p.add_argument("--swap-sweeps", type=int, default=2)
p.add_argument("--kappa", type=float, default=0.2)
p.add_argument("--clip", type=float, default=0.02,
               help="cap on |d ln r| per Euler step; the endpoint is NOT converged at 0.15")
p.add_argument("--out", default="results")
p.add_argument("--checks", action="store_true", help="also run the verification suite at the end")
p.add_argument("--skip", nargs="*", default=[],
               choices=["opt", "starts", "dense", "growth", "fluct", "shear"])
p.add_argument("--dry-run", action="store_true", help="print the job list and exit")
a = p.parse_args()

# ---- child environment: one BLAS thread per process unless asked otherwise
CHILD_ENV = dict(os.environ)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    CHILD_ENV[var] = str(a.threads)


def job(exp, b, d, N, *extra):
    return [sys.executable, os.path.join(HERE, "run_experiment.py"), exp,
            "--b", f"{b:.6f}", "--domain-seed", str(d),
            "--n-sinks", str(N), "--kind", a.kind,
            "--kappa", str(a.kappa), "--clip", str(a.clip), "--out", a.out, *map(str, extra)]


first, rest = [], []
for b, d, N in itertools.product(a.b, a.domain_seeds, a.n_sinks):
    if "opt" not in a.skip:
        first.append(job("opt", b, d, N, "--swap-sweeps", a.swap_sweeps))
    if "starts" not in a.skip:
        rest += [job("starts", b, d, N, "--seed", s, "--max-steps", a.max_steps, "--tol", a.tol)
                 for s in range(a.starts)]
    if "dense" not in a.skip:
        rest.append(job("dense", b, d, N, "--max-steps", a.max_steps, "--tol", a.tol))
    if "growth" not in a.skip:
        rest += [job("growth", b, d, N, "--stages", K, "--growth-mode", "peripheral",
                     "--max-steps", a.max_steps, "--tol", a.tol) for K in a.growth_stages]
        rest += [job("growth", b, d, N, "--stages", K, "--growth-mode", "isotropic",
                     "--max-steps", a.max_steps, "--tol", a.tol) for K in a.isotropic_stages]
        if a.seed_control:
            rest += [job("growth", b, d, N, "--stages", K, "--growth-mode", "peripheral",
                         "--r-seed", 0.25, "--max-steps", a.max_steps, "--tol", a.tol)
                     for K in a.growth_stages]
        if a.no_reactivate_control:
            rest += [job("growth", b, d, N, "--stages", K, "--growth-mode", "peripheral",
                         "--no-reactivate", "--max-steps", a.max_steps, "--tol", a.tol)
                     for K in a.growth_stages]
    if "fluct" not in a.skip:
        rest += [job("fluct", b, d, N, "--sigma", sg, "--max-steps", a.max_steps,
                     "--tol", a.fluct_tol, "--mc-samples", a.mc_samples) for sg in a.sigmas]
        if a.match_budget_control:
            rest += [job("fluct", b, d, N, "--sigma", sg, "--match-budget", "--max-steps", a.max_steps,
                         "--tol", a.fluct_tol, "--mc-samples", a.mc_samples) for sg in a.sigmas]
    if "shear" not in a.skip:
        rest.append(job("shear", b, d, N))

if a.dry_run:
    for j in first + rest:
        print(" ".join(j[1:]))
    print(f"{len(first) + len(rest)} jobs; {a.procs} processes x {a.threads} threads "
          f"= {a.procs * a.threads} threads at peak")
    sys.exit()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, env=CHILD_ENV)
    tag = " ".join(cmd[2:8])
    print(("ok   " if r.returncode == 0 else "FAIL ") + tag, flush=True)
    if r.returncode:
        print(r.stderr[-800:], file=sys.stderr)
    return r.returncode


os.makedirs(a.out, exist_ok=True)
print(f"{len(first) + len(rest)} jobs; {a.procs} processes x {a.threads} BLAS threads "
      f"= {a.procs * a.threads} threads at peak\n", flush=True)
# ThreadPoolExecutor only waits on subprocess.run; the concurrency is in the children.
with ThreadPoolExecutor(a.procs) as ex:
    fails = sum(ex.map(run, first))
with ThreadPoolExecutor(a.procs) as ex:
    fails += sum(ex.map(run, rest))
print(f"done: {len(first) + len(rest) - fails} ok, {fails} failed")

rc = subprocess.run([sys.executable, os.path.join(HERE, "collect.py"), "--out", a.out],
                    env=CHILD_ENV).returncode
if rc:
    print("collect.py failed", file=sys.stderr)
if a.checks:
    # the verification suite must run at the same integration step as the campaign,
    # or C9 and C11 would test a different discretisation
    rc_checks = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_checks.py"), "--out", a.out,
         "--b", *[f"{x:.6f}" for x in a.b], "--n-sinks", str(a.n_sinks[0]),
         "--domain-seed", str(a.domain_seeds[0]), "--kind", a.kind,
         "--clip", str(a.clip)], env=CHILD_ENV).returncode
    if rc_checks:
        print("run_checks.py reported failures", file=sys.stderr)
    rc = rc or rc_checks
# a non-zero exit lets a scheduler or a CI step notice that the campaign is not clean
sys.exit(1 if (fails or rc) else 0)
