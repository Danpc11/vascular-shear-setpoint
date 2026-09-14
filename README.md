# vascular-shear-setpoint

Dissipation-optimal vascular trees and local shear set-point remodeling with a
metabolic cost exponent. Companion code to the manuscript; all code is in `src/`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check

```bash
python3 src/run_experiment.py opt    --b 0.75 --n-sinks 120 --domain-seed 1 --out /tmp/smoke
python3 src/run_experiment.py growth --b 0.75 --n-sinks 120 --domain-seed 1 --stages 4 --out /tmp/smoke
python3 src/collect.py --out /tmp/smoke
```

## Full campaign

```bash
python3 src/run_campaign.py --procs 16 --out results
python3 src/make_figures.py --out results
```

`run_campaign.py` runs every experiment as an independent subprocess with
`--procs` workers. Every setting is a flag; `--help` lists them. Useful ones:

```
--b 1.0 0.75 0.6667          metabolic exponents
--domain-seeds 3 4 5         independent domains
--n-sinks 600                sinks per domain
--starts 8                   random-start runs per (b, domain)
--growth-stages 3 6 12 24    peripheral growth
--isotropic-stages 6 12      isotropic growth
--sigmas 0.5 1 1.5 2 3 4     demand fluctuation amplitudes
--max-steps 20000 --tol 1e-6 --fluct-tol 1e-5
--mc-samples 32              Monte Carlo instead of exact covariance (use for n-sinks > 1000)
--skip fluct growth          leave out experiment families
--dry-run                    print the job list
```

The `opt` jobs run first (they write the τ₀ reference the other jobs read); the
rest run in parallel afterwards. `collect.py` is called at the end and writes
`results/summary.tsv`, where `D_ratio` is each network's dissipation relative to
the best loop-free network found for the same exponent and domain.

## Single experiments

```bash
python3 src/run_experiment.py <exp> --b B --domain-seed D [--n-sinks N] [--seed S] [--out DIR] ...
```

`<exp>` is one of `opt`, `starts`, `dense`, `growth`, `fluct`, `shear`. Each run
appends one row to `DIR/<exp>.tsv` and saves the network as `.npz`. Run `opt` for
a given `(B, D, N)` before the others.

## Lattice comparison

```bash
cd src && python3 lattice_treesearch.py
```

A hierarchical H-tree against the best tree found on a 7×7 rectilinear lattice with 16
sinks, for the three exponents. This comparison supports the companion manuscript rather
than the Letter.

## Verification suite

```bash
python3 src/run_checks.py --out results          # every check
python3 src/run_checks.py --list                 # what each one does
python3 src/run_checks.py --only C3 C4 C7        # a subset
python3 src/run_campaign.py --procs 16 --checks  # campaign then checks
```

Writes `results/checks.tsv` with one row per check: the measured quantity, the value the
theory requires, the deviation, the tolerance and a pass flag. Exit status is 1 if any
check fails, so it can gate a release.

```
C1  gradient of the dissipation against central differences
C2  the closed-form tree allocation is the constrained minimum on that tree
C3  set-point constancy on an optimal tree
C4  the invariant D/c and D/m^b are uniform
C5  Murray recovery at b = 1; non-uniform shear for b < 1
C6  joint regression recovers both exponents; the r-only fit is biased
C7  the Lyapunov identity reproduces the measured dF/dt
C8  D_norm is invariant under rescaling all radii
C9  optimal and adapted trees are both rest points yet differ in dissipation
C10 exact certificate: all spanning trees of a small domain enumerated
C11 robustness to kappa, prune cut, clip and tolerance
C12 robustness to domain seed and candidate-graph type
C13 convergence of the fluctuating-demand runs
```

C10 and C11 report rather than assert. C10 measures how far the heuristic and the
dynamics sit from a certified optimum, which is the only place the "best candidate"
language can be bounded. C11 measures how much the endpoint of the descent moves with
the integration parameters; that spread is the systematic uncertainty on every
excess-dissipation number.

## Outputs

```
results/<exp>.tsv        one row per run
results/*.npz            radii and active edges of each network
results/summary.tsv      all runs, rebaselined (collect.py)
results/checks.tsv       one row per verification (run_checks.py)
results/figures/         excess_dissipation.png, loops_vs_sigma.png, growth_stages.png
```
