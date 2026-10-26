# vascular-shear-setpoint

Companion code to the manuscript *Contingent identity and conserved statistics in adaptively
optimized vascular networks*. A local shear set-point rule with a power-law maintenance cost
descends the global transport functional but leaves the network architecture degenerate. The
code covers the optimal trees and the remodeling dynamics, replica ensembles that separate
noise from developmental history, tree statistics in a fundus-photo window, a bilateral
(left vs right eye) analysis of retinal vascular networks, and the shear–pressure analysis of
the rat mesenteric network. All code is in `src/`.

| Script | Purpose |
|---|---|
| `run_experiment.py`, `run_campaign.py`, `collect.py`, `make_figures.py` | Optimal trees, remodeling from dense and random starts, staged growth, fluctuating demand |
| `run_checks.py` | Verification suite (C1–C14) |
| `replicas.py` | Model core (shared by all scripts) and window tree statistics |
| `ensembles.py` | Replica ensembles behind Tables 1–2 and Figures 2–3, with deterministic seeds |
| `figures_procb.py` | Figures 1–4 of the manuscript from the outputs of the scripts above |
| `retina_segment.py` | Per-eye vessel database from fundus images, expert disc contours or external vessel masks |
| `retina_bilateral.py` | Identity overlap, network statistics and disc-rooted trees, left vs right eye |
| `mesentery_shear.py` | Shear, radius, length and pressure in the Pries–Secomb mesenteric network |

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
--b 1.0 0.75 0.6667 0.5 0.25 metabolic exponents; 1 is the volume cost, 3/4 and 2/3 are
                             Kleiber and Rubner, and the last two approach the b -> 0 limit
                             in which the budget counts channels rather than weighing them
--threads 1                  BLAS threads inside each process; total load is procs x threads
--clip 0.02                  integration step cap; 0.15 is not converged, see check C14
--domain-seeds 3 4 5         independent domains
--n-sinks 600                sinks per domain
--starts 8                   random-start runs per (b, domain)
--growth-stages 3 6 12 24    peripheral growth
--isotropic-stages 6 12      isotropic growth
--sigmas 0.5 1 1.5 2 3 4     demand fluctuation amplitudes
--max-steps 20000 --tol 1e-6 --fluct-tol 1e-5
--mc-samples 32              Monte Carlo instead of exact covariance (use for n-sinks > 1000)
--skip fluct growth          leave out experiment families
--no-reactivate-control      repeat peripheral growth with regrowth of pruned edges disabled,
                             which separates enlarging the domain from rebuilding lost vessels
--match-budget-control       repeat the fluctuating sweep with tau0 chosen so every run lands
                             on the reference budget, so loop counts are not confounded with
                             how much material was bought
--seed-control               repeat peripheral growth at a second seeding scale, since the
                             seeding of newly admissible edges is a free parameter of the
                             protocol and the answer depends on it
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
sinks, for the three exponents. This comparison is not part of the manuscript's main results.

## Replica ensembles and window statistics

```bash
python3 src/replicas.py replicas --mode dense  --b 0.75 --N 800 --seeds 41-56 --out results/W_dense_b0.75.pkl
python3 src/replicas.py replicas --mode staged --b 0.75 --N 800 --seeds 41-52 --out results/W_staged_b0.75.pkl
python3 src/replicas.py window --trees results/W_dense_b0.75.pkl results/W_staged_b0.75.pkl \
        --b 0.75 --N 800 --json results/model_window.json
```

Replicas share every parameter and differ in the initial-radius noise (`--eta`, default 0.1) and,
unless `--same-substrate SEED` is given, in the random sink positions. `--mode` is `dense`,
`staged` (12 outward stages with regrowth of pruned connections) or `staged_noregrow`.
Replica files are rewritten after each replica, so an interrupted run resumes. `window`
reconstructs the supply tree inside the annulus 0.11–0.30 of the domain radius, which mimics a
disc-centred fundus photograph, and reports the load-distribution exponent τ, the Hack exponent
and the width–load slope.

## Bilateral retinal analysis

The analysis uses PAPILA (Kovalyk et al., Sci. Data 2022, doi:10.1038/s41597-022-01388-1):
`ImagesWithContours/`, `ExpertsSegmentations/Contours/` and the two clinical spreadsheets.
The data are not redistributed here; place them under `data/`, which is ignored by git.

```bash
python3 src/retina_segment.py --images data/ImagesWithContours --contours data/Contours \
        --clinical-od data/patient_data_od.xlsx --clinical-os data/patient_data_os.xlsx \
        --out results/eyes.pkl
python3 src/retina_bilateral.py overlap --eyes results/eyes.pkl --out results/overlap.pkl
python3 src/retina_bilateral.py metrics --eyes results/eyes.pkl
python3 src/retina_bilateral.py trees   --eyes results/eyes.pkl --gap 0.15 --model-json results/model_window.json
```

Only individuals labelled healthy in both eyes are used. The optic disc is the average of the two
experts' disc polygons; the built-in vessel segmentation is a black top-hat with hysteresis
thresholding at 0.35 of full resolution. External masks (for example from AutoMorph) can replace
it with `--vessel-glob` and, for arteries and veins, `--av-glob` (red-dominant pixels are read as
arteries, blue-dominant as veins); files are matched by `RETxxxOD`/`RETxxxOS` in their names.
`trees` reconstructs arterial and venous trees separately when an artery/vein map is present. Run
it with `--gap 0.15` and `--gap 0.30`: exponents that change between the two are dominated by
segmentation gaps. With the built-in segmentation the trees are used only to validate the
pipeline, since fine branches are under-detected.

Reference results (163 healthy individuals, 137 pairs after quality control):

| Quantity | Value |
|---|---|
| Identity overlap, same person vs other people | 0.222 vs 0.213, excess +0.010 [0.004, 0.015], p = 0.002 |
| Same, large vessels | excess +0.014 [0.005, 0.022], p = 0.004 |
| Same eye re-imaged after rotation and noise (ceiling, `ceiling`) | 0.75 (five eyes, range 0.41–0.90) |
| Shared layout on the 0–1 scale (`similarity`) | 0.018 [0.007, 0.028] |
| Branch points per area, left–right Spearman ρ, fully adjusted | +0.42, p = 2 × 10⁻⁷ |

## Mesenteric shear and pressure

```bash
python3 src/mesentery_shear.py
```

Downloads the Pries–Secomb mesenteric network (`network.dat` from github.com/secomb/NetFlowV2)
if it is not present, solves the flows with the in vivo viscosity law and reports, for arterioles
and venules, the radius-only shear fit, the dissipation-to-cost fit and the share of shear
variation explained by intravascular pressure.

## Reproducing the manuscript

```bash
# model ensembles (resumable; --budget limits the seconds per call)
for m in dense staged staged_noregrow; do python3 src/ensembles.py shared --mode $m --b 0.75 --eta 0.1; done
python3 src/ensembles.py shared --mode dense --b 0.75 --eta 0.01
python3 src/ensembles.py shared --mode dense --b 0.75 --eta 0.3
python3 src/ensembles.py shared --mode staged --b 0.75 --eta 0.01
for m in dense staged; do python3 src/ensembles.py shared --mode $m --b 0.5 --eta 0.1; done
for b in 1 0.75 0.5 0.25; do for d in 3 4 5; do python3 src/ensembles.py fate --b $b --domain $d; done; done
for m in dense staged; do for b in 1 0.75 0.5; do python3 src/ensembles.py independent --mode $m --b $b --seeds 21-32; done
  python3 src/ensembles.py independent --mode $m --b 0.75 --seeds 21-26 --graph 21 --tag same; done
python3 src/ensembles.py summary --json results/ensembles_summary.json
python3 src/replicas.py replicas --mode dense  --b 0.75 --N 800 --seeds 41-56 --out results/W_dense_b0.75.pkl
python3 src/replicas.py replicas --mode staged --b 0.75 --N 800 --seeds 41-52 --out results/W_staged_b0.75.pkl
python3 src/replicas.py window --trees results/W_dense_b0.75.pkl results/W_staged_b0.75.pkl --b 0.75 --N 800 --json results/model_window.json
# retina (PAPILA under data/)
python3 src/retina_segment.py --images data/ImagesWithContours --contours data/Contours \
        --clinical-od data/patient_data_od.xlsx --clinical-os data/patient_data_os.xlsx --out results/eyes.pkl
python3 src/retina_bilateral.py overlap --eyes results/eyes.pkl --out results/overlap.pkl
python3 src/retina_bilateral.py metrics --eyes results/eyes.pkl
python3 src/retina_bilateral.py ceiling --eyes results/eyes.pkl --images data/ImagesWithContours
python3 src/retina_bilateral.py similarity --eyes results/eyes.pkl --overlap results/overlap.pkl --ceiling 0.752 --out results/similarity.json
python3 src/retina_bilateral.py trees --eyes results/eyes.pkl --gap 0.15 --model-json results/model_window.json
python3 src/mesentery_shear.py
# figures (summary.tsv is electronic supplementary material, table S1)
python3 src/figures_procb.py --summary-tsv results/summary.tsv --out results/figures
```

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
C11 robustness to kappa, prune cut and tolerance (the step is C14)
C12 robustness to domain seed and candidate-graph type
C13 convergence of the fluctuating-demand runs
C14 the endpoint is converged in the integration step
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
results/W_*.pkl          replica ensembles (replicas.py)
results/model_window.json  window tree statistics of the model
results/eyes.pkl         per-eye vessel database (retina_segment.py)
results/overlap.pkl      identity overlap rows (retina_bilateral.py overlap)
```

## Citation

See `CITATION.cff`. Please also cite the PAPILA dataset when using the retinal analysis.

## License

MIT. See `LICENSE`.
