#!/usr/bin/env python3
"""Rebaseline every run against the best network found for its (b, domain):
D_ratio = D_norm / min D_norm over all tree runs with the same b, n_sinks, domain_seed, kind."""
import glob, os, sys, pandas as pd
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
frames = [pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(out, "*.tsv")) if os.path.basename(f) != "summary.tsv"]
if not frames:
    raise SystemExit(f"no per-experiment .tsv files in {out!r}: run src/run_experiment.py "
                     f"or src/run_campaign.py first")
df = pd.concat(frames, ignore_index=True)
# run_experiment.py appends, so re-running the campaign leaves duplicate rows for
# the same configuration. Keep the most recent of each; `seconds` differs between
# repeats but nothing else does, so this cannot hide a genuine disagreement.
ident = [c for c in ["exp", "b", "n_sinks", "domain_seed", "kind", "seed", "sigma",
                     "stages", "growth_mode", "mc_samples", "clip", "kappa", "reactivate", "match_budget"] if c in df]
before = len(df)
df = df.drop_duplicates(subset=ident, keep="last").reset_index(drop=True)
if before != len(df):
    print(f"dropped {before - len(df)} duplicate rows from repeated runs")
key = ["b", "n_sinks", "domain_seed", "kind"]
# Rank each run by the dissipation it actually experienced: the mean demand for the
# steady protocols, tr(L^+ Q) for the fluctuating ones, which is what those runs
# minimise. D_best is taken over loop-free networks only, so the reference is always a
# tree and the two loadings are never mixed inside it.
df["D_eff"] = df["D_norm"]
if "D_norm_load" in df:
    has = df["D_norm_load"].notna()
    df.loc[has, "D_eff"] = df.loc[has, "D_norm_load"]
pool = df[(df.get("beta", 0) == 0) & df["D_norm"].notna()]
best = pool.groupby(key)["D_norm"].min().rename("D_best").reset_index()
df = df.merge(best, on=key, how="left")
df["D_ratio"] = df["D_norm"] / df["D_best"]
# For fluctuating runs the reference stored per row, D_ref_load, is the mean-demand
# topology with its allocation re-optimised under the same loading: a fixed-topology
# reference, not the best tree under fluctuations. D_best is the mean-demand dissipation
# and would compare different loadings. Rows without D_ref_load fall back to D_best.
ref = df["D_best"]
if "D_ref_load" in df:
    ref = df["D_ref_load"].where(df["D_ref_load"].notna(), df["D_best"])
df["D_ratio_load"] = df["D_eff"] / ref
df.to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False)
cols = [c for c in ["exp", "b", "n_sinks", "domain_seed", "seed", "sigma", "stages", "growth_mode", "edges", "beta", "D_ratio", "converged", "steps", "seconds"] if c in df]
print(df[cols].sort_values(["b", "exp"]).to_string(index=False))
