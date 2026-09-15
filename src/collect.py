#!/usr/bin/env python3
"""Rebaseline every run against the best network found for its (b, domain):
D_ratio = D_norm / min D_norm over all tree runs with the same b, n_sinks, domain_seed, kind."""
import glob, os, sys, pandas as pd
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
frames = [pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(out, "*.tsv")) if os.path.basename(f) != "summary.tsv"]
df = pd.concat(frames, ignore_index=True)
# run_experiment.py appends, so re-running the campaign leaves duplicate rows for
# the same configuration. Keep the most recent of each; `seconds` differs between
# repeats but nothing else does, so this cannot hide a genuine disagreement.
ident = [c for c in ["exp", "b", "n_sinks", "domain_seed", "kind", "seed", "sigma",
                     "stages", "growth_mode", "mc_samples", "clip", "kappa", "reactivate"] if c in df]
before = len(df)
df = df.drop_duplicates(subset=ident, keep="last").reset_index(drop=True)
if before != len(df):
    print(f"dropped {before - len(df)} duplicate rows from repeated runs")
key = ["b", "n_sinks", "domain_seed", "kind"]
best = df[df.get("beta", 0) == 0].groupby(key)["D_norm"].min().rename("D_best").reset_index()
df = df.merge(best, on=key, how="left"); df["D_ratio"] = df["D_norm"] / df["D_best"]
df.to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False)
cols = [c for c in ["exp", "b", "n_sinks", "domain_seed", "seed", "sigma", "stages", "growth_mode", "edges", "beta", "D_ratio", "converged", "steps", "seconds"] if c in df]
print(df[cols].sort_values(["b", "exp"]).to_string(index=False))
