#!/usr/bin/env python3
"""Rebaseline every run against the best network found for its (b, domain):
D_ratio = D_norm / min D_norm over all tree runs with the same b, n_sinks, domain_seed, kind."""
import glob, os, sys, pandas as pd
import argparse; _p = argparse.ArgumentParser(); _p.add_argument("--out", default="results"); out = _p.parse_args().out
frames = [pd.read_csv(f, sep="\t") for f in glob.glob(os.path.join(out, "*.tsv")) if os.path.basename(f) != "summary.tsv"]
df = pd.concat(frames, ignore_index=True)
key = ["b", "n_sinks", "domain_seed", "kind"]
best = df[df.get("beta", 0) == 0].groupby(key)["D_norm"].min().rename("D_best").reset_index()
df = df.merge(best, on=key, how="left"); df["D_ratio"] = df["D_norm"] / df["D_best"]
df.to_csv(os.path.join(out, "summary.tsv"), sep="\t", index=False)
cols = [c for c in ["exp", "b", "n_sinks", "domain_seed", "seed", "sigma", "stages", "growth_mode", "edges", "beta", "D_ratio", "converged", "steps", "seconds"] if c in df]
print(df[cols].sort_values(["b", "exp"]).to_string(index=False))
