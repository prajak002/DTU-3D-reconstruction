"""Merge benchmark rows from one run CSV into another, by the benchmark key (model, scan, seed, radius, fill).

    python scripts/merge_rows.py SRC.csv DST.csv [--only-models a,b] [--drop-models a,b]

Rows of DST whose model is in --drop-models are removed first (used to discard rows produced by an incomplete head).
Rows of SRC (restricted to --only-models when given) whose key is not in DST are appended. DST_summary.csv is rebuilt
with hallu3d.summarize. Used on 2026-11-06 to adopt the Kaggle Pi3X head rows (kernel hallu3d-p3-pi3x) into the box
run CSVs so that the resumable stages skip them.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hallu3d as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--only-models", default="")
    ap.add_argument("--drop-models", default="")
    a = ap.parse_args()
    src = pd.read_csv(a.src)
    if a.only_models:
        src = src[src["model"].isin(a.only_models.split(","))]
    if os.path.exists(a.dst):
        dst = pd.read_csv(a.dst)
        n0 = len(dst)
        if a.drop_models:
            dst = dst[~dst["model"].isin(a.drop_models.split(","))]
        print(f"{a.dst}: {n0} rows, {n0 - len(dst)} dropped")
    else:
        dst = src.iloc[0:0].copy()
        print(f"{a.dst}: new")
    have = {H._key(r) for r in dst.to_dict("records")}
    add = src[[H._key(r) not in have for r in src.to_dict("records")]]
    out = pd.concat([dst, add], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    out.to_csv(a.dst, index=False)
    print(f"{a.dst}: +{len(add)} of {len(src)} source rows -> {len(out)} rows; models {sorted(out['model'].unique())}")
    H.summarize(a.dst).to_csv(a.dst.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
