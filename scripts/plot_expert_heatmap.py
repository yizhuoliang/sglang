#!/usr/bin/env python3
import argparse
import os
import sys
import glob

import duckdb  # pip install duckdb
import pandas as pd
import seaborn as sns  # pip install seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate expert routing CSV logs into a layer x expert heatmap"
    )
    p.add_argument(
        "--input-glob",
        type=str,
        default="expert_routing_*.csv",
        help="Glob for input CSVs (relative to current working directory)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="expert_heatmap.png",
        help="Output image path (PNG)",
    )
    p.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "layer", "global"],
        help="Normalization for heat values: none (counts), layer (per-layer share), global (share of total)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="DuckDB execution threads",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cwd = os.getcwd()
    pattern = os.path.join(cwd, args.input_glob)
    files = sorted(glob.glob(pattern))

    # Use DuckDB to scan and aggregate large CSVs efficiently.
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}")

    # Column names can be either:
    # - forward_pass_id, rank, layer, token_index, position, k_slot, expert_logical (new)
    # - forward_pass_id, rank, layer, token_index, position, expert_top1_logical (legacy)
    # We only need layer and expert id for counting.
    # Try new schema first (expert_logical), then legacy (expert_top1_logical)
    query_new = f"""
        SELECT
            layer::INTEGER AS layer,
            TRY_CAST(expert_logical AS INTEGER) AS expert,
            COUNT(*)::BIGINT AS cnt
        FROM read_csv_auto('{pattern}', HEADER=TRUE)
        WHERE expert_logical IS NOT NULL
          AND TRY_CAST(expert_logical AS INTEGER) IS NOT NULL
          AND TRY_CAST(expert_logical AS INTEGER) >= 0
        GROUP BY layer, expert
        ORDER BY layer, expert
    """
    query_legacy = f"""
        SELECT
            layer::INTEGER AS layer,
            TRY_CAST(expert_top1_logical AS INTEGER) AS expert,
            COUNT(*)::BIGINT AS cnt
        FROM read_csv_auto('{pattern}', HEADER=TRUE)
        WHERE expert_top1_logical IS NOT NULL
          AND TRY_CAST(expert_top1_logical AS INTEGER) IS NOT NULL
          AND TRY_CAST(expert_top1_logical AS INTEGER) >= 0
        GROUP BY layer, expert
        ORDER BY layer, expert
    """

    # Handle consolidated wide schema (per_token) by unpivoting expert_logical_k*
    agg_df = None
    if files:
        # Peek header to detect expert_logical_k columns
        with open(files[0], "r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        k_cols = [c for c in header if c.startswith("expert_logical_k")]
        if k_cols:
            # Build UNNEST query of wide columns
            cols_expr = ", ".join(k_cols)
            query_wide = f"""
                WITH base AS (
                    SELECT layer::INTEGER AS layer,
                           LIST_VALUE({cols_expr}) AS experts
                    FROM read_csv_auto('{pattern}', HEADER=TRUE)
                )
                SELECT layer, TRY_CAST(expert AS INTEGER) AS expert, COUNT(*)::BIGINT AS cnt
                FROM (
                    SELECT layer, UNNEST(experts) AS expert FROM base
                )
                WHERE expert IS NOT NULL
                  AND TRY_CAST(expert AS INTEGER) IS NOT NULL
                  AND TRY_CAST(expert AS INTEGER) >= 0
                GROUP BY layer, expert
                ORDER BY layer, expert
            """
            try:
                agg_df = con.execute(query_wide).fetchdf()
            except duckdb.CatalogException:
                agg_df = None

    if agg_df is None:
        try:
            agg_df = con.execute(query_new).fetchdf()
            if agg_df.empty:
                agg_df = con.execute(query_legacy).fetchdf()
        except duckdb.CatalogException:
            agg_df = con.execute(query_legacy).fetchdf()

    if agg_df.empty:
        print("No rows found in input CSVs.")
        return

    # Pivot to matrix: rows = expert, cols = layer
    heat = (
        agg_df.pivot(index="expert", columns="layer", values="cnt")
        .fillna(0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Normalization options
    if args.normalize == "layer":
        col_sums = heat.sum(axis=0)
        # Avoid division by zero
        col_sums[col_sums == 0] = 1
        heat = heat.divide(col_sums, axis=1)
    elif args.normalize == "global":
        total = heat.values.sum()
        if total == 0:
            total = 1
        heat = heat / total

    # Plot
    # Make tiles square by using equal per-cell scaling for width and height
    cell_scale = 0.25
    fig_w = max(8, heat.shape[1] * cell_scale)
    fig_h = max(8, heat.shape[0] * cell_scale)
    plt.figure(figsize=(fig_w, fig_h))
    cmap = "viridis"
    # Draw heatmap without built-in colorbar so we can attach an aligned one
    ax = sns.heatmap(heat, cmap=cmap, square=True, cbar=False)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(ax.collections[0], cax=cax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Expert Index")
    title_suffix = {
        "none": "(counts)",
        "layer": "(per-layer share)",
        "global": "(global share)",
    }[args.normalize]
    ax.set_title(f"Expert Routing Heatmap {title_suffix}")

    plt.tight_layout()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()


