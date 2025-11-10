#!/usr/bin/env python3
import argparse
import csv
import os
from glob import glob
from typing import Iterable, List, Sequence


DEFAULT_GLOB = "expert_routing_*.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge expert_routing per_token CSVs emitted by workers into a single CSV (simple, streaming)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory or glob pattern. If directory, uses --glob within it.",
    )
    parser.add_argument(
        "--glob",
        dest="pattern",
        default=DEFAULT_GLOB,
        help=f"Glob pattern used inside directories (default: {DEFAULT_GLOB}).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "csv_gz", "parquet"],
        default="csv",
        help="Output format. 'csv_gz' writes a gzip-compressed CSV; 'parquet' writes columnar Parquet.",
    )
    return parser.parse_args()


def find_input_files(input_path: str, pattern: str) -> List[str]:
    files: List[str] = []
    if any(ch in input_path for ch in ["*", "?", "["]):
        files.extend(glob(input_path))
    elif os.path.isdir(input_path):
        files.extend(glob(os.path.join(input_path, pattern)))
    elif os.path.isfile(input_path):
        files.append(input_path)
    # Ensure uniqueness and stable order
    files = sorted(dict.fromkeys(files))
    return files


def read_csv_header(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            header = []
    return header


def row_iter_for_file(path: str) -> Iterable[List[str]]:
    f = open(path, "r", newline="")
    reader = csv.reader(f)
    try:
        next(reader)  # skip header
    except StopIteration:
        f.close()
        return []

    def _iter():
        try:
            for row in reader:
                yield row
        finally:
            f.close()

    return _iter()


def write_rows(
    output_path: str,
    header: List[str],
    rows: Iterable[List[str]],
    *,
    output_format: str = "csv",
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_format == "csv":
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
    elif output_format == "csv_gz":
        import gzip

        with gzip.open(output_path, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
    elif output_format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:
            raise SystemExit(
                "Parquet output requires 'pyarrow'. Install via: pip install pyarrow"
            ) from e

        # Determine expert columns
        k_cols = [c for c in header if c.startswith("expert_logical_k")]
        # Map column to index for fast lookup
        col_idx = {c: i for i, c in enumerate(header)}

        # Define schema (use 32-bit ints; expert columns nullable)
        fields = [
            pa.field("rid", pa.int32(), nullable=False),
            pa.field("rank", pa.int32(), nullable=False),
            pa.field("layer", pa.int32(), nullable=False),
            pa.field("token_index", pa.int32(), nullable=False),
        ] + [pa.field(c, pa.int32(), nullable=True) for c in k_cols]
        schema = pa.schema(fields)

        writer = None
        chunk_size = 500000
        batch_cols_template = ["rid", "rank", "layer", "token_index"] + k_cols

        def flush_batch(batch_cols):
            nonlocal writer
            table = pa.table(batch_cols, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema=schema)
            writer.write_table(table)

        # Prepare empty accumulators
        batch_cols = {c: [] for c in batch_cols_template}
        n_in_batch = 0

        def parse_int_safe(val: str):
            s = (val or "").strip()
            if s == "" or s.lower() == "none":
                return None
            return int(s)

        for row in rows:
            if not row:
                continue
            # Required columns
            batch_cols["rid"].append(int(row[col_idx["rid"]]))
            batch_cols["rank"].append(int(row[col_idx["rank"]]))
            batch_cols["layer"].append(int(row[col_idx["layer"]]))
            batch_cols["token_index"].append(int(row[col_idx["token_index"]]))
            # Expert columns
            for c in k_cols:
                v = row[col_idx[c]] if col_idx[c] < len(row) else ""
                # Empty -> None, otherwise int
                batch_cols[c].append(parse_int_safe(v))

            n_in_batch += 1
            if n_in_batch >= chunk_size:
                flush_batch(batch_cols)
                batch_cols = {c: [] for c in batch_cols_template}
                n_in_batch = 0

        if n_in_batch > 0:
            flush_batch(batch_cols)
        if writer is not None:
            writer.close()
    else:
        raise SystemExit(f"Unsupported output format: {output_format}")


def merge_csvs(
    input_files: List[str],
    output_path: str,
    output_format: str = "csv",
) -> None:
    if not input_files:
        raise SystemExit("No input files matched.")

    # Expect identical headers; use the first file's header.
    target_header = read_csv_header(input_files[0])
    if not target_header:
        raise SystemExit(f"First input file {input_files[0]} is empty.")
    for path in input_files[1:]:
        hdr = read_csv_header(path)
        if hdr and hdr != target_header:
            raise SystemExit(f"Header mismatch between files:\n{input_files[0]}: {target_header}\n{path}: {hdr}")

    # Stream rows from all files and remap 'rid' to a compact range [0..N-1]
    def stream_with_rid_remap():
        rid_map = {}
        next_id = 0
        # Deduplicate decisions across ranks: keep first row per (rid, layer, token_index)
        seen = set()
        # Column indices (must exist in header)
        col_idx = {c: i for i, c in enumerate(target_header)}

        def _is_empty_rid(val) -> bool:
            if val is None:
                return True
            s = str(val).strip()
            return s == "" or s.lower() == "none"

        def _parse_int_safe(val):
            try:
                return int(str(val).strip())
            except Exception:
                return None

        for path in input_files:
            for row in row_iter_for_file(path):
                if not row:
                    continue
                # Assume first column is 'rid'
                rid = row[0] if len(row) > 0 else None
                if not _is_empty_rid(rid):
                    if rid not in rid_map:
                        rid_map[rid] = next_id
                        next_id += 1
                    row = [str(rid_map[rid])] + row[1:]
                # Dedup by (rid, layer, token_index)
                rid_val = _parse_int_safe(row[col_idx.get("rid", 0)])  # default to first col
                layer_val = _parse_int_safe(row[col_idx["layer"]]) if "layer" in col_idx else None
                token_val = _parse_int_safe(row[col_idx["token_index"]]) if "token_index" in col_idx else None
                key = (rid_val, layer_val, token_val)
                if key in seen:
                    continue
                seen.add(key)
                yield row

    write_rows(output_path, target_header, stream_with_rid_remap(), output_format=output_format)


def main():
    args = parse_args()
    files = find_input_files(args.input, args.pattern)
    merge_csvs(input_files=files, output_path=args.output, output_format=args.output_format)


if __name__ == "__main__":
    main()


