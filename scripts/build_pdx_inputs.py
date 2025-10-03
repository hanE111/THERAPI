#!/usr/bin/env python3
"""Generate PDX expression and metadata files aligned to THERAPI expectations."""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def read_gdsc_expression(path: Path):
    df = pd.read_csv(path, index_col=0)
    df = df.astype(np.float64)
    return df


def read_pdx_expression(path: Path):
    df = pd.read_csv(path)
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df = df.astype(np.float64)
    return df


def intersect_genes(ref, query):
    query_index = pd.Index(query)
    shared = query_index.intersection(ref, sort=False)
    return shared.tolist()


def zscore_columns(df):
    values = df.values
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    normalized = (values - means) / stds
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def build_pdx_gex(gdsc_path, pdx_path, allowed_samples=None):
    gdsc = read_gdsc_expression(gdsc_path)
    pdx = read_pdx_expression(pdx_path)
    common_genes = intersect_genes(gdsc.columns, pdx.index)
    aligned = pdx.loc[common_genes].transpose()
    aligned = aligned.astype(np.float64)
    aligned = aligned.sort_index(axis=0)
    if allowed_samples is not None:
        missing_samples = [model for model in allowed_samples if model not in aligned.index]
        if missing_samples:
            LOGGER.warning(
                "Skipping %s models without expression values",
                ", ".join(missing_samples),
            )
        aligned = aligned.loc[aligned.index.intersection(allowed_samples)]
        aligned = aligned.sort_index(axis=0)
    aligned = aligned.reindex(columns=common_genes)
    aligned_z = zscore_columns(aligned)
    # restore original column order
    aligned_z = aligned_z.loc[:, common_genes]
    return aligned_z


def build_pdx_info(raw_path, mapping):
    records = {}
    with raw_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["Model"].strip()
            tissue = row["Tumor Type"].strip()
            if not model:
                continue
            if tissue:
                records.setdefault(model, tissue)
    rows = []
    for model, tissue_code in records.items():
        mapped = mapping.get(tissue_code)
        if mapped is None:
            LOGGER.warning("Skipping model %s with unmapped tissue %s", model, tissue_code)
            continue
        tissue_name, label = mapped
        rows.append({
            "Sample": model,
            "primary_disease": tissue_name,
            "tissue": tissue_name,
            "tissue_label": label,
        })
    info_df = pd.DataFrame(rows)
    return info_df, [row["Sample"] for row in rows]


def parse_args():
    parser = argparse.ArgumentParser(description="Build PDX inputs aligned to THERAPI data format")
    parser.add_argument("--gdsc-gex", type=Path, default=Path("../data/GDSC_gex.csv"), help="Path to GDSC expression CSV")
    parser.add_argument("--pdx-gex", type=Path, default=Path("../../data/pdx_csvs/RNAseq_fpkm.csv"), help="Path to PDX RNAseq CSV")
    parser.add_argument("--pdx-raw", type=Path, default=Path("../../data/pdx_csvs/train_PCT_raw_data.csv"), help="Path to raw PDX metadata CSV")
    parser.add_argument("--out-gex", type=Path, default=Path("../data/PDX_gex.csv"), help="Output path for aligned expression")
    parser.add_argument("--out-info", type=Path, default=Path("../data/PDX_info.csv"), help="Output path for metadata")
    return parser.parse_args()


PDX_TISSUE_MAPPING = {
    "GC": ("gastric", 10),
    "CRC": ("colon", 4),
    "BRCA": ("breast", 2),
    "PDAC": ("pancreatic", 9),
    "NSCLC": ("lung", 0),
    "CM": ("skin", 6),
}


def main() -> None:
    args = parse_args()
    info, models = build_pdx_info(args.pdx_raw, PDX_TISSUE_MAPPING)
    aligned = build_pdx_gex(args.gdsc_gex, args.pdx_gex, models)
    aligned_samples = aligned.index.tolist()

    info_indexed = info.set_index("Sample")
    missing_metadata = [sample for sample in aligned_samples if sample not in info_indexed.index]
    if missing_metadata:
        LOGGER.warning(
            "Dropping %s expression samples without metadata",
            ", ".join(missing_metadata),
        )
        aligned = aligned.loc[[sample for sample in aligned_samples if sample in info_indexed.index]]
        aligned_samples = aligned.index.tolist()

    metadata_only = [sample for sample in info_indexed.index if sample not in aligned_samples]
    if metadata_only:
        LOGGER.warning(
            "Excluding %s metadata entries without expression",
            ", ".join(metadata_only),
        )

    info_indexed = info_indexed.loc[info_indexed.index.intersection(aligned_samples)]

    info = info_indexed.reindex(aligned_samples).reset_index()
    args.out_gex.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(args.out_gex)
    info.to_csv(args.out_info, index=False)
    LOGGER.info("Saved expression to %s with shape %s", args.out_gex, aligned.shape)
    LOGGER.info("Saved metadata to %s with %s samples", args.out_info, info.shape[0])


if __name__ == "__main__":
    main()
