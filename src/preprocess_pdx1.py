"""Utilities to preprocess PDX1 cohorts into THERAPI-compatible matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PDX_TUMOR_TO_TISSUE = {
    "BRCA": "breast",
    "CM": "skin",
    "CRC": "colon",
    "GC": "gastric",
    "NSCLC": "lung",
    "PDAC": "pancreatic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PDX1 data for THERAPI")
    parser.add_argument("--pdx_rna", type=str, default="../data_pdx1/RNAseq_fpkm.csv", help="Path to raw PDX1 RNAseq FPKM matrix")
    parser.add_argument("--train_raw", type=str, default="../data_pdx1/train_PCT_raw_data.csv", help="Path to train cohort raw metrics (for tumor types)")
    parser.add_argument("--test_raw", type=str, default="../data_pdx1/test_PCT_raw_data.csv", help="Path to test cohort raw metrics (for tumor types)")
    parser.add_argument("--output_dir", type=str, default="../data/", help="Directory to write processed outputs")
    parser.add_argument("--gdsc_gex", type=str, default="../data/GDSC_gex.csv", help="Reference GDSC expression matrix (for gene ordering)")
    parser.add_argument("--gdsc_rank", type=str, default="../data/GDSC_rankrepresentation.csv", help="Reference GDSC rank representations")
    parser.add_argument("--gdsc_info", type=str, default="../data/GDSC_info.csv", help="Reference GDSC metadata with tissue labels")
    parser.add_argument("--rank_alpha", type=float, default=1.0, help="Ridge regression strength for rank mapping")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--extra_tumor_map", type=str, default=None, help="Optional JSON mapping of PDX tumor codes to GDSC tissue names")
    return parser.parse_args()


def load_expression(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Sample" not in df.columns:
        raise ValueError("Expected a 'Sample' column in PDX RNAseq matrix")
    df = df.set_index("Sample")
    expr = df.transpose()
    expr.index.name = "Model"
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.fillna(0.0)
    return expr


def load_metadata(train_path: Path, test_path: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in [train_path, test_path]:
        if path is None:
            continue
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "Model" not in df.columns or "Tumor Type" not in df.columns:
            continue
        frames.append(df[["Model", "Tumor Type"]])
    if not frames:
        raise ValueError("No metadata tables with 'Model' and 'Tumor Type' were found")
    meta = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    meta = meta.rename(columns={"Tumor Type": "tumor_type_code"})
    meta["tumor_type_code"] = meta["tumor_type_code"].astype(str).str.strip()
    meta = meta.set_index("Model")
    return meta


def prepare_tissue_mapping(gdsc_info: pd.DataFrame, extra_map: Dict[str, str] | None = None) -> Tuple[Dict[str, str], Dict[str, int]]:
    tissue_lookup = gdsc_info[["tissue", "tissue_label"]].drop_duplicates()
    tissue_lookup["tissue_lower"] = tissue_lookup["tissue"].str.lower()
    tissue_to_label = dict(zip(tissue_lookup["tissue_lower"], tissue_lookup["tissue_label"]))

    tumor_to_tissue = dict(PDX_TUMOR_TO_TISSUE)
    if extra_map:
        tumor_to_tissue.update({k: v for k, v in extra_map.items() if v})

    for code, tissue in tumor_to_tissue.items():
        if tissue.lower() not in tissue_to_label:
            raise ValueError(f"Missing tissue label in GDSC metadata for tissue '{tissue}' mapped from tumor code '{code}'")

    return tumor_to_tissue, tissue_to_label


def align_expression(expr: pd.DataFrame, gene_order: Iterable[str]) -> pd.DataFrame:
    expr = expr.reindex(columns=gene_order, fill_value=0.0)
    return expr


def filter_common_models(expr: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = sorted(set(expr.index) & set(meta.index))
    if not common:
        raise ValueError("No overlapping models between expression matrix and metadata")
    expr_f = expr.loc[common]
    meta_f = meta.loc[common]
    return expr_f, meta_f


def compute_rank_mapping(gdsc_expr: pd.DataFrame, gdsc_rank: pd.DataFrame, alpha: float) -> Tuple[StandardScaler, Ridge]:
    if gdsc_expr.index.difference(gdsc_rank.index).any():
        gdsc_rank = gdsc_rank.reindex(gdsc_expr.index)
    if gdsc_rank.isna().any().any():
        raise ValueError("GDSC rank representation contains NaNs after alignment")
    log_expr = np.log2(gdsc_expr + 1.0)
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(log_expr)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(expr_scaled, gdsc_rank.values)
    return scaler, model


def transform_rank_features(expr: pd.DataFrame, scaler: StandardScaler, model: Ridge) -> pd.DataFrame:
    log_expr = np.log2(expr + 1.0)
    expr_scaled = scaler.transform(log_expr)
    rank_values = model.predict(expr_scaled)
    cols = [str(i) for i in range(rank_values.shape[1])]
    return pd.DataFrame(rank_values, index=expr.index, columns=cols)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdx_rna_path = Path(args.pdx_rna)
    train_raw_path = Path(args.train_raw)
    test_raw_path = Path(args.test_raw)
    gdsc_gex_path = Path(args.gdsc_gex)
    gdsc_rank_path = Path(args.gdsc_rank)
    gdsc_info_path = Path(args.gdsc_info)

    pdx_expr_raw = load_expression(pdx_rna_path)
    meta = load_metadata(train_raw_path, test_raw_path)

    if args.extra_tumor_map:
        with open(args.extra_tumor_map, "r", encoding="utf-8") as fp:
            extra_map = json.load(fp)
    else:
        extra_map = None

    gdsc_expr = pd.read_csv(gdsc_gex_path, index_col=0)
    gdsc_rank = pd.read_csv(gdsc_rank_path, index_col=0)
    gdsc_info = pd.read_csv(gdsc_info_path)

    tumor_to_tissue, tissue_to_label = prepare_tissue_mapping(gdsc_info, extra_map)

    pdx_expr_filtered, meta_filtered = filter_common_models(pdx_expr_raw, meta)

    meta_filtered["tissue"] = meta_filtered["tumor_type_code"].map(tumor_to_tissue)
    if meta_filtered["tissue"].isna().any():
        missing_codes = sorted(meta_filtered[meta_filtered["tissue"].isna()].index.unique())
        raise ValueError(f"Unmapped tumor type codes for models: {missing_codes}")
    meta_filtered["tissue_label"] = meta_filtered["tissue"].str.lower().map(tissue_to_label)
    if meta_filtered["tissue_label"].isna().any():
        missing_tissues = sorted(meta_filtered[meta_filtered["tissue_label"].isna()]["tissue"].unique())
        raise ValueError(f"Missing tissue labels for tissues: {missing_tissues}")

    gene_order = list(gdsc_expr.columns)
    pdx_expr_aligned = align_expression(pdx_expr_filtered, gene_order)
    pdx_expr_log = np.log2(pdx_expr_aligned + 1.0)

    scaler, rank_model = compute_rank_mapping(gdsc_expr, gdsc_rank, alpha=args.rank_alpha)
    pdx_rank = transform_rank_features(pdx_expr_aligned, scaler, rank_model)

    info_df = meta_filtered.copy()
    info_df = info_df.reset_index().rename(columns={"index": "Model"})

    gex_out = output_dir / "PDX1_gex.csv"
    info_out = output_dir / "PDX1_info.csv"
    rank_out = output_dir / "PDX1_rankrepresentation.csv"
    model_out = output_dir / "pdx1_rank_transform.joblib"

    for path in [gex_out, info_out, rank_out, model_out]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output file {path} already exists. Use --overwrite to replace it.")

    pdx_expr_log.to_csv(gex_out)
    info_df.to_csv(info_out, index=False)
    pdx_rank.to_csv(rank_out)
    joblib.dump({"scaler": scaler, "model": rank_model, "gene_order": gene_order}, model_out)

    summary = {
        "n_models": len(pdx_expr_log),
        "n_genes": len(gene_order),
        "rank_dim": pdx_rank.shape[1],
        "output_dir": str(output_dir.resolve()),
        "missing_models": sorted(set(meta.index) - set(meta_filtered.index)),
    }

    summary_path = output_dir / "PDX1_preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved PDX1 expression to {gex_out}")
    print(f"Saved PDX1 metadata to {info_out}")
    print(f"Saved PDX1 rank representations to {rank_out}")
    print(f"Saved rank transformation model to {model_out}")
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()
