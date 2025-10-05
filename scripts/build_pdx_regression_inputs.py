#!/usr/bin/env python3
"""Generate PDX regression inputs without aligner dependency.

This script prepares data for the simplified regression approach that uses
only genomic expression + chemical features (no perturbations or rank features).
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


def read_gdsc_expression(path: Path):
    """Read GDSC expression data for gene alignment."""
    df = pd.read_csv(path, index_col=0)
    df = df.astype(np.float64)
    return df


def read_pdx_expression(path: Path):
    """Read PDX expression data."""
    df = pd.read_csv(path)
    gene_col = df.columns[0]
    df = df.set_index(gene_col)
    df = df.astype(np.float64)
    return df


def intersect_genes(ref, query):
    """Find common genes between reference and query datasets."""
    query_index = pd.Index(query)
    shared = query_index.intersection(ref, sort=False)
    return shared.tolist()


def zscore_normalize(df):
    """Z-score normalize columns (samples)."""
    values = df.values
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1.0
    normalized = (values - means) / stds
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def align_pdx_expression(gdsc_path, pdx_path, allowed_samples=None):
    """Align PDX expression to GDSC gene space."""
    gdsc = read_gdsc_expression(gdsc_path)
    pdx = read_pdx_expression(pdx_path)

    common_genes = intersect_genes(gdsc.columns, pdx.index)

    LOGGER.info(
        "Gene alignment: GDSC has %d genes, PDX has %d genes, %d common",
        len(gdsc.columns),
        len(pdx.index),
        len(common_genes),
    )

    # Filter to common genes and transpose to (samples x genes)
    aligned = pdx.loc[common_genes].transpose()
    aligned = aligned.astype(np.float64)
    aligned = aligned.sort_index(axis=0)

    if allowed_samples is not None:
        missing_samples = [model for model in allowed_samples if model not in aligned.index]
        if missing_samples:
            LOGGER.warning(
                "Skipping %d models without expression values: %s",
                len(missing_samples),
                ", ".join(missing_samples[:5]),
            )
        aligned = aligned.loc[aligned.index.intersection(allowed_samples)]
        aligned = aligned.sort_index(axis=0)

    # Pad to full GDSC gene space with zeros
    aligned = aligned.reindex(columns=gdsc.columns, fill_value=0.0)

    # Z-score normalize
    aligned_z = zscore_normalize(aligned)

    return aligned_z


def build_pdx_info(raw_path, mapping):
    """Build PDX metadata from raw data."""
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


def build_smiles_dictionary(paths):
    """Build SMILES lookup from GDSC data and known PDX drugs."""
    mapping = {}

    # Add known SMILES for PDX drugs that may not be in GDSC
    pdx_drug_smiles = {
        'dacarbazine': 'CN(C)C(=N)N=NC(=O)N',
        'cetuximab': None,  # Monoclonal antibody - no SMILES
    }
    mapping.update(pdx_drug_smiles)

    # Load GDSC SMILES
    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            LOGGER.warning("Skipping missing SMILES source %s", path)
            continue
        df = pd.read_csv(path_obj)
        cols = {c.lower(): c for c in df.columns}
        if "drug" not in cols or "canonical_smiles" not in cols:
            continue
        drug_col = cols["drug"]
        smiles_col = cols["canonical_smiles"]
        for drug, smiles in df[[drug_col, smiles_col]].dropna().itertuples(index=False):
            drug_str = str(drug).strip()
            smiles_str = str(smiles).strip()
            if smiles_str:
                # Add both original case and lowercase for matching
                mapping.setdefault(drug_str, smiles_str)
                mapping.setdefault(drug_str.lower(), smiles_str)

    return mapping


def smiles_to_features(smiles, n_bits, radius, target_dim=978):
    """Convert SMILES to Morgan fingerprint features."""
    if not smiles:
        return np.zeros(target_dim, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        LOGGER.warning("Unable to parse SMILES '%s'", smiles)
        return np.zeros(target_dim, dtype=np.float32)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    if n_bits < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:n_bits] = arr
        return padded
    return arr[:target_dim]


def build_response_table(curves):
    """Extract BestAvgResponse as regression target."""
    working = curves.copy()
    working = working.rename(columns={"Treatment": "Drug", "BestAvgResponse": "Target"})
    working = working.dropna(subset=["Target"])
    working["Target"] = working["Target"].astype(np.float32)
    return working


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build PDX regression inputs (genomic + chemical only)"
    )
    parser.add_argument("--gdsc-gex", type=Path, default=Path("../data/GDSC_gex.csv"))
    parser.add_argument("--gdsc-smiles", type=Path, default=Path("../data/GDSC_resp.csv"))
    parser.add_argument("--pdx-gex", type=Path, default=Path("../../data/pdx_csvs/RNAseq_fpkm.csv"))
    parser.add_argument("--pdx-raw", type=Path, default=Path("../../data/pdx_csvs/train_PCT_raw_data.csv"))
    parser.add_argument("--pdx-curves", type=Path, default=Path("../../data/pdx_csvs/test_PCT_curve_metrics.csv"))
    parser.add_argument("--compound-bits", type=int, default=1024)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--out-gex", type=Path, default=Path("../data/PDX_regression_gex.csv"))
    parser.add_argument("--out-compound", type=Path, default=Path("../data/PDX_regression_compound.npy"))
    parser.add_argument("--out-resp", type=Path, default=Path("../data/PDX_regression_resp.csv"))
    parser.add_argument("--out-info", type=Path, default=Path("../data/PDX_regression_info.csv"))
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

    # Load metadata
    LOGGER.info("Loading PDX metadata")
    info, models = build_pdx_info(args.pdx_raw, PDX_TISSUE_MAPPING)

    # Align PDX expression to GDSC gene space
    LOGGER.info("Aligning PDX expression to GDSC gene space")
    aligned_gex = align_pdx_expression(args.gdsc_gex, args.pdx_gex, models)

    # Load response data
    LOGGER.info("Loading PDX response data")
    curves = pd.read_csv(args.pdx_curves)

    # Filter to models with expression data
    present_models = set(aligned_gex.index)
    curves = curves[curves["Model"].isin(present_models)]

    if curves.empty:
        raise RuntimeError("No PDX models with both expression and response data")

    # Build response table with regression targets
    resp_df = build_response_table(curves)
    if resp_df.empty:
        raise RuntimeError("No labeled PDX entries after filtering")

    LOGGER.info("Found %d model-drug pairs with response data", len(resp_df))

    # Get unique models that have response data
    models_with_response = sorted(set(resp_df["Model"]))

    # Filter expression to only models with response data
    aligned_gex = aligned_gex.loc[models_with_response]

    # Build SMILES lookup
    LOGGER.info("Preparing compound descriptors")
    smiles_lookup = build_smiles_dictionary([args.gdsc_smiles])

    # Build compound features for each drug
    unique_drugs = sorted(resp_df["Drug"].unique())
    compound_features = {}
    for drug in unique_drugs:
        smiles = smiles_lookup.get(drug) or smiles_lookup.get(drug.lower())
        if smiles is None:
            LOGGER.warning("No SMILES found for drug '%s', using zero vector", drug)
        compound_features[drug] = smiles_to_features(
            smiles,
            n_bits=args.compound_bits,
            radius=args.fingerprint_radius,
        )

    # Build aligned arrays: one row per model-drug pair
    LOGGER.info("Building aligned feature arrays")
    genomic_rows = []
    compound_rows = []

    for model, drug in resp_df[["Model", "Drug"]].itertuples(index=False):
        genomic_rows.append(aligned_gex.loc[model].to_numpy(dtype=np.float32))
        compound_rows.append(compound_features[drug])

    genomic_array = np.asarray(genomic_rows, dtype=np.float32)
    compound_array = np.asarray(compound_rows, dtype=np.float32)

    # Verify dimensions
    assert genomic_array.shape[0] == compound_array.shape[0] == resp_df.shape[0], \
        f"Dimension mismatch: genomic {genomic_array.shape[0]}, compound {compound_array.shape[0]}, resp {resp_df.shape[0]}"

    LOGGER.info("Genomic features shape: %s", genomic_array.shape)
    LOGGER.info("Compound features shape: %s", compound_array.shape)

    # Create genomic dataframe with model-drug pair indices
    gex_indices = [f"{model}_{drug}" for model, drug in resp_df[["Model", "Drug"]].itertuples(index=False)]
    genomic_df = pd.DataFrame(genomic_array, index=gex_indices, columns=aligned_gex.columns)

    # Add SMILES to response dataframe
    resp_out = resp_df[["Model", "Drug", "ResponseCategory", "Target"]].copy()
    resp_out["Canonical_SMILES"] = [smiles_lookup.get(drug) for drug in resp_out["Drug"]]

    # Filter info to models with response data
    info_indexed = info.set_index("Sample")
    filtered_info = info_indexed.loc[info_indexed.index.intersection(models_with_response)].reset_index()

    # Save outputs
    LOGGER.info("Saving artifacts")
    args.out_gex.parent.mkdir(parents=True, exist_ok=True)

    genomic_df.to_csv(args.out_gex)
    np.save(args.out_compound, compound_array)
    resp_out.to_csv(args.out_resp, index=False)
    filtered_info.to_csv(args.out_info, index=False)

    LOGGER.info("Saved genomic features to %s with shape %s", args.out_gex, genomic_array.shape)
    LOGGER.info("Saved compound features to %s with shape %s", args.out_compound, compound_array.shape)
    LOGGER.info("Saved response data to %s with %d pairs", args.out_resp, resp_df.shape[0])
    LOGGER.info("Saved metadata to %s with %d models", args.out_info, filtered_info.shape[0])

    # Summary statistics
    LOGGER.info("\n=== Summary ===")
    LOGGER.info("Total model-drug pairs: %d", len(resp_df))
    LOGGER.info("Unique models: %d", resp_df["Model"].nunique())
    LOGGER.info("Unique drugs: %d", resp_df["Drug"].nunique())
    LOGGER.info("Target statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                resp_df["Target"].mean(),
                resp_df["Target"].std(),
                resp_df["Target"].min(),
                resp_df["Target"].max())

    smiles_count = resp_out["Canonical_SMILES"].notna().sum()
    LOGGER.info("SMILES coverage: %d/%d (%.1f%%)",
                smiles_count, len(resp_out), 100 * smiles_count / len(resp_out))


if __name__ == "__main__":
    main()
