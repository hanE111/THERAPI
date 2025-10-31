"""Generate predictor-ready inputs for a target cohort using aligner weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_TREATMENT_ALIAS: Dict[str, str] = {
    "5FU": "5-Fluorouracil",
    "ABRAXANE": "Paclitaxel",
    "BKM120": "Buparlisib",
    "BYL719": "Alpelisib",
    "INC424": "Ruxolitinib",
    "LEE011": "Ribociclib",
}

RESPONSE_POSITIVE_PREFIXES = ("CR", "PR", "SD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build predictor inputs for target cohort")
    parser.add_argument("--align_prefix", type=str, default="../data/PDX1_align", help="Prefix of exported aligner embeddings (without suffix)")
    parser.add_argument("--rank_path", type=str, default="../data/PDX1_rankrepresentation.csv", help="Path to target rank representation CSV")
    parser.add_argument("--gdsc_resp", type=str, default="../data/GDSC_resp.csv", help="Path to GDSC response CSV")
    parser.add_argument("--gdsc_pert", type=str, default="../data/GDSC_perturbation.npy", help="Path to GDSC perturbation embeddings")
    parser.add_argument("--gdsc_comp", type=str, default="../data/GDSC_perturbation_compound.npy", help="Path to GDSC compound embeddings")
    parser.add_argument("--response_files", nargs="*", default=("../data_pdx1/unlabeled_PCT_curve_metrics.csv", "../data_pdx1/labeled_PCT_curve_metrics.csv"), help="List of PDX response metric CSVs")
    parser.add_argument("--treatment_map", type=str, default=None, help="Optional CSV with columns Treatment,GDSC_Drug,Canonical_SMILES")
    parser.add_argument("--output_prefix", type=str, default="../data/PDX1", help="Prefix for generated predictor assets")
    parser.add_argument("--allow_combinations", action="store_true", help="Whether to keep combination therapies (requires mappings)")
    return parser.parse_args()


def load_weights(prefix: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.load(f"{prefix}_weights.npy")
    target_ids = np.load(f"{prefix}_target_ids.npy")
    source_ids = np.load(f"{prefix}_source_ids.npy")
    return weights, target_ids, source_ids


def load_rank_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def load_response_tables(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if {"Model", "Treatment", "ResponseCategory"}.issubset(df.columns):
            frames.append(df[["Model", "Treatment", "ResponseCategory"]])
    if not frames:
        raise ValueError("No response tables with required columns were found")
    resp = pd.concat(frames, ignore_index=True)
    resp["Model"] = resp["Model"].astype(str).str.strip()
    resp["Treatment"] = resp["Treatment"].astype(str).str.strip()
    resp["ResponseCategory"] = resp["ResponseCategory"].astype(str).str.strip()
    resp = resp.dropna(subset=["Model", "Treatment"])
    resp = resp.sort_values(["Model", "Treatment"]).drop_duplicates(subset=["Model", "Treatment"], keep="first")
    return resp


def load_treatment_map(path: Path | None) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if path is None or not path.exists():
        return mapping
    df = pd.read_csv(path)
    required_cols = {"Treatment", "GDSC_Drug"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Treatment map must contain columns {required_cols}")
    for _, row in df.iterrows():
        treatment = str(row["Treatment"]).strip()
        gdsc_drug = str(row["GDSC_Drug"]).strip()
        mapping[treatment.upper()] = {
            "gdsc_drug": gdsc_drug,
            "Canonical_SMILES": row.get("Canonical_SMILES", None),
        }
    return mapping


def normalize_treatment(name: str) -> str:
    cleaned = name.replace('"', '').replace("'", "").replace("®", "").replace("\u2019", "")
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    return cleaned.strip()


def is_positive_response(category: str) -> int:
    cat = category.upper()
    if any(cat.startswith(prefix) for prefix in RESPONSE_POSITIVE_PREFIXES) and "PD" not in cat:
        return 1
    if "CR" in cat or "PR" in cat:
        return 1
    return 0


def build_gdsc_lookup(gdsc_resp: pd.DataFrame, source_ids: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    source_index = {cid: idx for idx, cid in enumerate(source_ids)}
    drug_to_rows: Dict[str, List[int]] = {}
    drug_to_cells: Dict[str, List[int]] = {}
    drug_to_smiles: Dict[str, str] = {}

    for idx, row in gdsc_resp.iterrows():
        cell_id = row['Cell_ID']
        if cell_id not in source_index:
            continue
        drug = str(row['Drug']).strip()
        key = drug.upper()
        drug_to_rows.setdefault(key, []).append(idx)
        drug_to_cells.setdefault(key, []).append(source_index[cell_id])
        if key not in drug_to_smiles and isinstance(row.get('Canonical_SMILES'), str):
            drug_to_smiles[key] = row['Canonical_SMILES']

    drug_to_rows = {k: np.array(v, dtype=int) for k, v in drug_to_rows.items()}
    drug_to_cells = {k: np.array(v, dtype=int) for k, v in drug_to_cells.items()}
    return drug_to_rows, drug_to_cells, drug_to_smiles


def resolve_drug_name(treatment: str, gdsc_drug_keys: Dict[str, str], treatment_map: Dict[str, Dict[str, str]], allow_combo: bool) -> Tuple[str | None, str | None]:
    cleaned = normalize_treatment(treatment)
    key = cleaned.upper()
    if not allow_combo and ('+' in key or '/' in key):
        return None, None

    if key in gdsc_drug_keys:
        return gdsc_drug_keys[key], None

    if key in DEFAULT_TREATMENT_ALIAS:
        alias_key = DEFAULT_TREATMENT_ALIAS[key].upper()
        if alias_key in gdsc_drug_keys:
            return gdsc_drug_keys[alias_key], None

    if key in treatment_map:
        mapped = treatment_map[key]['gdsc_drug']
        alias_key = str(mapped).strip().upper()
        if alias_key in gdsc_drug_keys:
            return gdsc_drug_keys[alias_key], treatment_map[key].get('Canonical_SMILES')

    return None, None


def main():
    args = parse_args()

    weights, target_ids, source_ids = load_weights(Path(args.align_prefix))
    rank_df = load_rank_features(Path(args.rank_path))
    response_df = load_response_tables([Path(p) for p in args.response_files])
    treatment_map = load_treatment_map(Path(args.treatment_map) if args.treatment_map else None)

    gdsc_resp = pd.read_csv(args.gdsc_resp)
    gdsc_pert = np.load(args.gdsc_pert)
    gdsc_comp = np.load(args.gdsc_comp)

    drug_to_rows, drug_to_cells, drug_to_smiles = build_gdsc_lookup(gdsc_resp, source_ids)
    gdsc_drug_keys = {drug.upper(): drug for drug in gdsc_resp['Drug'].unique()}

    target_index = {sid: idx for idx, sid in enumerate(target_ids)}

    perturb_list: List[np.ndarray] = []
    chem_list: List[np.ndarray] = []
    rank_list: List[np.ndarray] = []
    records: List[Dict[str, object]] = []
    skipped_records: List[Dict[str, object]] = []

    for _, row in response_df.iterrows():
        model = str(row['Model']).strip()
        treatment = str(row['Treatment']).strip()
        response_cat = row['ResponseCategory']

        if model not in target_index or model not in rank_df.index:
            skipped_records.append({"Model": model, "Treatment": treatment, "reason": "missing_model"})
            continue

        gdsc_name, override_smiles = resolve_drug_name(treatment, gdsc_drug_keys, treatment_map, args.allow_combinations)
        if gdsc_name is None:
            skipped_records.append({"Model": model, "Treatment": treatment, "reason": "unmapped_treatment"})
            continue

        gdsc_key = gdsc_name.upper()
        if gdsc_key not in drug_to_rows:
            skipped_records.append({"Model": model, "Treatment": treatment, "reason": "no_gdsc_profiles"})
            continue

        rows_idx = drug_to_rows[gdsc_key]
        cells_idx = drug_to_cells[gdsc_key]
        weight_vec = weights[target_index[model]][cells_idx]
        weight_sum = weight_vec.sum()
        if weight_sum <= 0:
            skipped_records.append({"Model": model, "Treatment": treatment, "reason": "zero_attention"})
            continue
        normalized_weights = weight_vec / weight_sum

        perturb_emb = np.average(gdsc_pert[rows_idx], axis=0, weights=normalized_weights)
        chem_emb = np.average(gdsc_comp[rows_idx], axis=0, weights=normalized_weights)

        perturb_list.append(perturb_emb.astype(np.float32))
        chem_list.append(chem_emb.astype(np.float32))
        rank_list.append(rank_df.loc[model].to_numpy(dtype=np.float32))

        label = is_positive_response(response_cat)
        canonical_smiles = override_smiles or drug_to_smiles.get(gdsc_key, None)
        records.append({
            "Model": model,
            "Drug": gdsc_name,
            "Original_Treatment": treatment,
            "Canonical_SMILES": canonical_smiles,
            "Label": label,
        })

    if not records:
        raise RuntimeError("No predictor samples were generated. Check mapping configurations.")

    perturb_array = np.stack(perturb_list)
    chem_array = np.stack(chem_list)
    rank_array = np.stack(rank_list)

    sample_order = np.array([record["Model"] for record in records])

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{output_prefix}_perturbation.npy", perturb_array)
    np.save(f"{output_prefix}_perturbation_compound.npy", chem_array)
    np.save(f"{output_prefix}_rank.npy", rank_array)
    np.save(f"{output_prefix}_samples.npy", sample_order)

    resp_df = pd.DataFrame(records)
    resp_df.to_csv(f"{output_prefix}_resp.csv", index=False)

    skips_path = f"{output_prefix}_skipped.json"
    with open(skips_path, "w", encoding="utf-8") as fp:
        json.dump(skipped_records, fp, indent=2)

    print(f"Generated {len(records)} predictor samples")
    print(f"Perturbation embeddings saved to {output_prefix}_perturbation.npy")
    print(f"Chemical embeddings saved to {output_prefix}_perturbation_compound.npy")
    print(f"Rank features saved to {output_prefix}_rank.npy")
    print(f"Sample order saved to {output_prefix}_samples.npy")
    print(f"Response table saved to {output_prefix}_resp.csv")
    print(f"Skipped combinations logged to {skips_path}")

if __name__ == "__main__":
    main()
