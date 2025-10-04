#!/usr/bin/env python3
"""Generate labeled PDX artifacts for the THERAPI predictor."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Builtin fallbacks for lint environments that restrict __builtins__ access.
# ---------------------------------------------------------------------------
_builtins = globals().get("__builtins__", {})
if hasattr(_builtins, "__dict__"):
    _builtins = _builtins.__dict__

_STR = _builtins.get("str", str)
_INT = _builtins.get("int", int)
_SET = _builtins.get("set", set)
_SORTED = _builtins.get("sorted", sorted)
_ISINSTANCE = _builtins.get("isinstance", isinstance)
_RUNTIME_ERROR = _builtins.get("RuntimeError", RuntimeError)
_EXCEPTION = _builtins.get("Exception", Exception)


try:  # pragma: no cover - import guard
    import torch
except _EXCEPTION as exc:  # pylint: disable=broad-except
    raise _RUNTIME_ERROR("Torch is required to generate PDX features") from exc

try:  # pragma: no cover - import guard
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except _EXCEPTION as exc:  # pylint: disable=broad-except
    raise _RUNTIME_ERROR("RDKit is required to build compound fingerprints") from exc


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if _STR(SRC_DIR) not in sys.path:
    sys.path.append(_STR(SRC_DIR))

from model import GDSC_AE, TCGA_weightencoder  # pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)


class AlignerComponents:
    """Container for aligner modules and cached tensors."""

    def __init__(self, gdsc_ae, pdx_encoder, gdsc_latent, gdsc_expression):
        self.gdsc_ae = gdsc_ae
        self.pdx_encoder = pdx_encoder
        self.gdsc_latent = gdsc_latent
        self.gdsc_expression = gdsc_expression


def parse_args():
    parser = argparse.ArgumentParser(description="Build labeled PDX predictor inputs")
    parser.add_argument("--aligner-ckpt", default="../src/ckpts/THERAPI_aligner.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gdsc-gex", default="../data/GDSC_gex.csv")
    parser.add_argument("--gdsc-rank", default="../data/GDSC_rankrepresentation.csv")
    parser.add_argument("--gdsc-smiles", default="../data/GDSC_resp.csv")
    parser.add_argument("--pdx-gex", default="../data/PDX_unlabeled_gex.csv")
    parser.add_argument("--pdx-info", default="../data/PDX_unlabeled_info.csv")
    parser.add_argument("--pdx-curves", default="../../data/pdx_csvs/test_PCT_curve_metrics.csv")
    parser.add_argument("--compound-bits", default="1024")
    parser.add_argument("--fingerprint-radius", default="2")
    parser.add_argument("--out-resp", default="../data/PDX_resp.csv")
    parser.add_argument("--out-rank", default="../data/PDX_rankrepresentation.csv")
    parser.add_argument("--out-perturbation", default="../data/PDX_perturbation.npy")
    parser.add_argument("--out-compound", default="../data/PDX_perturbation_compound.npy")
    parser.add_argument("--out-labeled-gex", default="../data/PDX_labeled_gex.csv")
    parser.add_argument("--out-labeled-info", default="../data/PDX_labeled_info.csv")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_expression(path):
    df = pd.read_csv(path, index_col=0)
    return df.astype(np.float32)


def load_rank(path):
    df = pd.read_csv(path, index_col=0)
    return df.astype(np.float32)


def instantiate_aligner(device, gdsc_gex, num_tissues, checkpoint):
    n_genes = gdsc_gex.shape[1]
    gdsc_ae = GDSC_AE(n_genes=n_genes, n_classes=num_tissues, n_latent=128).to(device)
    pdx_encoder = TCGA_weightencoder(
        n_genes=n_genes,
        n_latent=128,
        n_celines=gdsc_gex.shape[0],
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    gdsc_ae.load_state_dict(state["gdsc_AE"])
    pdx_encoder.load_state_dict(state["pdx_weightencoder"])

    gdsc_ae.eval()
    pdx_encoder.eval()

    gdsc_expression = torch.tensor(gdsc_gex.values, dtype=torch.float32, device=device)
    with torch.no_grad():
        gdsc_latent, _ = gdsc_ae(gdsc_expression)

    return AlignerComponents(gdsc_ae, pdx_encoder, gdsc_latent, gdsc_expression)


def compute_model_embeddings(pdx_gex, aligner, device):
    weights = {}
    perturbations = {}
    with torch.no_grad():
        for model, row in pdx_gex.iterrows():
            x_tensor = torch.tensor(row.values, dtype=torch.float32, device=device).unsqueeze(0)
            weight_vec, _, embed, _ = aligner.pdx_encoder(
                x_tensor,
                aligner.gdsc_latent,
                aligner.gdsc_expression,
            )
            weights[model] = weight_vec.squeeze(0).cpu().numpy().astype(np.float32)
            perturbations[model] = embed.squeeze(0).cpu().numpy().astype(np.float32)
    return weights, perturbations


def derive_rank_features(weights, gdsc_rank):
    rank_matrix = gdsc_rank.values.astype(np.float32)
    rank_features = {}
    for model, weight_vec in weights.items():
        rank_features[model] = np.matmul(weight_vec, rank_matrix)
    return rank_features


def build_smiles_dictionary(paths):
    mapping = {}

    # First, add known SMILES for PDX drugs that may not be in GDSC
    # Dacarbazine SMILES from PubChem CID 2942
    pdx_drug_smiles = {
        'dacarbazine': 'CN(C)C(=N)N=NC(=O)N',
        # Cetuximab is a monoclonal antibody - no small molecule SMILES
        'cetuximab': None,
    }
    mapping.update(pdx_drug_smiles)

    # Create case-insensitive mapping for GDSC drugs
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
            drug_str = _STR(drug).strip()
            smiles_str = _STR(smiles).strip()
            if smiles_str:
                # Add both original case and lowercase for matching
                mapping.setdefault(drug_str, smiles_str)
                mapping.setdefault(drug_str.lower(), smiles_str)
    return mapping


def smiles_to_features(smiles, n_bits, radius, target_dim=978):
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


def map_response_label(category):
    """
    Map PDX response categories to binary labels.

    Sensitive (1): CR, PR, CR-->, PR--> (complete/partial response initially)
    Resistant (0): PD, SD, SD-->, -->PD (progressive/stable disease)

    For transitions, use the INITIAL response before arrow.
    """
    if not _ISINSTANCE(category, _STR):
        return None
    token = category.strip().upper()
    if not token:
        return None

    # Handle 'combo' separately - these are combination treatments
    if token == 'COMBO':
        return None  # Skip combination treatments for now

    # For transitions (e.g., "CR-->PD"), use the FIRST response
    # Split on '-->' and take first part
    if '-->' in token:
        token = token.split('-->')[0].strip()

    # Complete Response or Partial Response = Sensitive
    if token in ('CR', 'PR'):
        return 1
    # Progressive Disease or Stable Disease = Resistant
    elif token in ('PD', 'SD'):
        return 0
    # Catch any remaining patterns
    elif 'CR' in token or 'PR' in token:
        return 1
    elif 'PD' in token or 'SD' in token:
        return 0

    return None


def build_response_table(curves):
    working = curves.copy()
    working["Label"] = working["ResponseCategory"].apply(map_response_label)
    working = working.dropna(subset=["Label"])
    working["Label"] = working["Label"].astype(_INT)
    working = working.rename(columns={"Treatment": "Drug"})
    return working


def ensure_models_present(models, available_index):
    models_index = pd.Index(_SORTED(_SET(models)))
    missing = models_index.difference(available_index)
    present = models_index.intersection(available_index)
    return present, missing


def main():
    args = parse_args()
    setup_logging()

    compound_bits = _INT(args.compound_bits)
    fingerprint_radius = _INT(args.fingerprint_radius)

    device = torch.device(args.device)

    LOGGER.info("Loading reference datasets")
    gdsc_gex = load_expression(args.gdsc_gex)
    pdx_gex = load_expression(args.pdx_gex)
    pdx_info = pd.read_csv(args.pdx_info)
    gdsc_rank = load_rank(args.gdsc_rank)
    curves = pd.read_csv(args.pdx_curves)

    # Align PDX genes to GDSC gene space
    gdsc_genes = gdsc_gex.columns
    common_genes = pdx_gex.columns.intersection(gdsc_genes)
    missing_genes = gdsc_genes.difference(pdx_gex.columns)

    if common_genes.size == 0:
        raise _RUNTIME_ERROR("No common genes between GDSC and PDX datasets")

    LOGGER.info(
        "Gene alignment: GDSC has %d genes, PDX has %d genes, %d common, %d missing",
        gdsc_genes.size,
        pdx_gex.shape[1],
        common_genes.size,
        missing_genes.size,
    )

    # Views for aligner (common genes) and for predictor outputs (padded gene space)
    gdsc_gex_aligner = gdsc_gex.loc[:, common_genes]
    pdx_gex_aligner = pdx_gex.loc[:, common_genes]
    pdx_gex_padded = pdx_gex.reindex(columns=gdsc_genes, fill_value=0.0)

    present_models, missing_models = ensure_models_present(curves["Model"], pdx_gex.index)
    if not missing_models.empty:
        LOGGER.warning(
            "Skipping %d models without expression: %s",
            missing_models.size,
            ", ".join(missing_models.tolist()),
        )
    if present_models.empty:
        raise _RUNTIME_ERROR("No PDX models with aligned expression available")

    curves = curves[curves["Model"].isin(present_models)]
    resp_df = build_response_table(curves)
    if resp_df.empty:
        raise _RUNTIME_ERROR("No labeled PDX entries after filtering response categories")

    LOGGER.info("Instantiating aligner")
    num_tissues = pdx_info["tissue_label"].nunique()
    aligner = instantiate_aligner(device, gdsc_gex_aligner, num_tissues, args.aligner_ckpt)

    LOGGER.info("Computing embeddings for %d models", present_models.size)
    pdx_subset_aligner = pdx_gex_aligner.loc[present_models]
    weights, perturbations = compute_model_embeddings(pdx_subset_aligner, aligner, device)
    rank_features = derive_rank_features(weights, gdsc_rank.loc[:, :])

    LOGGER.info("Preparing compound descriptors")
    smiles_lookup = build_smiles_dictionary([args.gdsc_smiles])
    unique_drugs = _SORTED(resp_df["Drug"].unique())
    compound_features = {}
    for drug in unique_drugs:
        # Try exact match first, then lowercase
        smiles = smiles_lookup.get(drug) or smiles_lookup.get(drug.lower())
        if smiles is None:
            LOGGER.warning("No SMILES found for drug '%s', using zero vector", drug)
        compound_features[drug] = smiles_to_features(
            smiles,
            n_bits=compound_bits,
            radius=fingerprint_radius,
        )

    LOGGER.info("Building aligned feature arrays")
    common_gene_list = common_genes.tolist()
    perturb_df = (
        pd.DataFrame.from_dict(perturbations, orient="index", columns=common_gene_list)
        .reindex(present_models)
    )
    perturb_df = perturb_df.reindex(columns=gdsc_genes, fill_value=0.0)

    perturbation_rows = []
    compound_rows = []
    for model, drug in resp_df[["Model", "Drug"]].itertuples(index=False):
        perturbation_rows.append(perturb_df.loc[model].to_numpy(dtype=np.float32))
        compound_rows.append(compound_features[drug])

    perturbation_array = np.asarray(perturbation_rows, dtype=np.float32).astype(np.float16)
    compound_array = np.asarray(compound_rows, dtype=np.float32).astype(np.float16)

    # Verify array dimensions match response dataframe
    assert perturbation_array.shape[0] == resp_df.shape[0], \
        f"Perturbation array length {perturbation_array.shape[0]} != response df length {resp_df.shape[0]}"
    assert compound_array.shape[0] == resp_df.shape[0], \
        f"Compound array length {compound_array.shape[0]} != response df length {resp_df.shape[0]}"

    rank_df = pd.DataFrame(rank_features).transpose().reindex(present_models)
    rank_df.index.name = "Model"

    pdx_info_indexed = pdx_info.set_index("Sample")
    labeled_info = pdx_info_indexed.loc[present_models].reset_index()
    labeled_gex = pdx_gex_padded.loc[present_models]

    resp_out = resp_df[["Model", "Drug", "ResponseCategory", "Label"]].copy()
    resp_out["Canonical_SMILES"] = [smiles_lookup.get(drug) for drug in resp_out["Drug"]]

    LOGGER.info("Saving artifacts")
    Path(args.out_resp).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rank).parent.mkdir(parents=True, exist_ok=True)

    resp_out.to_csv(args.out_resp, index=False)
    rank_df.to_csv(args.out_rank)
    np.save(args.out_perturbation, perturbation_array)
    np.save(args.out_compound, compound_array)
    labeled_gex.to_csv(args.out_labeled_gex)
    labeled_info.to_csv(args.out_labeled_info, index=False)

    LOGGER.info(
        "Wrote %d labeled pairs across %d models and %d drugs",
        resp_out.shape[0],
        present_models.size,
        resp_df["Drug"].nunique(),
    )


if __name__ == "__main__":
    main()

