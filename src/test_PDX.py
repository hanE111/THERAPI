import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from model import ExpDrugDataset, Response_predictor


def get_test_results(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for emb, genef, chemical, resp in dataloader:
            emb = emb.to(device)
            genef = genef.to(device)
            chemical = chemical.to(device)
            output = model(emb, genef, chemical)
            preds.append(output.squeeze().cpu())
            targets.append(resp)
    return preds, targets


def get_test_metrics(preds, targets, cutoff=0.0):
    probs = torch.cat(preds).numpy()
    labels = (probs > cutoff).astype(int)
    targets_np = torch.cat(targets).numpy()

    auc = roc_auc_score(targets_np, probs)
    auprc = average_precision_score(targets_np, probs)
    acc = accuracy_score(targets_np, labels)
    precision = precision_score(targets_np, labels)
    f1 = f1_score(targets_np, labels)

    return auc, auprc, acc, precision, f1


def test_PDX(args):
    batch_size = 512
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 1
    folds = 10

    resp_path = os.path.join(args.data_dir, "PDX_resp.csv")
    rank_path = os.path.join(args.data_dir, "PDX_rankrepresentation.csv")
    pert_path = os.path.join(args.data_dir, "PDX_perturbation.npy")
    comp_path = os.path.join(args.data_dir, "PDX_perturbation_compound.npy")

    resp_df = pd.read_csv(resp_path)
    rank_df = pd.read_csv(rank_path, index_col=0)
    rank_df = rank_df.reindex(resp_df["Model"])
    if rank_df.isnull().values.any():
        missing_models = rank_df[rank_df.isnull().any(axis=1)].index.unique().tolist()
        logging.warning(
            "Rank features missing for %d models; filling with zeros (showing up to 5): %s",
            len(missing_models),
            missing_models[:5],
        )
        rank_df = rank_df.fillna(0.0)
    rank_np = rank_df.to_numpy(dtype=np.float32)
    pert_np = np.load(pert_path)
    comp_np = np.load(comp_path)

    # Align perturbation features to the GDSC gene space expected by the predictor checkpoints
    gdsc_gex_path = os.path.join(args.data_dir, "GDSC_gex.csv")
    pdx_gex_path = os.path.join(args.data_dir, "PDX_labeled_gex.csv")

    if os.path.exists(gdsc_gex_path) and os.path.exists(pdx_gex_path):
        gdsc_genes = pd.read_csv(gdsc_gex_path, nrows=0).columns[1:]
        pdx_genes = pd.read_csv(pdx_gex_path, index_col=0, nrows=0).columns

        missing_from_pdX = [gene for gene in gdsc_genes if gene not in pdx_genes]
        pert_df = pd.DataFrame(pert_np, columns=pdx_genes)
        pert_df = pert_df.reindex(columns=gdsc_genes, fill_value=0.0)
        pert_np = pert_df.to_numpy(dtype=np.float32)

        if missing_from_pdX:
            logging.warning(
                "Padding %d genes absent in PDX with zeros to match predictor checkpoints",
                len(missing_from_pdX),
            )

    label_col = "Label"
    dataset = ExpDrugDataset(
        pert_np,
        rank_np,
        comp_np,
        resp_df[label_col].values,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    models = [
        Response_predictor(
            dataset.emb_dim,
            dataset.genef_dim,
            dataset.chemical_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim,
        ).to(args.device)
        for _ in range(folds)
    ]

    model_names = [f"{args.model_name}_CV{fold}" for fold in range(folds)]

    for idx, model in enumerate(models):
        ckpt_path = os.path.join("ckpts", f"{model_names[idx]}.pt")
        state_dict = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(state_dict)

    pdx_metrics = []
    for fold_idx, model in enumerate(models):
        preds, targets = get_test_results(model, dataloader, args.device)
        pdx_metrics.append(get_test_metrics(preds, targets, cutoff=args.cutoff))

    metrics_df = pd.DataFrame(pdx_metrics, columns=["AUC", "AUPRC", "Accuracy", "Precision", "F1"])
    metrics_df.loc["mean"] = metrics_df.mean()
    metrics_df.loc["std"] = metrics_df.std()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "THERAPI_test_PDX.csv")
    metrics_df.round(4).to_csv(output_path, index=False)

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--model_name", type=str, default="THERAPI_predictor")
    parser.add_argument("--output_dir", type=str, default="../output/")
    parser.add_argument("--cutoff", type=float, default=0.0)
    args = parser.parse_args()

    test_PDX(args)
