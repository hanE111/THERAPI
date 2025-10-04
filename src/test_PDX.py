import os
import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

from model import Response_predictor


class PDXRegressionDataset(Dataset):
    """Dataset for PDX regression using genomic + chemical embeddings only."""
    def __init__(self, genomic, compound, targets):
        self.genomic = torch.tensor(genomic, dtype=torch.float32)
        self.compound = torch.tensor(compound, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.genomic[idx], self.compound[idx], self.targets[idx]


def get_test_results(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for genomic, chemical, resp in dataloader:
            genomic = genomic.to(device)
            chemical = chemical.to(device)
            output = model(genomic, chemical)
            preds.append(output.squeeze().cpu())
            targets.append(resp.squeeze())
    return preds, targets


def get_test_metrics(preds, targets):
    """Calculate regression metrics."""
    preds_np = torch.cat(preds).numpy()
    targets_np = torch.cat(targets).numpy()

    mse = mean_squared_error(targets_np, preds_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_np, preds_np)
    r2 = r2_score(targets_np, preds_np)

    # Pearson and Spearman correlations
    pearson_r, pearson_p = scipy.stats.pearsonr(targets_np, preds_np)
    spearman_r, spearman_p = scipy.stats.spearmanr(targets_np, preds_np)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_r": spearman_r,
        "Spearman_p": spearman_p,
    }


def test_PDX(args):
    batch_size = 512

    resp_path = os.path.join(args.data_dir, "PDX_regression_resp.csv")
    gex_path = os.path.join(args.data_dir, "PDX_regression_gex.csv")
    comp_path = os.path.join(args.data_dir, "PDX_regression_compound.npy")

    resp_df = pd.read_csv(resp_path)
    gex_df = pd.read_csv(gex_path, index_col=0)
    comp_np = np.load(comp_path)

    # Verify dimensions match
    assert gex_df.shape[0] == comp_np.shape[0] == resp_df.shape[0], \
        f"Dimension mismatch: gex {gex_df.shape[0]}, compound {comp_np.shape[0]}, resp {resp_df.shape[0]}"

    genomic_data = gex_df.values.astype(np.float32)
    targets = resp_df["Target"].values.astype(np.float32)

    logging.info(f"Loaded {len(targets)} samples for testing")
    logging.info(f"Genomic features: {genomic_data.shape[1]}, Compound features: {comp_np.shape[1]}")

    dataset = PDXRegressionDataset(genomic_data, comp_np, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load single trained model
    model = Response_predictor(
        n_genes=gex_df.shape[1],
        n_compound=comp_np.shape[1]
    ).to(args.device)

    ckpt_path = os.path.join("ckpts", f"{args.model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    logging.info(f"Loaded model from {ckpt_path}")

    # Evaluate
    preds, targets = get_test_results(model, dataloader, args.device)
    metrics = get_test_metrics(preds, targets)

    logging.info("PDX Regression Test Results:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "THERAPI_test_PDX_regression.csv")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.round(4).to_csv(output_path, index=False)
    logging.info(f"Saved results to {output_path}")

    return metrics_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--model_name", type=str, default="PDX_regression_predictor")
    parser.add_argument("--output_dir", type=str, default="../output/")
    args = parser.parse_args()

    test_PDX(args)
