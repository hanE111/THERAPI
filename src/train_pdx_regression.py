#!/usr/bin/env python3
"""Train Response_predictor for PDX regression task."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from model import Response_predictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)


class PDXRegressionDataset(Dataset):
    def __init__(self, genomic, compound, targets):
        self.genomic = torch.tensor(genomic, dtype=torch.float32)
        self.compound = torch.tensor(compound, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.genomic[idx], self.compound[idx], self.targets[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for genomic, compound, targets in loader:
        genomic = genomic.to(device)
        compound = compound.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(genomic, compound)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for genomic, compound, targets in loader:
            genomic = genomic.to(device)
            compound = compound.to(device)

            predictions = model(genomic, compound)
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    return mse, mae, r2, preds, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdx-gex", default="../data/PDX_regression_gex.csv")
    parser.add_argument("--pdx-compound", default="../data/PDX_regression_compound.npy")
    parser.add_argument("--pdx-resp", default="../data/PDX_regression_resp.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-model", default="../src/ckpts/PDX_regression_predictor.pt")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load data
    LOGGER.info("Loading PDX data")
    gex = pd.read_csv(args.pdx_gex, index_col=0)
    compound = np.load(args.pdx_compound).astype(np.float32)
    resp = pd.read_csv(args.pdx_resp)

    # Verify dimensions
    assert gex.shape[0] == compound.shape[0] == resp.shape[0], \
        f"Dimension mismatch: gex {gex.shape[0]}, compound {compound.shape[0]}, resp {resp.shape[0]}"

    genomic_data = gex.values.astype(np.float32)
    targets = resp["Target"].values.astype(np.float32)

    LOGGER.info(f"Dataset size: {len(targets)} samples")
    LOGGER.info(f"Target statistics: mean={targets.mean():.4f}, std={targets.std():.4f}, min={targets.min():.4f}, max={targets.max():.4f}")

    # K-fold cross-validation
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(genomic_data), 1):
        LOGGER.info(f"\n=== Fold {fold}/{args.n_folds} ===")

        train_dataset = PDXRegressionDataset(
            genomic_data[train_idx],
            compound[train_idx],
            targets[train_idx]
        )
        val_dataset = PDXRegressionDataset(
            genomic_data[val_idx],
            compound[val_idx],
            targets[val_idx]
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = Response_predictor(
            n_genes=gex.shape[1],
            n_compound=compound.shape[1]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        best_val_mse = float('inf')
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_mse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                LOGGER.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_mse={val_mse:.4f}, val_mae={val_mae:.4f}, val_r2={val_r2:.4f}"
                )

            if patience_counter >= 20:
                LOGGER.info(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        val_mse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)

        fold_results.append({
            "fold": fold,
            "mse": val_mse,
            "mae": val_mae,
            "r2": val_r2
        })

        LOGGER.info(f"Fold {fold} - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")

    # Summary
    avg_mse = np.mean([r["mse"] for r in fold_results])
    avg_mae = np.mean([r["mae"] for r in fold_results])
    avg_r2 = np.mean([r["r2"] for r in fold_results])

    LOGGER.info(f"\n=== Cross-Validation Results ===")
    LOGGER.info(f"Average MSE: {avg_mse:.4f} ± {np.std([r['mse'] for r in fold_results]):.4f}")
    LOGGER.info(f"Average MAE: {avg_mae:.4f} ± {np.std([r['mae'] for r in fold_results]):.4f}")
    LOGGER.info(f"Average R²: {avg_r2:.4f} ± {np.std([r['r2'] for r in fold_results]):.4f}")

    # Train final model on all data
    LOGGER.info("\nTraining final model on full dataset")
    full_dataset = PDXRegressionDataset(genomic_data, compound, targets)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    final_model = Response_predictor(
        n_genes=gex.shape[1],
        n_compound=compound.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(final_model, full_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            LOGGER.info(f"Epoch {epoch}: loss={train_loss:.4f}")

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), args.out_model)
    LOGGER.info(f"Saved model to {args.out_model}")


if __name__ == "__main__":
    main()
