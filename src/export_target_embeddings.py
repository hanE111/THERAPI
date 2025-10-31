"""Export heterogeneity-aware weights and embeddings for a target cohort using a trained aligner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import AlignerDataset, GDSC_AE, TCGA_weightencoder, Emb_Dis_classifier, Exp_Dis_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export target embeddings from THERAPI aligner")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained aligner checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda:1", help="Torch device for inference")
    parser.add_argument("--data_dir", type=str, default="../data/", help="Root directory for processed data")
    parser.add_argument("--gdsc_gex", type=str, default=None, help="Path to GDSC expression matrix (CSV)")
    parser.add_argument("--gdsc_info", type=str, default=None, help="Path to GDSC metadata CSV")
    parser.add_argument("--target_dataset", type=str, default="pdx1", help="Target dataset name (for logging)")
    parser.add_argument("--target_gex", type=str, default=None, help="Path to target expression matrix (CSV)")
    parser.add_argument("--target_info", type=str, default=None, help="Path to target metadata CSV")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for target inference")
    parser.add_argument("--output_prefix", type=str, default="../data/PDX1_align", help="Prefix for output artifacts")
    return parser.parse_args()


def load_tables(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    gdsc_gex_path = Path(args.gdsc_gex) if args.gdsc_gex else data_dir / "GDSC_gex.csv"
    gdsc_info_path = Path(args.gdsc_info) if args.gdsc_info else data_dir / "GDSC_info.csv"

    target_gex_path = Path(args.target_gex) if args.target_gex else data_dir / f"{args.target_dataset.upper()}_gex.csv"
    target_info_path = Path(args.target_info) if args.target_info else data_dir / f"{args.target_dataset.upper()}_info.csv"

    gdsc_data_df = pd.read_csv(gdsc_gex_path, index_col=0)
    gdsc_info_df = pd.read_csv(gdsc_info_path)
    target_data_df = pd.read_csv(target_gex_path, index_col=0)
    target_info_df = pd.read_csv(target_info_path)

    if gdsc_data_df.shape[0] != len(gdsc_info_df):
        raise ValueError("Mismatch between GDSC expression and metadata rows")
    if target_data_df.shape[0] != len(target_info_df):
        raise ValueError("Mismatch between target expression and metadata rows")

    # Ensure gene ordering matches GDSC encoder
    if not np.array_equal(gdsc_data_df.columns, target_data_df.columns):
        missing_genes = [gene for gene in gdsc_data_df.columns if gene not in target_data_df.columns]
        if missing_genes:
            print(f"Target dataset missing {len(missing_genes)} genes; zero-padding those columns.")
        target_data_df = target_data_df.reindex(columns=gdsc_data_df.columns, fill_value=0.0)

    combined_labels = sorted(set(gdsc_info_df['tissue_label'].tolist()) | set(target_info_df['tissue_label'].tolist()))
    label_mapping = {label: idx for idx, label in enumerate(combined_labels)}

    gdsc_info_df['tissue_label_mapped'] = gdsc_info_df['tissue_label'].map(label_mapping).astype(int)
    target_info_df['tissue_label_mapped'] = target_info_df['tissue_label'].map(label_mapping).astype(int)

    return gdsc_data_df, gdsc_info_df, target_data_df, target_info_df, len(label_mapping)


def export_embeddings(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gdsc_data_df, gdsc_info_df, target_data_df, target_info_df, num_tissue = load_tables(args)

    gdsc_dataset = AlignerDataset(gdsc_data_df, 'gdsc', gdsc_info_df['tissue_label_mapped'], domain_flag=1.0)
    target_dataset = AlignerDataset(target_data_df, args.target_dataset, target_info_df['tissue_label_mapped'], domain_flag=0.0)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    dim_latent = 128
    gdsc_AE = GDSC_AE(n_genes=gdsc_dataset.n_genes, n_classes=num_tissue, n_latent=dim_latent).to(device)
    target_weightencoder = TCGA_weightencoder(n_genes=target_dataset.n_genes, n_latent=dim_latent, n_celines=gdsc_dataset.n_samples).to(device)
    emb_dis_classifier = Emb_Dis_classifier(n_latent=dim_latent, n_classes=num_tissue).to(device)
    exp_dis_classifier = Exp_Dis_classifier(n_genes=target_dataset.n_genes, n_latent=dim_latent, n_classes=num_tissue).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    gdsc_AE.load_state_dict(checkpoint['gdsc_AE'])
    target_weightencoder.load_state_dict(checkpoint['tcga_weightencoder'])
    emb_dis_classifier.load_state_dict(checkpoint['emb_dis_classifier'])
    exp_dis_classifier.load_state_dict(checkpoint['exp_dis_classifier'])

    gdsc_AE.eval()
    target_weightencoder.eval()

    with torch.no_grad():
        gdsc_gex = gdsc_dataset.data.to(device)
        gdsc_z, _ = gdsc_AE(gdsc_gex)

    weights_list = []
    latent_list = []
    wgex_list = []
    recon_list = []

    with torch.no_grad():
        for batch_data, _, _ in target_loader:
            batch_data = batch_data.to(device)
            weights, latent, wgex, recon = target_weightencoder(batch_data, gdsc_z, gdsc_gex)
            weights_list.append(weights.cpu().numpy())
            latent_list.append(latent.cpu().numpy())
            wgex_list.append(wgex.cpu().numpy())
            recon_list.append(recon.cpu().numpy())

    target_ids = np.array(target_dataset.sample_ids)
    source_ids = np.array(gdsc_dataset.sample_ids)
    weights_arr = np.concatenate(weights_list, axis=0)
    latent_arr = np.concatenate(latent_list, axis=0)
    wgex_arr = np.concatenate(wgex_list, axis=0)
    recon_arr = np.concatenate(recon_list, axis=0)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(f"{prefix}_weights.npy", weights_arr)
    np.save(f"{prefix}_latent.npy", latent_arr)
    np.save(f"{prefix}_source_ids.npy", source_ids)
    np.save(f"{prefix}_target_ids.npy", target_ids)

    wgex_df = pd.DataFrame(wgex_arr, index=target_ids, columns=gdsc_data_df.columns)
    recon_df = pd.DataFrame(recon_arr, index=target_ids, columns=gdsc_data_df.columns)
    wgex_df.to_csv(f"{prefix}_wgex.csv")
    recon_df.to_csv(f"{prefix}_recon.csv")

    summary = {
        "target_dataset": args.target_dataset,
        "n_target": int(weights_arr.shape[0]),
        "n_source": int(weights_arr.shape[1]),
        "latent_dim": int(latent_arr.shape[1]),
        "gene_dim": int(wgex_arr.shape[1]),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
    }
    with open(f"{prefix}_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print(f"Saved weights to {prefix}_weights.npy")
    print(f"Saved latent embeddings to {prefix}_latent.npy")
    print(f"Saved weighted gene expression to {prefix}_wgex.csv")
    print(f"Saved reconstructions to {prefix}_recon.csv")
    print(f"Summary saved to {prefix}_summary.json")

if __name__ == "__main__":
    export_embeddings(parse_args())
