"""
Training script for Hierarchical THERAPI.

Trains the unified tissue-aware model using TRANSACT's data.
Includes specialized losses for tissue routing and hierarchical attention.
"""

import os
import sys
import argparse
import yaml
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import original THERAPI utilities from src
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
from utils import set_seed, Logger, EarlyStopper
from center_loss import CenterLoss
sys.path.pop(0)  # Remove src from path

# Import hierarchical components
from hierarchical_utils.data_loader import TransactDataLoader
from hierarchical_utils.tissue_mapping import TissueMapper
from models.hierarchical_therapi import HierarchicalTHERAPI


class HierarchicalAlignerDataset(Dataset):
    """Dataset for hierarchical alignment training."""

    def __init__(self, data_df, tissue_labels, domain_label):
        self.data = torch.tensor(data_df.values, dtype=torch.float32)
        self.n_genes = self.data.shape[1]
        self.n_samples = self.data.shape[0]

        # Domain label (0 for target, 1 for source)
        if domain_label.lower() == 'gdsc':
            self.domain_label = torch.ones(self.n_samples, dtype=torch.float32)
        else:
            self.domain_label = torch.zeros(self.n_samples, dtype=torch.float32)

        # Tissue labels
        self.tissue_label = torch.tensor(tissue_labels, dtype=torch.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.domain_label[idx], self.tissue_label[idx]


def compute_hierarchical_losses(model, patient_expr, cell_line_exprs,
                                tissue_labels, tissue_weights,
                                center_criterion, config):
    """
    Compute all losses for hierarchical training.

    Args:
        model: HierarchicalTHERAPI model
        patient_expr: Patient gene expression
        cell_line_exprs: Cell line expressions
        tissue_labels: True tissue labels
        tissue_weights: Predicted tissue weights
        center_criterion: CenterLoss for tissue clustering
        config: Configuration dict

    Returns:
        Dictionary of losses
    """
    batch_size = patient_expr.size(0)

    # Get hierarchical representation
    output = model.hierarchical_attention_forward(patient_expr, cell_line_exprs)
    representation = output['representation']
    tissue_weights = output['tissue_weights']

    # 1. Reconstruction loss (optional - can add decoder)
    # For now, we focus on alignment losses

    # 2. Center loss for tissue clustering
    center_loss = center_criterion(representation, tissue_labels)

    # 3. Tissue routing consistency loss
    # Encourage routing to match true tissue
    tissue_routing_loss = F.cross_entropy(tissue_weights, tissue_labels)

    # 4. Entropy regularization (encourage decisive routing)
    epsilon = 1e-8
    entropy = -torch.sum(tissue_weights * torch.log(tissue_weights + epsilon), dim=-1)
    entropy_loss = entropy.mean()

    # 5. Diversity loss (different patients should use different tissues in batch)
    batch_tissue_dist = tissue_weights.mean(dim=0)
    diversity = -torch.sum(batch_tissue_dist * torch.log(batch_tissue_dist + epsilon))
    diversity_loss = -diversity  # Negative because we want to maximize diversity

    # 6. Temperature regularization (anneal towards target value)
    target_temp = config.get('target_temperature', 0.5)
    temp_loss = F.mse_loss(model.tissue_router.temperature, torch.tensor(target_temp).to(model.tissue_router.temperature.device))

    # Combine losses
    total_loss = (
        config['loss_weights']['center'] * center_loss +
        config['loss_weights']['routing'] * tissue_routing_loss +
        config['loss_weights']['entropy'] * entropy_loss +
        config['loss_weights']['diversity'] * diversity_loss +
        config['loss_weights']['temperature'] * temp_loss
    )

    return {
        'total': total_loss,
        'center': center_loss,
        'routing': tissue_routing_loss,
        'entropy': entropy_loss,
        'diversity': diversity_loss,
        'temperature': temp_loss
    }


def train_hierarchical_aligner(args, config):
    """
    Train hierarchical aligner (Step 1 of Hierarchical THERAPI).
    """
    model_name = f'HierarchicalTHERAPI_aligner_{args.source}_{args.target}'
    logger = Logger(model_name)
    logger(f'Start training {model_name} model')
    set_seed(args.seed, logger)

    # Load data using TRANSACT loader
    data_loader = TransactDataLoader(args.data_dir)
    tissue_mapper = TissueMapper()

    # Load source and target data
    if args.source == 'GDSC':
        source_data = data_loader.load_gdsc_data()
        source_expr = source_data['expression']
        source_tissues = source_data['tissue_mapping']
    else:
        raise ValueError(f"Unknown source: {args.source}")

    if args.target == 'TCGA':
        target_data = data_loader.load_tcga_data()
        target_expr = target_data['expression']
        # Get tissue info from tissue_info
        if 'tissue_info' in target_data and not target_data['tissue_info'].empty:
            tissue_info = target_data['tissue_info']
            # Assume first column is sample ID, second is tissue
            target_tissues = dict(zip(tissue_info.iloc[:, 0], tissue_info.iloc[:, 1]))
        else:
            logger("Warning: No tissue info found for target, using 'other'")
            target_tissues = {idx: 'other' for idx in target_expr.index}
    elif args.target == 'External':
        # Load external data (PDX or other)
        target_data = data_loader.load_pdx_data()
        target_expr = target_data['expression']
        if target_data['biospecimen'] is not None:
            biospecimen = target_data['biospecimen']
            target_tissues = dict(zip(biospecimen.iloc[:, 0], biospecimen.iloc[:, 1]))
        else:
            target_tissues = {idx: 'other' for idx in target_expr.index}
    else:
        raise ValueError(f"Unknown target: {args.target}")

    # Harmonize genes
    common_genes = data_loader.harmonize_genes(source_expr, target_expr)
    logger(f'Using {len(common_genes)} common genes')

    source_expr = source_expr[common_genes]
    target_expr = target_expr[common_genes]

    # Create tissue labels
    source_tissue_labels = [tissue_mapper.get_tissue_idx(source_tissues.get(idx, 'other'))
                           for idx in source_expr.index]
    target_tissue_labels = [tissue_mapper.get_tissue_idx(target_tissues.get(idx, 'other'))
                           for idx in target_expr.index]

    # Create tissue-cell line mask
    tissue_cell_mask, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(source_tissues)
    tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)

    # Create datasets
    source_dataset = HierarchicalAlignerDataset(source_expr, source_tissue_labels, args.source)
    target_dataset = HierarchicalAlignerDataset(target_expr, target_tissue_labels, args.target)

    # Dataloaders
    batch_size = config.get('batch_size', 128)
    target_dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Initialize model
    model = HierarchicalTHERAPI(
        n_genes=len(common_genes),
        n_tissues=tissue_mapper.n_tissue_groups,
        n_latent=config.get('latent_dim', 128),
        tissue_cell_mask=tissue_cell_mask_tensor,
        routing_strategy=config.get('routing_strategy', 'gumbel')
    ).to(args.device)

    # Loss functions
    center_criterion = CenterLoss(
        num_classes=tissue_mapper.n_tissue_groups,
        feat_dim=config.get('latent_dim', 128),
        device=args.device
    )

    # Optimizer
    lr = config.get('learning_rate', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create checkpoint directory
    os.makedirs('ckpts', exist_ok=True)

    # Training loop
    n_epochs = config.get('n_epochs', 200)
    source_data_tensor = source_dataset.data.to(args.device)

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {
            'total': 0, 'center': 0, 'routing': 0,
            'entropy': 0, 'diversity': 0, 'temperature': 0
        }

        for target_gex, _, target_tissue in target_dataloader:
            target_gex = target_gex.to(args.device)
            target_tissue = target_tissue.to(args.device)

            # Compute losses
            losses = compute_hierarchical_losses(
                model, target_gex, source_data_tensor,
                target_tissue, None, center_criterion, config
            )

            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

        # Average losses
        n_batches = len(target_dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger(
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Total: {epoch_losses['total']:.4f} | "
                f"Center: {epoch_losses['center']:.4f} | "
                f"Routing: {epoch_losses['routing']:.4f} | "
                f"Entropy: {epoch_losses['entropy']:.4f} | "
                f"Temp: {model.tissue_router.temperature.item():.4f}"
            )

    # Save model
    save_path = f'ckpts/{model_name}.pt'
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'tissue_mapper': tissue_mapper,
        'common_genes': common_genes
    }, save_path)

    logger(f'Model saved to {save_path}')
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical THERAPI Aligner')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--data_dir', type=str, default='data/', help='Data directory')
    parser.add_argument('--source', type=str, default='GDSC', help='Source domain')
    parser.add_argument('--target', type=str, default='TCGA', help='Target domain')
    parser.add_argument('--config', type=str, default='configs/hierarchical_config.yaml',
                       help='Config file path')

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'batch_size': 128,
            'latent_dim': 128,
            'learning_rate': 1e-3,
            'n_epochs': 200,
            'routing_strategy': 'gumbel',
            'target_temperature': 0.5,
            'loss_weights': {
                'center': 0.8,
                'routing': 0.4,
                'entropy': 0.2,
                'diversity': 0.1,
                'temperature': 0.1
            }
        }
        print(f"Config file not found at {args.config}, using defaults")

    # Train
    model = train_hierarchical_aligner(args, config)

    print(f"\nTraining completed successfully!")


if __name__ == '__main__':
    main()
