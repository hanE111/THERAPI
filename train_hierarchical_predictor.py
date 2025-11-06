"""
Training script for Hierarchical THERAPI Predictor (Step 2).

This trains the drug response predictor using representations from the
hierarchical aligner trained in Step 1.
"""

import os
import sys
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import original THERAPI utilities from src
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
from utils import set_seed, Logger, EarlyStopper
sys.path.pop(0)  # Remove src from path

# Import hierarchical components
from hierarchical_utils.data_loader import TransactDataLoader
from hierarchical_utils.tissue_mapping import TissueMapper
from models.hierarchical_therapi import HierarchicalTHERAPI, HierarchicalResponsePredictor


class HierarchicalDrugDataset(Dataset):
    """Dataset for drug response prediction with hierarchical representations."""

    def __init__(self, patient_repr, genef_list, chemical_list, resp_list):
        """
        Args:
            patient_repr: Pre-computed patient representations from aligner
            genef_list: Gene features (perturbation/rank representation)
            chemical_list: Drug chemical features
            resp_list: Response labels
        """
        self.patient_repr = patient_repr
        self.genef = genef_list
        self.chemical = chemical_list
        self.resp = torch.tensor(resp_list, dtype=torch.float32)
        self.n_samples = len(resp_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        patient_emb = torch.tensor(self.patient_repr[idx], dtype=torch.float32)
        genef = torch.tensor(self.genef[idx], dtype=torch.float32)
        chemical = torch.tensor(self.chemical[idx], dtype=torch.float32)
        resp = self.resp[idx]

        return patient_emb, genef, chemical, resp


def compute_patient_representations(aligner, gdsc_expr, gdsc_cell_exprs, device):
    """
    Compute patient representations using trained aligner.

    Args:
        aligner: Trained hierarchical aligner
        gdsc_expr: GDSC expression data
        gdsc_cell_exprs: GDSC cell line expressions (for attention)
        device: torch device

    Returns:
        Array of patient representations [n_samples, latent_dim]
    """
    aligner.eval()
    representations = []

    batch_size = 64
    gdsc_tensor = torch.tensor(gdsc_expr.values, dtype=torch.float32).to(device)
    cell_tensor = torch.tensor(gdsc_cell_exprs.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i in range(0, len(gdsc_expr), batch_size):
            batch = gdsc_tensor[i:i+batch_size]
            output = aligner.hierarchical_attention_forward(batch, cell_tensor)
            representations.append(output['representation'].cpu().numpy())

    return np.vstack(representations)


def train_hierarchical_predictor(args, config):
    """
    Train drug response predictor using hierarchical representations.
    """
    model_name = 'HierarchicalTHERAPI_predictor'
    logger = Logger(model_name)
    logger(f'Start training {model_name}')
    set_seed(args.seed, logger)

    # Load data
    data_loader = TransactDataLoader(args.data_dir)
    gdsc_data = data_loader.load_gdsc_data()

    # Load trained aligner
    logger(f'Loading aligner from {args.aligner_path}')
    aligner_checkpoint = torch.load(args.aligner_path, map_location=args.device)
    tissue_mapper = aligner_checkpoint.get('tissue_mapper', TissueMapper())
    common_genes = aligner_checkpoint.get('common_genes', None)
    aligner_config = aligner_checkpoint.get('config', {})

    # Prepare cell line data
    if common_genes:
        gdsc_expr = gdsc_data['expression'][common_genes]
    else:
        gdsc_expr = gdsc_data['expression']

    # Initialize aligner - filter tissue mapping to match expression data
    source_tissues_filtered = {idx: gdsc_data['tissue_mapping'][idx]
                              for idx in gdsc_expr.index
                              if idx in gdsc_data['tissue_mapping']}
    tissue_cell_mask, _ = tissue_mapper.create_cell_line_tissue_matrix(source_tissues_filtered)
    tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)

    aligner = HierarchicalTHERAPI(
        n_genes=len(common_genes) if common_genes else gdsc_expr.shape[1],
        n_tissues=tissue_mapper.n_tissue_groups,
        n_latent=aligner_config.get('latent_dim', 128),
        tissue_cell_mask=tissue_cell_mask_tensor
    ).to(args.device)

    aligner.load_state_dict(aligner_checkpoint['model_state_dict'])
    logger('Aligner loaded successfully')

    # Load drug response data
    # For TRANSACT format
    resp_path = os.path.join(args.data_dir, 'GDSC/response/GDSC1_fitted_dose_response_27Oct23.xlsx')
    if os.path.exists(resp_path):
        drug_response = pd.read_excel(resp_path)
        logger(f"Loaded TRANSACT GDSC1 format: {drug_response.shape}")

        # TRANSACT format has: CELL_LINE_NAME, DRUG_NAME, AUC, LN_IC50, etc.
        # Convert to binary response: sensitive (AUC < 0.8) vs resistant (AUC >= 0.8)
        # Or use continuous AUC values directly
        if 'AUC' in drug_response.columns:
            # Binarize AUC: lower AUC = more sensitive (label=1), higher AUC = resistant (label=0)
            drug_response['Label'] = (drug_response['AUC'] < 0.8).astype(int)
            logger(f"Created binary labels from AUC (threshold=0.8): {drug_response['Label'].value_counts().to_dict()}")
        elif 'LN_IC50' in drug_response.columns:
            # Use LN_IC50: lower value = more sensitive
            drug_response['Label'] = (drug_response['LN_IC50'] < drug_response['LN_IC50'].median()).astype(int)
            logger(f"Created binary labels from LN_IC50 (median split)")
    else:
        # Fallback to original THERAPI format
        resp_path = os.path.join(args.data_dir, 'GDSC/GDSC_Drug_SMILES_Response.csv')
        if os.path.exists(resp_path):
            drug_response = pd.read_csv(resp_path)
            logger(f"Loaded original THERAPI format: {drug_response.shape}")
        else:
            raise FileNotFoundError("No drug response data found")

    # Load perturbation and chemical features
    pert_path = os.path.join(args.data_dir, 'GDSC/GDSC_perturbation_float16.npy')
    comp_path = os.path.join(args.data_dir, 'GDSC/GDSC_perturbation_compound_float16.npy')
    rank_path = os.path.join(args.data_dir, 'GDSC/GDSC_rankrepresentation.csv')

    if os.path.exists(pert_path):
        gdsc_pert = np.load(pert_path).astype(np.float32)
    else:
        logger("Warning: Perturbation features not found, using zeros")
        gdsc_pert = np.zeros((len(drug_response), gdsc_expr.shape[1]), dtype=np.float32)

    if os.path.exists(comp_path):
        gdsc_comp = np.load(comp_path).astype(np.float32)
    else:
        logger("Warning: Chemical features not found, using zeros")
        gdsc_comp = np.zeros((len(drug_response), 2048), dtype=np.float32)

    if os.path.exists(rank_path):
        gdsc_rank = pd.read_csv(rank_path, index_col=0)
        if 'ID' in drug_response.columns:
            gdsc_rank = gdsc_rank.loc[drug_response['ID']].values
        else:
            gdsc_rank = gdsc_rank.values
    else:
        logger("Warning: Rank representation not found, using zeros")
        gdsc_rank = np.zeros((len(drug_response), 100), dtype=np.float32)

    # Get response labels
    if 'Label' in drug_response.columns:
        labels = drug_response['Label'].values
    elif 'response' in drug_response.columns:
        labels = drug_response['response'].values
    else:
        raise ValueError("No response label column found")

    # Compute patient representations using aligner
    logger('Computing patient representations...')
    patient_representations = compute_patient_representations(
        aligner, gdsc_expr, gdsc_expr, args.device
    )
    logger(f'Computed representations shape: {patient_representations.shape}')

    # Map cell lines to their representations
    # Create a mapping from cell line names to representation indices
    cell_line_to_repr_idx = {cell_line: idx for idx, cell_line in enumerate(gdsc_expr.index)}

    # For each drug response entry, find the corresponding cell line representation
    if 'CELL_LINE_NAME' in drug_response.columns:
        cell_line_col = 'CELL_LINE_NAME'
    elif 'SANGER_MODEL_ID' in drug_response.columns:
        cell_line_col = 'SANGER_MODEL_ID'
    elif 'COSMIC_ID' in drug_response.columns:
        cell_line_col = 'COSMIC_ID'
    else:
        logger("Warning: No cell line identifier found, assuming order matches")
        cell_line_indices = np.arange(len(drug_response)) % len(patient_representations)

    if cell_line_col in drug_response.columns:
        # Map each drug response to its cell line representation
        cell_line_indices = []
        missing_count = 0
        for cell_line in drug_response[cell_line_col]:
            if cell_line in cell_line_to_repr_idx:
                cell_line_indices.append(cell_line_to_repr_idx[cell_line])
            else:
                # If cell line not found, use first representation (fallback)
                cell_line_indices.append(0)
                missing_count += 1

        cell_line_indices = np.array(cell_line_indices)
        if missing_count > 0:
            logger(f"Warning: {missing_count}/{len(drug_response)} cell lines not found in expression data")
        logger(f"Mapped {len(drug_response)} drug responses to {len(np.unique(cell_line_indices))} unique cell lines")

    # Expand patient representations to match drug response data
    # Each drug response gets the representation of its corresponding cell line
    aligned_patient_repr = patient_representations[cell_line_indices]
    logger(f'Aligned representations shape: {aligned_patient_repr.shape}')

    # Create dataset
    dataset = HierarchicalDrugDataset(
        aligned_patient_repr,
        gdsc_rank,
        gdsc_comp,
        labels
    )

    # Load fold indices if available
    fold_dir = os.path.join(args.data_dir, 'GDSC/GDSC_split')
    if os.path.exists(fold_dir):
        n_folds = 10
        logger(f'Using {n_folds}-fold cross-validation')
    else:
        n_folds = 1
        logger('No fold split found, using single training')

    # Training parameters
    batch_size = config.get('batch_size', 512)
    lr = config.get('learning_rate', 1e-3)
    hidden_dim1 = config['predictor'].get('hidden_dim1', 256)
    hidden_dim2 = config['predictor'].get('hidden_dim2', 128)

    os.makedirs('ckpts', exist_ok=True)

    # Train models
    for fold in range(n_folds):
        fold_model_name = f'{model_name}_CV{fold}'
        fold_logger = Logger(fold_model_name)
        fold_logger(f'Training fold {fold+1}/{n_folds}')

        # Load fold indices
        if n_folds > 1:
            fold_path = os.path.join(fold_dir, f'fold_{fold}_indices.pkl')
            with open(fold_path, 'rb') as f:
                train_idx, valid_idx, test_idx = pickle.load(f)

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            valid_dataset = torch.utils.data.Subset(dataset, valid_idx)
        else:
            # Simple 80/20 split
            n_samples = len(dataset)
            train_size = int(0.8 * n_samples)
            train_idx = list(range(train_size))
            valid_idx = list(range(train_size, n_samples))

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed)
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Initialize predictor
        predictor = HierarchicalResponsePredictor(
            emb_dim=aligner_config.get('latent_dim', 128),
            genef_dim=gdsc_rank.shape[1],
            chemical_dim=gdsc_comp.shape[1],
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            output_dim=1,
            dropout=config['predictor'].get('dropout', 0.1)
        ).to(args.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        early_stopper = EarlyStopper(
            patience=10,
            path=f'ckpts/{fold_model_name}.pt',
            verbose=True
        )

        # Training loop
        epoch = 0
        while True:
            epoch += 1
            predictor.train()
            train_loss = 0

            for patient_emb, genef, chemical, resp in train_dataloader:
                patient_emb = patient_emb.to(args.device)
                genef = genef.to(args.device)
                chemical = chemical.to(args.device)
                resp = resp.to(args.device)

                optimizer.zero_grad()
                output = predictor(patient_emb, genef, chemical)
                loss = criterion(output.squeeze(), resp)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            # Validation
            predictor.eval()
            valid_loss = 0
            with torch.no_grad():
                for patient_emb, genef, chemical, resp in valid_dataloader:
                    patient_emb = patient_emb.to(args.device)
                    genef = genef.to(args.device)
                    chemical = chemical.to(args.device)
                    resp = resp.to(args.device)

                    output = predictor(patient_emb, genef, chemical)
                    loss = criterion(output.squeeze(), resp)
                    valid_loss += loss.item()

            valid_loss /= len(valid_dataloader)

            fold_logger(f'Epoch {epoch}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}')

            if early_stopper(valid_loss, predictor):
                break

        fold_logger(f'Fold {fold+1} training completed')

    logger('All folds training completed!')


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical THERAPI Predictor')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--aligner_path', type=str, required=True,
                       help='Path to trained aligner model')
    parser.add_argument('--config', type=str, default='configs/hierarchical_config.yaml')

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_raw = yaml.safe_load(f)

        # Flatten nested config for easier access
        config = {
            'batch_size': config_raw.get('training', {}).get('batch_size', 512),
            'learning_rate': config_raw.get('training', {}).get('learning_rate', 1e-3),
            'predictor': config_raw.get('predictor', {
                'hidden_dim1': 256,
                'hidden_dim2': 128,
                'dropout': 0.1
            })
        }
    else:
        config = {
            'batch_size': 512,
            'learning_rate': 1e-3,
            'predictor': {
                'hidden_dim1': 256,
                'hidden_dim2': 128,
                'dropout': 0.1
            }
        }

    train_hierarchical_predictor(args, config)


if __name__ == '__main__':
    main()
