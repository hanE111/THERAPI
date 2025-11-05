"""
Test Hierarchical THERAPI on TCGA data (similar to original test_TCGA.py).

This script:
1. Loads trained aligner and predictor models
2. Computes hierarchical representations for TCGA patients
3. Predicts drug responses
4. Evaluates performance with standard metrics
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score

# Import from src
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
from model import ExpDrugDataset
sys.path.remove(src_path)

# Import hierarchical components
from utils.data_loader import TransactDataLoader
from utils.tissue_mapping import TissueMapper
from models.hierarchical_therapi import HierarchicalTHERAPI, HierarchicalResponsePredictor


def compute_tcga_representations(aligner, tcga_expr, gdsc_cell_exprs, device, batch_size=64):
    """
    Compute TCGA patient representations using trained aligner.

    Args:
        aligner: Trained hierarchical aligner
        tcga_expr: TCGA expression data
        gdsc_cell_exprs: GDSC cell line expressions
        device: torch device
        batch_size: Batch size for processing

    Returns:
        Array of patient representations [n_patients, latent_dim]
    """
    aligner.eval()
    representations = []

    tcga_tensor = torch.tensor(tcga_expr.values, dtype=torch.float32)
    cell_tensor = torch.tensor(gdsc_cell_exprs.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i in range(0, len(tcga_expr), batch_size):
            batch = tcga_tensor[i:i+batch_size].to(device)
            output = aligner.hierarchical_attention_forward(batch, cell_tensor)
            representations.append(output['representation'].cpu().numpy())

    return np.vstack(representations)


def get_test_results(predictor, test_dataloader, device):
    """Get predictions from predictor."""
    predictor.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for emb, genef, chemical, resp in test_dataloader:
            emb = emb.to(device)
            genef = genef.to(device)
            chemical = chemical.to(device)

            output = predictor(emb, genef, chemical)
            preds.append(output.squeeze().cpu())
            targets.append(resp)

    return preds, targets


def get_test_metrics_withcutoff(preds, targets, cutoff=0.0):
    """Compute metrics with specified cutoff."""
    probs = torch.cat(preds).numpy()
    preds_binary = (probs > cutoff).astype(int)
    targets = torch.cat(targets).numpy()

    auc = roc_auc_score(targets, probs)
    auprc = average_precision_score(targets, probs)
    acc = accuracy_score(targets, preds_binary)
    precision = precision_score(targets, preds_binary)
    f1 = f1_score(targets, preds_binary)

    return auc, auprc, acc, precision, f1


def test_hierarchical_tcga(args):
    """
    Test Hierarchical THERAPI on TCGA data.
    """
    print(f"Testing Hierarchical THERAPI on TCGA")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    data_loader = TransactDataLoader(args.data_dir)
    tcga_data = data_loader.load_tcga_data()
    gdsc_data = data_loader.load_gdsc_data()

    # Load aligner
    print(f"\nLoading aligner from {args.aligner_path}")
    aligner_checkpoint = torch.load(args.aligner_path, map_location=args.device)
    tissue_mapper = aligner_checkpoint.get('tissue_mapper', TissueMapper())
    common_genes = aligner_checkpoint.get('common_genes', None)
    aligner_config = aligner_checkpoint.get('config', {})

    # Prepare data with common genes
    if common_genes:
        tcga_expr = tcga_data['expression'][common_genes]
        gdsc_expr = gdsc_data['expression'][common_genes]
    else:
        tcga_expr = tcga_data['expression']
        gdsc_expr = gdsc_data['expression']

    print(f"TCGA samples: {tcga_expr.shape[0]}")
    print(f"GDSC cell lines: {gdsc_expr.shape[0]}")
    print(f"Genes: {tcga_expr.shape[1]}")

    # Initialize aligner
    tissue_cell_mask, _ = tissue_mapper.create_cell_line_tissue_matrix(
        gdsc_data['tissue_mapping']
    )
    tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)

    aligner = HierarchicalTHERAPI(
        n_genes=tcga_expr.shape[1],
        n_tissues=tissue_mapper.n_tissue_groups,
        n_latent=aligner_config.get('latent_dim', 128),
        tissue_cell_mask=tissue_cell_mask_tensor
    ).to(args.device)

    aligner.load_state_dict(aligner_checkpoint['model_state_dict'])
    print("Aligner loaded successfully")

    # Compute TCGA representations
    print("\nComputing TCGA patient representations...")
    tcga_representations = compute_tcga_representations(
        aligner, tcga_expr, gdsc_expr, args.device
    )
    print(f"Representations shape: {tcga_representations.shape}")

    # Load drug response data
    # Try TRANSACT format first
    resp_path = os.path.join(args.data_dir, 'TCGA/response/response.csv')
    if os.path.exists(resp_path):
        tcga_resp = pd.read_csv(resp_path)
    else:
        # Fallback to THERAPI format
        resp_path = os.path.join(args.data_dir, 'TCGA/TCGA_Drug_SMILES_Response.csv')
        if os.path.exists(resp_path):
            tcga_resp = pd.read_csv(resp_path)
        else:
            print("Error: No TCGA drug response data found")
            return None

    print(f"Drug response records: {len(tcga_resp)}")

    # Load features
    rank_path = os.path.join(args.data_dir, 'TCGA/TCGA_rankrepresentation.csv')
    pert_path = os.path.join(args.data_dir, 'TCGA/TCGA_perturbation_float16.npy')
    comp_path = os.path.join(args.data_dir, 'TCGA/TCGA_perturbation_compound_float16.npy')

    if os.path.exists(rank_path):
        tcga_rank = pd.read_csv(rank_path, index_col=0).values
    else:
        print("Warning: Rank representation not found")
        tcga_rank = np.zeros((len(tcga_resp), 100))

    if os.path.exists(pert_path):
        tcga_pert = np.load(pert_path).astype(np.float32)
    else:
        print("Warning: Perturbation not found, using representations")
        tcga_pert = tcga_representations

    if os.path.exists(comp_path):
        tcga_comp = np.load(comp_path).astype(np.float32)
    else:
        print("Warning: Chemical features not found")
        tcga_comp = np.zeros((len(tcga_resp), 2048))

    # Get labels
    if 'Label' in tcga_resp.columns:
        labels = tcga_resp['Label'].values
    elif 'response' in tcga_resp.columns:
        labels = tcga_resp['response'].values
    else:
        print("Error: No response label column found")
        return None

    # Create dataset
    tcga_dataset = ExpDrugDataset(tcga_pert, tcga_rank, tcga_comp, labels)
    tcga_dataloader = DataLoader(tcga_dataset, batch_size=512, shuffle=False)

    # Load predictors (all folds)
    n_folds = args.n_folds
    predictors = []

    print(f"\nLoading {n_folds} predictor models...")
    for fold in range(n_folds):
        predictor_path = f'ckpts/HierarchicalTHERAPI_predictor_CV{fold}.pt'

        if not os.path.exists(predictor_path):
            print(f"Warning: Predictor {fold} not found at {predictor_path}")
            continue

        predictor = HierarchicalResponsePredictor(
            emb_dim=aligner_config.get('latent_dim', 128),
            genef_dim=tcga_rank.shape[1],
            chemical_dim=tcga_comp.shape[1],
            hidden_dim1=256,
            hidden_dim2=128,
            output_dim=1
        ).to(args.device)

        predictor.load_state_dict(torch.load(predictor_path, map_location=args.device))
        predictors.append(predictor)

    print(f"Loaded {len(predictors)} predictors")

    if len(predictors) == 0:
        print("Error: No predictor models found!")
        print("Please train predictors first using train_hierarchical_predictor.py")
        return None

    # Evaluate each fold
    print("\nEvaluating on TCGA...")
    results = []

    for i, predictor in enumerate(predictors):
        preds, targets = get_test_results(predictor, tcga_dataloader, args.device)
        metrics = get_test_metrics_withcutoff(preds, targets, cutoff=0.0)
        results.append(metrics)
        print(f"Fold {i}: AUC={metrics[0]:.4f}, AUPRC={metrics[1]:.4f}")

    # Aggregate results
    results_df = pd.DataFrame(
        results,
        columns=['AUC', 'AUPRC', 'Accuracy', 'Precision', 'F1']
    )

    means = results_df.mean()
    stds = results_df.std()

    results_df.loc['mean'] = means
    results_df.loc['std'] = stds

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.round(4))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'HierarchicalTHERAPI_test_TCGA.csv')
    results_df.round(4).to_csv(output_path)
    print(f"\nResults saved to {output_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Test Hierarchical THERAPI on TCGA')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--aligner_path', type=str,
                       default='ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt',
                       help='Path to trained aligner')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of predictor folds')
    parser.add_argument('--output_dir', type=str, default='output/')

    args = parser.parse_args()

    # Set device
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    test_hierarchical_tcga(args)


if __name__ == '__main__':
    main()
