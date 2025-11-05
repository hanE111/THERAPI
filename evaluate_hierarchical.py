"""
Evaluation script for Hierarchical THERAPI using TRANSACT's protocol.

Evaluates using:
1. Mann-Whitney U test per drug (TRANSACT's primary metric)
2. AUROC as effect size
3. Tissue routing accuracy
4. Computational efficiency metrics
"""

import os
import sys
import argparse
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import time

# Import from src
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
from model import ExpDrugDataset, Response_predictor
sys.path.remove(src_path)

# Import hierarchical components
from utils.data_loader import TransactDataLoader
from utils.tissue_mapping import TissueMapper
from models.hierarchical_therapi import HierarchicalTHERAPI, HierarchicalTHERAPIFull


def evaluate_transact_style(model, aligner, test_data_dict, cell_line_data,
                           device, logger_func=print):
    """
    Evaluate using TRANSACT's protocol: Mann-Whitney U test per drug.

    Args:
        model: Response predictor model
        aligner: Hierarchical aligner model
        test_data_dict: Dictionary with test data
        cell_line_data: Cell line expression data
        device: torch device
        logger_func: Logging function

    Returns:
        DataFrame with per-drug evaluation results
    """
    results = []

    # Get unique drugs
    if 'drug_response' in test_data_dict and test_data_dict['drug_response'] is not None:
        drug_response = test_data_dict['drug_response']
        unique_drugs = drug_response['drug_name'].unique() if 'drug_name' in drug_response.columns else []
    else:
        logger_func("Warning: No drug response data available")
        return pd.DataFrame()

    logger_func(f"\nEvaluating {len(unique_drugs)} drugs...")

    for drug in unique_drugs:
        # Get patients who received this drug
        drug_data = drug_response[drug_response['drug_name'] == drug]

        if len(drug_data) == 0:
            continue

        # Predict responses
        predictions = []
        true_responses = []
        tissue_routing_info = []

        model.eval()
        aligner.eval()

        with torch.no_grad():
            for idx, row in drug_data.iterrows():
                # Get patient expression
                patient_id = row['patient_id'] if 'patient_id' in row else idx
                if patient_id in test_data_dict['expression'].index:
                    patient_expr = test_data_dict['expression'].loc[patient_id].values
                    patient_expr_tensor = torch.tensor(patient_expr, dtype=torch.float32).unsqueeze(0).to(device)

                    # Get hierarchical representation
                    align_output = aligner.hierarchical_attention_forward(
                        patient_expr_tensor,
                        cell_line_data.to(device),
                        return_attention=True
                    )

                    # Get drug features (would need to be loaded from data)
                    # For now, use placeholder
                    # genef = ...
                    # chemical = ...

                    # Predict (simplified - needs actual drug features)
                    # prediction = model(align_output['representation'], genef, chemical)
                    # predictions.append(prediction.cpu().item())

                    # Get true response category
                    if 'response_category' in row:
                        true_responses.append(row['response_category'])

                    # Store routing info
                    tissue_routing_info.append({
                        'patient_id': patient_id,
                        'tissue_weights': align_output['tissue_weights'].cpu().numpy()
                    })

        # Compute Mann-Whitney U test if we have predictions
        if len(predictions) > 0 and len(true_responses) > 0:
            # Separate responders and non-responders
            responder_preds = [p for p, r in zip(predictions, true_responses)
                             if r in ['PR', 'CR', 'Complete Response', 'Partial Response']]
            non_responder_preds = [p for p, r in zip(predictions, true_responses)
                                  if r in ['SD', 'PD', 'Stable Disease', 'Progressive Disease']]

            if len(responder_preds) > 0 and len(non_responder_preds) > 0:
                statistic, pvalue = mannwhitneyu(
                    responder_preds,
                    non_responder_preds,
                    alternative='greater'
                )

                # Compute AUROC
                y_true = [1 if r in ['PR', 'CR', 'Complete Response', 'Partial Response'] else 0
                         for r in true_responses]
                try:
                    auroc = roc_auc_score(y_true, predictions)
                except:
                    auroc = np.nan

                results.append({
                    'drug': drug,
                    'mann_whitney_statistic': statistic,
                    'mann_whitney_pvalue': pvalue,
                    'auroc': auroc,
                    'n_responders': len(responder_preds),
                    'n_non_responders': len(non_responder_preds),
                    'n_total': len(predictions)
                })

    results_df = pd.DataFrame(results)

    # Apply multiple testing correction (Benjamini-Hochberg)
    if len(results_df) > 0 and 'mann_whitney_pvalue' in results_df.columns:
        from statsmodels.stats.multitest import multipletests
        _, pvals_corrected, _, _ = multipletests(
            results_df['mann_whitney_pvalue'],
            method='fdr_bh'
        )
        results_df['pvalue_corrected'] = pvals_corrected

    return results_df


def analyze_tissue_routing_accuracy(aligner, test_data_dict, cell_line_data,
                                   tissue_mapper, device, logger_func=print):
    """
    Analyze tissue routing accuracy.

    Args:
        aligner: Hierarchical aligner model
        test_data_dict: Test data dictionary
        cell_line_data: Cell line expressions
        tissue_mapper: TissueMapper instance
        device: torch device
        logger_func: Logging function

    Returns:
        Dictionary with routing accuracy metrics
    """
    aligner.eval()
    correct_routing = 0
    total = 0
    tissue_confusion = np.zeros((tissue_mapper.n_tissue_groups, tissue_mapper.n_tissue_groups))

    test_expr = test_data_dict['expression']
    test_tissues = test_data_dict.get('tissue_info', None)

    if test_tissues is None:
        logger_func("Warning: No tissue info available for routing analysis")
        return {}

    with torch.no_grad():
        for idx in test_expr.index[:100]:  # Sample first 100 for efficiency
            # Get patient expression
            patient_expr = torch.tensor(
                test_expr.loc[idx].values,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Get routing
            output = aligner.hierarchical_attention_forward(
                patient_expr,
                cell_line_data.to(device)
            )

            predicted_tissue_idx = output['tissue_weights'].argmax(dim=-1).item()

            # Get true tissue
            if idx in test_tissues.index:
                true_tissue_name = test_tissues.loc[idx, test_tissues.columns[1]]
                true_tissue_idx = tissue_mapper.get_tissue_idx(true_tissue_name)

                # Update metrics
                tissue_confusion[true_tissue_idx, predicted_tissue_idx] += 1
                if predicted_tissue_idx == true_tissue_idx:
                    correct_routing += 1
                total += 1

    accuracy = correct_routing / total if total > 0 else 0

    return {
        'routing_accuracy': accuracy,
        'n_samples': total,
        'confusion_matrix': tissue_confusion
    }


def analyze_computational_efficiency(hierarchical_model, flat_model,
                                    test_data, cell_line_data, device,
                                    n_iterations=100):
    """
    Compare computational efficiency of hierarchical vs flat attention.

    Args:
        hierarchical_model: Hierarchical THERAPI model
        flat_model: Flat (original) THERAPI model
        test_data: Test expression data
        cell_line_data: Cell line expressions
        device: torch device
        n_iterations: Number of iterations for timing

    Returns:
        Dictionary with timing results
    """
    # Sample test data
    sample_patient = torch.randn(1, test_data.shape[1]).to(device)
    cell_lines = cell_line_data.to(device)

    # Warm up
    for _ in range(10):
        _ = hierarchical_model(sample_patient, cell_lines)
        if flat_model is not None:
            _ = flat_model(sample_patient, cell_lines)

    # Time hierarchical
    hierarchical_model.eval()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = hierarchical_model(sample_patient, cell_lines)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    hierarchical_time = (time.time() - start) / n_iterations

    # Time flat (if available)
    flat_time = None
    if flat_model is not None:
        flat_model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        with torch.no_grad():
            for _ in range(n_iterations):
                _ = flat_model(sample_patient, cell_lines)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        flat_time = (time.time() - start) / n_iterations

    speedup = flat_time / hierarchical_time if flat_time is not None else None

    return {
        'hierarchical_time_ms': hierarchical_time * 1000,
        'flat_time_ms': flat_time * 1000 if flat_time else None,
        'speedup': speedup
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hierarchical THERAPI')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained aligner model')
    parser.add_argument('--predictor_path', type=str,
                       help='Path to trained predictor model')
    parser.add_argument('--target', type=str, default='TCGA',
                       choices=['TCGA', 'PDX', 'HMF'])
    parser.add_argument('--output_dir', type=str, default='output/')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Initialize data loader and tissue mapper
    data_loader = TransactDataLoader(args.data_dir)
    tissue_mapper = checkpoint.get('tissue_mapper', TissueMapper())

    # Load test data
    print(f"\nLoading {args.target} test data...")
    if args.target == 'TCGA':
        test_data = data_loader.load_tcga_data()
    elif args.target == 'PDX':
        test_data = data_loader.load_pdx_data()
    elif args.target == 'HMF':
        test_data = data_loader.load_hmf_data()
    else:
        raise ValueError(f"Unknown target: {args.target}")

    # Load cell line data
    gdsc_data = data_loader.load_gdsc_data()
    common_genes = checkpoint.get('common_genes', None)

    if common_genes is not None:
        cell_line_expr = torch.tensor(
            gdsc_data['expression'][common_genes].values,
            dtype=torch.float32
        )
        test_expr = test_data['expression'][common_genes]
    else:
        print("Warning: No common genes in checkpoint, using all genes")
        cell_line_expr = torch.tensor(gdsc_data['expression'].values, dtype=torch.float32)
        test_expr = test_data['expression']

    # Recreate tissue cell mask
    tissue_cell_mask, _ = tissue_mapper.create_cell_line_tissue_matrix(
        gdsc_data['tissue_mapping']
    )
    tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)

    # Initialize model
    config = checkpoint.get('config', {})
    n_genes = len(common_genes) if common_genes else gdsc_data['expression'].shape[1]

    aligner = HierarchicalTHERAPI(
        n_genes=n_genes,
        n_tissues=tissue_mapper.n_tissue_groups,
        n_latent=config.get('latent_dim', 128),
        tissue_cell_mask=tissue_cell_mask_tensor
    ).to(device)

    aligner.load_state_dict(checkpoint['model_state_dict'])
    aligner.eval()

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # 1. Tissue routing accuracy
    print("\n1. Tissue Routing Accuracy")
    print("-" * 60)
    routing_results = analyze_tissue_routing_accuracy(
        aligner, test_data, cell_line_expr, tissue_mapper, device
    )
    if routing_results:
        print(f"Routing Accuracy: {routing_results['routing_accuracy']:.2%}")
        print(f"Samples Analyzed: {routing_results['n_samples']}")

        # Save confusion matrix
        confusion_df = pd.DataFrame(
            routing_results['confusion_matrix'],
            index=[tissue_mapper.get_tissue_name_from_idx(i) for i in range(tissue_mapper.n_tissue_groups)],
            columns=[tissue_mapper.get_tissue_name_from_idx(i) for i in range(tissue_mapper.n_tissue_groups)]
        )
        confusion_df.to_csv(os.path.join(args.output_dir, f'tissue_routing_confusion_{args.target}.csv'))

    # 2. Computational efficiency
    print("\n2. Computational Efficiency")
    print("-" * 60)
    efficiency_results = analyze_computational_efficiency(
        aligner, None, test_expr, cell_line_expr, device
    )
    print(f"Average inference time: {efficiency_results['hierarchical_time_ms']:.2f} ms")

    # 3. TRANSACT-style evaluation (if drug response data available)
    if test_data.get('drug_response') is not None and args.predictor_path:
        print("\n3. TRANSACT-Style Drug Response Evaluation")
        print("-" * 60)

        # Load predictor if available
        # This would require the full model implementation
        print("Drug response evaluation requires predictor model (not yet implemented)")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
