import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score

from torch.utils.data import DataLoader

from model import Response_predictor, ExpDrugDataset


def get_test_results(model, test_dataloader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for emb, genef, chemical, resp in test_dataloader:
            emb, genef, chemical = emb.to(device), genef.to(device), chemical.to(device)
            output = model(emb, genef, chemical)
            preds.append(output.squeeze().cpu())
            targets.append(resp)

    return preds, targets


def get_test_metrics_withcutoff(preds, targets, cutoffs):
    probs = torch.cat(preds).numpy()
    preds = (probs > cutoffs).astype(int)

    targets = torch.cat(targets).numpy()

    auc = roc_auc_score(targets, probs)
    auprc = average_precision_score(targets, probs)
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds)
    f1 = f1_score(targets, preds)

    return auc, auprc, acc, precision, f1


def test_PDX1(args):

    # parameters
    batch_size = 512
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 1
    folds = 10

    data_prefix = os.path.join(args.data_dir, args.dataset_prefix)

    perturbation_np = np.load(f"{data_prefix}_perturbation.npy")
    rank_np = np.load(f"{data_prefix}_rank.npy")
    compound_np = np.load(f"{data_prefix}_perturbation_compound.npy")
    samples = np.load(f"{data_prefix}_samples.npy")
    resp_df = pd.read_csv(f"{data_prefix}_resp.csv")

    if len(resp_df) != len(samples):
        raise ValueError("Sample order mismatch between response table and saved arrays")

    labels = resp_df['Label'].values.astype(np.float32)
    dataset = ExpDrugDataset(perturbation_np, rank_np, compound_np, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_models = [
        Response_predictor(dataset.emb_dim, dataset.genef_dim, dataset.chemical_dim, hidden_dim1, hidden_dim2, output_dim).to(args.device)
        for _ in range(folds)
    ]

    best_model_names = [
        args.model_name + f'_CV{fold}'
        for fold in range(folds)
    ]

    for i in range(folds):
        best_model_dir = f"ckpts/{best_model_names[i]}.pt"
        best_models[i].load_state_dict(torch.load(best_model_dir, map_location=args.device))

    pdx_results = []
    all_predictions = []
    for i in range(folds):
        test_preds, test_targets = get_test_results(best_models[i], dataloader, args.device)
        pdx_results.append(get_test_metrics_withcutoff(test_preds, test_targets, 0))
        all_predictions.append(torch.cat(test_preds).numpy())

    pdx_results_df = pd.DataFrame(pdx_results, columns=['AUC', 'AUPRC', 'Accuracy', 'Precision', 'F1'])
    means = pdx_results_df.mean()
    stds = pdx_results_df.std()
    pdx_results_df.loc['mean'] = means
    pdx_results_df.loc['std'] = stds

    mean_predictions = np.mean(np.vstack(all_predictions), axis=0)
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    pdx_results_df.round(4).to_csv(os.path.join(output_dir, 'THERAPI_test_PDX1_metrics.csv'), index=False)

    prediction_df = pd.DataFrame({
        'Model': samples,
        'Drug': resp_df['Drug'],
        'Original_Treatment': resp_df['Original_Treatment'],
        'Label': labels,
        'Probability': mean_predictions,
    })
    prediction_df.to_csv(os.path.join(output_dir, 'THERAPI_test_PDX1_predictions.csv'), index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--dataset_prefix', type=str, default='PDX1')
    parser.add_argument('--model_name', type=str, default='THERAPI_predictor')
    parser.add_argument('--output_dir', type=str, default='../output/')
    args = parser.parse_args()

    test_PDX1(args)
