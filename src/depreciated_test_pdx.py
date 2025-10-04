import os

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score

from torch.utils.data import DataLoader

from model import *

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


def test_PDX(args):

    # parameters
    batch_zise = 512
    hidden_dim1 = 256
    hidden_dim2 = 128
    output_dim = 1
    folds = 10

    # load data
    pdx_resp_dir = args.data_dir + 'PDX_resp.csv'
    pdx_rank_dir = args.data_dir + 'PDX_rankrepresentation.csv'
    pdx_pert_dir = args.data_dir + 'PDX_perturbation.npy'
    pdx_comp_dir = args.data_dir + 'PDX_perturbation_compound.npy'

    pdx_resp_df = pd.read_csv(pdx_resp_dir)
    pdx_rank_df = pd.read_csv(pdx_rank_dir, index_col=0).values
    pdx_pert_np = np.load(pdx_pert_dir)
    pdx_comp_np = np.load(pdx_comp_dir)

    pdx_label_col = 'Label'
    pdx_dataset = ExpDrugDataset(pdx_pert_np, pdx_rank_df, pdx_comp_np, pdx_resp_df[pdx_label_col].values)
    pdx_dataloader = DataLoader(pdx_dataset, batch_size=batch_zise, shuffle=False)

    # load trained model
    best_models = [
        Response_predictor(pdx_dataset.emb_dim, pdx_dataset.genef_dim, pdx_dataset.chemical_dim, hidden_dim1, hidden_dim2, output_dim).to(args.device)
        for _ in range(folds)
    ]

    best_model_names = [
        args.model_name + f'_CV{fold}'
        for fold in range(folds)
    ]

    for i, cv in enumerate(best_models):
        best_model_dir = f'ckpts/{best_model_names[i]}.pt'
        best_models[i].load_state_dict(torch.load(best_model_dir, map_location=args.device))

    # report test results
    pdx_results_mean = []
    for i, cv in enumerate(range(folds)):
        test_preds, test_targets = get_test_results(best_models[cv], pdx_dataloader, args.device)
        pdx_results_mean.append(get_test_metrics_withcutoff(test_preds, test_targets, 0))

    pdx_results_df = pd.DataFrame(pdx_results_mean)
    pdx_results_df.columns = ['AUC', 'AUPRC', 'Accuracy', 'Precision', 'F1']
    means = pdx_results_df.mean()
    stds = pdx_results_df.std()
    pdx_results_df.loc['mean'] = means
    pdx_results_df.loc['std'] = stds

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return  pdx_results_df.round(4).to_csv(args.output_dir+'THERAPI_test_PDX.csv', index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--model_name', type=str, default='THERAPI_predictor')
    parser.add_argument('--output_dir', type=str, default='../output/')
    args = parser.parse_args()

    test_PDX(args)
