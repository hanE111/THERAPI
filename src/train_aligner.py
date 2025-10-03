import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from model import *
from utils import set_seed, Logger
from center_loss import CenterLoss


def train_aligner(args):

    model_name = 'THERAPI_aligner'
    logger = Logger(model_name)
    logger('Start training {} model'.format(model_name))
    set_seed(args.seed, logger)

    # parameters
    batch_size = 128
    dim_latent = 128
    lr = 1e-3
    loss_a = 0.2
    loss_b = 0.8
    loss_c = 0.4

    # load data
    gdsc_data_dir = os.path.join(args.data_dir, 'GDSC_gex.csv')
    gdsc_info_dir = os.path.join(args.data_dir, 'GDSC_info.csv')

    pdx_data_dir = os.path.join(args.data_dir, 'PDX_gex.csv')
    pdx_info_dir = os.path.join(args.data_dir, 'PDX_info.csv')

    gdsc_data_df = pd.read_csv(gdsc_data_dir, index_col=0)
    gdsc_info_df = pd.read_csv(gdsc_info_dir)
    num_tissue = gdsc_info_df['tissue_label'].nunique()

    pdx_data_df = pd.read_csv(pdx_data_dir, index_col=0)
    pdx_info_df = pd.read_csv(pdx_info_dir)

    gdsc_data_df = gdsc_data_df.loc[:, pdx_data_df.columns]

    gdsc_dataset = AlignerDataset(gdsc_data_df, 'gdsc', gdsc_info_df['tissue_label'])
    pdx_dataset = AlignerDataset(pdx_data_df, 'pdx', pdx_info_df['tissue_label'])
    pdx_dataloader = DataLoader(pdx_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator = torch.Generator().manual_seed(args.seed))

    # model
    gdsc_AE = GDSC_AE(n_genes=gdsc_dataset.n_genes, n_classes=num_tissue, n_latent=dim_latent)
    pdx_weightencoder = TCGA_weightencoder(n_genes=pdx_dataset.n_genes, n_latent=dim_latent, n_celines=gdsc_data_df.shape[0])
    emb_dis_classifier = Emb_Dis_classifier(n_latent=dim_latent, n_classes=num_tissue)
    exp_dis_classifier = Exp_Dis_classifier(n_genes=pdx_dataset.n_genes, n_latent=dim_latent, n_classes=num_tissue)
    gdsc_AE.to(args.device)
    pdx_weightencoder.to(args.device)
    emb_dis_classifier.to(args.device)
    exp_dis_classifier.to(args.device)

    autoencoder_criterion = nn.MSELoss()
    center_criterion = CenterLoss(num_classes=num_tissue, feat_dim=dim_latent, device=args.device)
    classifier_criterion = nn.CrossEntropyLoss()
    joint_params = (
        parameter
        for module in (gdsc_AE, pdx_weightencoder, emb_dis_classifier, exp_dis_classifier)
        for parameter in module.parameters()
    )
    optimizer = torch.optim.Adam(joint_params, lr=lr)

    if not os.path.exists('ckpts'):
        os.makedirs('ckpts', exist_ok=True)

    # training
    for epoch in np.arange(args.epochs, dtype=np.int64):
        gdsc_AE.train()
        pdx_weightencoder.train()
        emb_dis_classifier.train()
        exp_dis_classifier.train()

        train_losses = 0
        g_losses = 0
        pdx_losses = 0
        batch_count = 0
        for pdx_gex, _, pdx_dis_label in pdx_dataloader:
            pdx_gex = pdx_gex.to(args.device)
            pdx_dis_label = pdx_dis_label.to(args.device)

            gdsc_gex = gdsc_dataset.data.to(args.device)
            gdsc_dis_label = gdsc_dataset.dis_label.to(args.device)
            gdsc_z, gdsc_recon = gdsc_AE(gdsc_gex)

            gdsc_emb_dis_pred = emb_dis_classifier(gdsc_z)
            gdsc_exp_dis_pred = exp_dis_classifier(gdsc_recon)
            
            # GDSC loss
            Grecon_loss = autoencoder_criterion(gdsc_recon, gdsc_gex)
            Gcenter_loss = center_criterion(gdsc_z, gdsc_dis_label)
            Gclass_loss_emb = classifier_criterion(gdsc_emb_dis_pred, gdsc_dis_label)
            Gclass_loss_exp = classifier_criterion(gdsc_exp_dis_pred, gdsc_dis_label)
            G_losses = loss_a*Grecon_loss + loss_b*Gcenter_loss + loss_c*(Gclass_loss_emb + Gclass_loss_exp) 

            # PDX loss
            pdx_weights, pdx_latent, pdx_wgex, pdx_recon = pdx_weightencoder(pdx_gex, gdsc_z, gdsc_gex)
            pdx_emb_dis_pred = emb_dis_classifier(pdx_latent)
            pdx_exp_dis_pred = exp_dis_classifier(pdx_wgex)

            PDXrecon_loss = autoencoder_criterion(pdx_recon, pdx_gex)
            PDXcenter_loss = center_criterion(pdx_latent, pdx_dis_label)
            PDXclass_loss_emb = classifier_criterion(pdx_emb_dis_pred, pdx_dis_label)
            PDXclass_loss_exp = classifier_criterion(pdx_exp_dis_pred, pdx_dis_label)
            PDX_losses = loss_a*PDXrecon_loss + loss_b*PDXcenter_loss + loss_c*(PDXclass_loss_emb + PDXclass_loss_exp)

            # update
            optimizer.zero_grad()
            total_losses = G_losses + PDX_losses
            total_losses.backward()
            optimizer.step()

            train_losses += total_losses.item()
            g_losses += G_losses.item()
            pdx_losses += PDX_losses.item()
            batch_count += 1

        if batch_count:
            train_losses /= batch_count
            g_losses /= batch_count
            pdx_losses /= batch_count
        epoch_idx = epoch.item() + 1
        logger(f'Epoch {epoch_idx}, Train loss {train_losses:.4f}, G_losses {g_losses:.4f}, PDX_losses {pdx_losses:.4f}')

    # save model
    torch.save({'epoch': epoch,
                'gdsc_AE': gdsc_AE.state_dict(),
                'pdx_weightencoder': pdx_weightencoder.state_dict(),
                'emb_dis_classifier': emb_dis_classifier.state_dict(),
                'exp_dis_classifier': exp_dis_classifier.state_dict(),
                'optimizer': optimizer.state_dict()
                }, f'ckpts/{model_name}.pt')
   
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=np.int64, default=0)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--epochs', type=np.int64, default=199)
    args = parser.parse_args()

    train_aligner(args)
