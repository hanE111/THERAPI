import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class AlignerDataset(Dataset):
    _LABEL_MAP = {
        'gdsc': 1.0,
        'tcga': 0.0,
        'pdx': 0.0,
    }

    def __init__(self, data_df, data_label, dis_label):
        label_key = data_label.lower()
        assert label_key in self._LABEL_MAP, f"Unsupported data_label '{data_label}'"

        self.data = torch.tensor(data_df.values, dtype=torch.float32)
        self.n_genes = self.data.shape[1]
        self.n_samples = self.data.shape[0]
        self.gene_set = data_df.columns.to_list()

        label_value = self._LABEL_MAP[label_key]
        self.data_label = torch.full((self.n_samples,), label_value, dtype=torch.float32)

        dis_array = np.asarray(dis_label)
        assert dis_array.shape[0] == self.n_samples, "Mismatch between data and label counts"
        self.dis_label = torch.tensor(dis_array, dtype=torch.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.data_label[idx], self.dis_label[idx]

class ExpDrugDataset(Dataset):
    def __init__(self, emb_list, genef_list, chemical_list, resp_list):
        self.emb = emb_list #torch.tensor(emb_list, dtype=torch.float32)
        first_emb = np.asarray(emb_list[0])
        self.emb_dim = first_emb.shape[0]
        self.n_samples = np.asarray(emb_list).shape[0]
        self.genef = genef_list
        self.genef_dim = np.asarray(genef_list[0]).shape[0]
        self.chemical = chemical_list #torch.tensor(chemical_list, dtype=torch.float32)
        self.chemical_dim = np.asarray(chemical_list[0]).shape[0]
        self.resp = torch.tensor(resp_list, dtype=torch.float32)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        emb = torch.tensor(self.emb[idx], dtype=torch.float32)
        genef = torch.tensor(self.genef[idx], dtype=torch.float32)
        chemical = torch.tensor(self.chemical[idx], dtype=torch.float32)
        resp = self.resp[idx]
        
        return emb, genef, chemical, resp

class GDSC_AE(nn.Module):
    def __init__(self, n_genes, n_classes, n_latent):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_genes)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon
    
class TCGA_weightencoder(nn.Module):
    def __init__(self, n_genes, n_latent, n_celines = 673):
        nn.Module.__init__(self)
        self.n_celines = n_celines

        self.Q = nn.Sequential(
            nn.Linear(n_genes, n_latent),
            nn.LayerNorm(n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )
        
        self.K = nn.Linear(n_latent, n_latent, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_genes)
        )

    def forward(self, x, celine_embs, celine_exps):
        q_x = self.Q(x)
        k_celine = self.K(celine_embs)
        attn = torch.matmul(q_x, k_celine.T)

        weights = torch.softmax(attn / np.sqrt(q_x.shape[1]), dim=1)
        wlatent_emb = torch.matmul(weights, celine_embs)
        wexp_emb = torch.matmul(weights, celine_exps)
        recon = self.decoder(wlatent_emb)
        return weights, wlatent_emb, wexp_emb, recon
    
class Emb_Dis_classifier(nn.Module):
    def __init__(self, n_latent, n_classes):
        nn.Module.__init__(self)
        self.label_classifier = nn.Sequential(
            nn.Linear(n_latent, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, z):
        return self.label_classifier(z)
    
class Exp_Dis_classifier(nn.Module):
    def __init__(self, n_genes, n_latent, n_classes):
        nn.Module.__init__(self)
        self.label_classifier = nn.Sequential(
            nn.Linear(n_genes, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, z):
        return self.label_classifier(z)
    
class Response_predictor(nn.Module):
    def __init__(self, emb_dim, genef_dim, chemical_dim
                 , hidden_dim1, hidden_dim2, output_dim, dropout=0.1):
        nn.Module.__init__(self)
        self.emb_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.genef_fc = nn.Sequential(
            nn.Linear(genef_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.chemical_fc = nn.Sequential(
            nn.Linear(chemical_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim1*3,hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1,hidden_dim2),
            # nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2,output_dim)
            )
    def forward(self, emb, genef, chemical):
        emb_out = self.emb_fc(emb)
        genef_out = self.genef_fc(genef)
        chemical_out = self.chemical_fc(chemical)
        out = torch.cat([emb_out, genef_out, chemical_out], dim=1)
        out = self.pred_head(out)
        return out