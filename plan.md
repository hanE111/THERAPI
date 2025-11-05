## Task: Implement Hierarchical THERAPI with TRANSACT's Data - A Unified Tissue-Aware Architecture

### Context
I have the original THERAPI code from https://github.com/Sunginyoung/THERAPI. The original THERAPI author is my co-author. I will be using TRANSACT's data (GDSC + PDX + TCGA + HMF) instead of THERAPI's data because it provides more comprehensive benchmarking with two independent clinical cohorts and PDX as an intermediate validation. My contribution will be a **hierarchical architecture** that uses a single unified model with tissue-aware routing, fundamentally different from THERAPI's multiple separate tissue-specific models approach.

### Data Structure Overview
```
data/
├── mini_cancer_genes.csv                   # Cancer gene list (1,817 genes)
├── mini_cancer_lookup_genes.csv           # Gene name to ENSEMBL ID mapping
├── GDSC/                                   # Cell line data
│   ├── model_list_20191104.csv            # Cell line metadata with tissue types
│   ├── rnaseq/GDSC_rnaseq_data.pkl       # Expression data
│   └── response/GDSC[1,2]_fitted_dose_response_*.xlsx  # Drug responses
├── PDXE/                                   # Patient-Derived Xenografts
│   ├── pancancer_biospecimen.csv          # PDX to tissue mapping
│   ├── fpkm/PDXE_fpkm_data.pkl           # Expression data
│   └── response/response.csv              # Drug responses
├── TCGA/                                   # Patient data (primary tumors)
│   ├── pancancer_sample_spec.csv          # Sample tissue types
│   ├── rnaseq/TCGA_rnaseq_data.pkl       # Expression data
│   └── response/response.csv              # Clinical responses
└── HMF/                                    # Metastatic patient data
    └── (to be populated when available)
```

### Objective
Implement "Hierarchical THERAPI" which:
1. Uses THERAPI's proven alignment and prediction components
2. Adds a tissue routing mechanism that automatically determines tissue relevance
3. Computes attention within each tissue's cell lines separately
4. Works with TRANSACT's data and evaluation protocol
5. Provides a single unified model instead of multiple tissue-specific models

### Data Loading and Preprocessing Instructions

#### DO ✅:

1. **Create data loading utilities** (`utils/data_loader.py`):
```python
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

class TransactDataLoader:
    def __init__(self, data_root='data/'):
        self.data_root = data_root
        self.cancer_genes = pd.read_csv(f'{data_root}/mini_cancer_genes.csv')
        self.gene_mapping = pd.read_csv(f'{data_root}/mini_cancer_lookup_genes.csv')
        
    def load_gdsc_data(self):
        """Load GDSC cell line data with tissue labels"""
        # Load expression data
        with open(f'{self.data_root}/GDSC/rnaseq/GDSC_rnaseq_data.pkl', 'rb') as f:
            gdsc_expr = pickle.load(f)
        
        # Load cell line metadata with tissue types
        cell_info = pd.read_csv(f'{self.data_root}/GDSC/model_list_20191104.csv')
        
        # Load drug responses from both GDSC1 and GDSC2
        gdsc1_resp = pd.read_excel(
            f'{self.data_root}/GDSC/response/GDSC1_fitted_dose_response_27Oct23.xlsx'
        )
        gdsc2_resp = pd.read_excel(
            f'{self.data_root}/GDSC/response/GDSC2_fitted_dose_response_27Oct23.xlsx'
        )
        
        # Extract tissue mapping
        tissue_mapping = dict(zip(cell_info['model_name'], cell_info['tissue']))
        
        return {
            'expression': gdsc_expr,
            'cell_info': cell_info,
            'drug_response': pd.concat([gdsc1_resp, gdsc2_resp]),
            'tissue_mapping': tissue_mapping
        }
    
    def load_tcga_data(self):
        """Load TCGA patient data with tissue labels"""
        # Load expression
        with open(f'{self.data_root}/TCGA/rnaseq/TCGA_rnaseq_data.pkl', 'rb') as f:
            tcga_expr = pickle.load(f)
            
        # Load sample annotations
        with open(f'{self.data_root}/TCGA/rnaseq/TCGA_rnaseq_sample_annot.pkl', 'rb') as f:
            sample_annot = pickle.load(f)
            
        # Load clinical responses
        tcga_resp = pd.read_csv(f'{self.data_root}/TCGA/response/response.csv')
        
        # Load tissue types
        tissue_info = pd.read_csv(f'{self.data_root}/TCGA/pancancer_sample_spec.csv')
        
        return {
            'expression': tcga_expr,
            'sample_annot': sample_annot,
            'drug_response': tcga_resp,
            'tissue_info': tissue_info
        }
    
    def load_pdx_data(self):
        """Load PDX data as intermediate validation"""
        # Similar structure for PDX
        pass
    
    def harmonize_genes(self, *datasets):
        """Ensure all datasets use same gene set"""
        # Use cancer gene panel for focused analysis
        common_genes = set(self.cancer_genes['gene_symbol'])
        
        for dataset in datasets:
            available_genes = set(dataset.columns)
            common_genes = common_genes.intersection(available_genes)
            
        return list(common_genes)
```

2. **Create tissue mapping utilities** (`utils/tissue_mapping.py`):
```python
class TissueMapper:
    """Map cell lines to tissue types and create hierarchical structure"""
    
    # Standardized tissue categories (consolidate similar tissues)
    TISSUE_GROUPS = {
        'breast': ['breast', 'breast cancer', 'mammary'],
        'lung': ['lung', 'lung cancer', 'NSCLC', 'SCLC'],
        'colon': ['colon', 'colorectal', 'large intestine'],
        'blood': ['leukemia', 'lymphoma', 'myeloma', 'blood', 'haematopoietic'],
        'brain': ['glioma', 'glioblastoma', 'CNS', 'brain'],
        'skin': ['melanoma', 'skin'],
        'pancreas': ['pancreas', 'pancreatic'],
        'ovary': ['ovary', 'ovarian'],
        'kidney': ['kidney', 'renal'],
        'liver': ['liver', 'hepatocellular'],
        # Add more mappings
    }
    
    def __init__(self):
        self.tissue_to_group = self._create_tissue_to_group_mapping()
        self.n_tissue_groups = len(self.TISSUE_GROUPS)
        
    def _create_tissue_to_group_mapping(self):
        """Create mapping from specific tissue names to groups"""
        mapping = {}
        for group, tissues in self.TISSUE_GROUPS.items():
            for tissue in tissues:
                mapping[tissue.lower()] = group
        return mapping
        
    def get_tissue_group(self, tissue_name):
        """Map specific tissue name to standardized group"""
        return self.tissue_to_group.get(tissue_name.lower(), 'other')
    
    def create_cell_line_tissue_matrix(self, cell_line_tissues):
        """Create binary matrix [n_tissues x n_cell_lines]"""
        n_cell_lines = len(cell_line_tissues)
        matrix = np.zeros((self.n_tissue_groups + 1, n_cell_lines))  # +1 for 'other'
        
        tissue_to_idx = {tissue: i for i, tissue in enumerate(self.TISSUE_GROUPS.keys())}
        tissue_to_idx['other'] = self.n_tissue_groups
        
        for i, (cell_line, tissue) in enumerate(cell_line_tissues.items()):
            tissue_group = self.get_tissue_group(tissue)
            tissue_idx = tissue_to_idx[tissue_group]
            matrix[tissue_idx, i] = 1
            
        return matrix, tissue_to_idx
```

3. **Implement the Hierarchical THERAPI architecture** (`models/hierarchical_therapi.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.therapi import THERAPI  # Import original THERAPI components

class HierarchicalTHERAPI(nn.Module):
    def __init__(self, base_config, n_tissues, tissue_mapping):
        super().__init__()
        
        # Reuse THERAPI's proven components
        self.patient_encoder = base_config.patient_encoder
        self.cell_line_encoder = base_config.cell_line_encoder
        self.attention_module = base_config.attention_module
        self.predictor = base_config.predictor
        
        # NEW: Tissue routing network (your key contribution)
        self.tissue_router = nn.Sequential(
            nn.Linear(base_config.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_tissues)
        )
        
        # Temperature for gumbel softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Store tissue mapping as buffer
        self.register_buffer('tissue_cell_mask', tissue_mapping)  # [n_tissues, n_cell_lines]
        
    def compute_tissue_weights(self, patient_encoding, strategy='gumbel'):
        """Compute tissue relevance weights with different strategies"""
        tissue_logits = self.tissue_router(patient_encoding)
        
        if strategy == 'softmax':
            return F.softmax(tissue_logits, dim=-1)
        elif strategy == 'gumbel':
            return F.gumbel_softmax(tissue_logits, tau=self.temperature, hard=False)
        elif strategy == 'sparsemax':
            # Implement sparsemax for sparse tissue selection
            return self.sparsemax(tissue_logits)
            
    def hierarchical_attention(self, patient_expr, cell_line_exprs, drug_features):
        """Core hierarchical attention mechanism"""
        # Encode patient
        patient_enc = self.patient_encoder(patient_expr)
        
        # Encode all cell lines
        cell_line_encs = self.cell_line_encoder(cell_line_exprs)
        
        # Compute tissue routing weights
        tissue_weights = self.compute_tissue_weights(patient_enc)
        
        # Store detailed attention info for analysis
        attention_breakdown = {
            'tissue_weights': tissue_weights.detach().cpu(),
            'tissue_attention': {}
        }
        
        # Compute attention within each tissue
        weighted_representations = []
        
        for tissue_idx in range(self.tissue_cell_mask.shape[0]):
            # Get cell lines for this tissue
            tissue_mask = self.tissue_cell_mask[tissue_idx]  # [n_cell_lines]
            
            if tissue_mask.sum() == 0:  # No cell lines for this tissue
                continue
                
            # Select cell lines for this tissue
            tissue_cell_encs = cell_line_encs[tissue_mask.bool()]
            
            # Compute attention within tissue using THERAPI's attention
            tissue_attention_weights = self.attention_module(
                query=patient_enc.unsqueeze(1),  # [batch, 1, dim]
                key=tissue_cell_encs.unsqueeze(0),  # [1, n_tissue_cells, dim]
                value=tissue_cell_encs.unsqueeze(0)
            )
            
            # Aggregate within tissue
            tissue_repr = (tissue_attention_weights @ tissue_cell_encs.unsqueeze(0)).squeeze(1)
            
            # Weight by tissue relevance
            weighted_repr = tissue_weights[:, tissue_idx:tissue_idx+1] * tissue_repr
            weighted_representations.append(weighted_repr)
            
            # Store for analysis
            attention_breakdown['tissue_attention'][tissue_idx] = {
                'weights': tissue_attention_weights.detach().cpu(),
                'n_cells': tissue_mask.sum().item()
            }
        
        # Combine all tissue representations
        final_representation = torch.stack(weighted_representations).sum(dim=0)
        
        # Predict drug response using THERAPI's predictor
        prediction = self.predictor(
            torch.cat([final_representation, drug_features], dim=-1)
        )
        
        return {
            'prediction': prediction,
            'representation': final_representation,
            'attention_breakdown': attention_breakdown,
            'tissue_weights': tissue_weights
        }
```

4. **Create training script** (`train_hierarchical.py`):
```python
def train_hierarchical_therapi(config):
    # Load data
    data_loader = TransactDataLoader(config.data_root)
    gdsc_data = data_loader.load_gdsc_data()
    tcga_data = data_loader.load_tcga_data()
    pdx_data = data_loader.load_pdx_data()
    
    # Harmonize genes across datasets
    common_genes = data_loader.harmonize_genes(
        gdsc_data['expression'],
        tcga_data['expression']
    )
    
    # Create tissue mappings
    tissue_mapper = TissueMapper()
    cell_line_tissue_matrix, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(
        gdsc_data['tissue_mapping']
    )
    
    # Initialize model
    model = HierarchicalTHERAPI(
        base_config=config.therapi_config,
        n_tissues=len(tissue_to_idx),
        tissue_mapping=torch.tensor(cell_line_tissue_matrix)
    )
    
    # Training loop with special losses
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    for epoch in range(config.n_epochs):
        # Regular THERAPI losses
        for batch in train_loader:
            output = model.hierarchical_attention(
                batch['patient_expr'],
                batch['cell_line_exprs'],
                batch['drug_features']
            )
            
            # Prediction loss
            pred_loss = F.mse_loss(output['prediction'], batch['response'])
            
            # Tissue routing regularization
            tissue_weights = output['tissue_weights']
            
            # Entropy regularization (encourage decisive routing)
            entropy = -torch.sum(tissue_weights * torch.log(tissue_weights + 1e-8), dim=-1)
            entropy_loss = entropy.mean()
            
            # Diversity loss (different patients should use different tissues)
            batch_tissue_dist = tissue_weights.mean(dim=0)
            diversity_loss = -torch.sum(batch_tissue_dist * torch.log(batch_tissue_dist + 1e-8))
            
            # Temperature regularization (anneal over time)
            temp_loss = F.mse_loss(model.temperature, torch.tensor(0.5))
            
            # Combined loss
            total_loss = (pred_loss + 
                         config.alpha * entropy_loss - 
                         config.beta * diversity_loss +
                         config.gamma * temp_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Log metrics
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}:")
                print(f"  Prediction Loss: {pred_loss.item():.4f}")
                print(f"  Entropy: {entropy.mean().item():.4f}")
                print(f"  Temperature: {model.temperature.item():.4f}")
                
        # Validate on PDX (no patient labels used!)
        if epoch % 5 == 0:
            validate_on_pdx(model, pdx_data)
```

5. **Create evaluation script** (`evaluate_hierarchical.py`):
```python
def evaluate_transact_style(model, test_data):
    """Evaluate using TRANSACT's protocol"""
    results = {}
    
    for drug in test_data['drugs']:
        # Get patients who received this drug
        drug_patients = test_data[test_data['drug'] == drug]
        
        # Predict responses
        predictions = []
        true_responses = []
        tissue_routing_info = []
        
        for patient in drug_patients:
            output = model.hierarchical_attention(
                patient['expression'],
                gdsc_cell_lines,
                drug['features']
            )
            
            predictions.append(output['prediction'])
            true_responses.append(patient['response_category'])
            tissue_routing_info.append({
                'true_tissue': patient['tissue'],
                'predicted_weights': output['tissue_weights'],
                'attention_entropy': compute_attention_entropy(output['attention_breakdown'])
            })
        
        # Mann-Whitney U test (TRANSACT's evaluation)
        responders = [p for p, r in zip(predictions, true_responses) if r in ['PR', 'CR']]
        non_responders = [p for p, r in zip(predictions, true_responses) if r in ['SD', 'PD']]
        
        if len(responders) > 0 and len(non_responders) > 0:
            statistic, pvalue = mannwhitneyu(
                responders, 
                non_responders,
                alternative='greater'
            )
            
            # Compute AUROC
            y_true = [1 if r in ['PR', 'CR'] else 0 for r in true_responses]
            auroc = roc_auc_score(y_true, predictions)
            
            results[drug] = {
                'mann_whitney_p': pvalue,
                'auroc': auroc,
                'n_responders': len(responders),
                'n_non_responders': len(non_responders)
            }
    
    return results

def analyze_hierarchical_benefits(model, test_data):
    """Analyze specific benefits of hierarchical approach"""
    
    # 1. Tissue routing accuracy
    correct_tissue_routing = 0
    total = 0
    
    for patient in test_data:
        output = model(patient)
        predicted_tissue = output['tissue_weights'].argmax()
        true_tissue = patient['tissue_idx']
        
        if predicted_tissue == true_tissue:
            correct_tissue_routing += 1
        total += 1
    
    # 2. Computational efficiency
    hierarchical_time = time_hierarchical_forward_pass(model)
    flat_time = time_flat_forward_pass(original_therapi)
    speedup = flat_time / hierarchical_time
    
    # 3. Attention entropy comparison
    hierarchical_entropy = compute_average_attention_entropy(model, test_data)
    flat_entropy = compute_average_attention_entropy(original_therapi, test_data)
    
    return {
        'tissue_routing_accuracy': correct_tissue_routing / total,
        'computational_speedup': speedup,
        'entropy_reduction': (flat_entropy - hierarchical_entropy) / flat_entropy,
        'metastatic_performance': evaluate_on_metastatic_subset(model, test_data)
    }
```

### Key Implementation Requirements

#### DO ✅:

1. **Preserve THERAPI's core strengths**:
   - Use their alignment approach
   - Use their attention mechanism (just apply it hierarchically)
   - Use their predictor architecture

2. **Add clear architectural innovations**:
   - Tissue routing network (learnable, not fixed)
   - Hierarchical attention (parallel within tissues)
   - Uncertainty via temperature-controlled Gumbel softmax

3. **Follow TRANSACT's evaluation exactly**:
   - Mann-Whitney U test per drug
   - AUROC as effect size
   - Multiple testing correction
   - Separate evaluation on TCGA and HMF

4. **Create comprehensive ablations**:
   ```python
   ablations = {
       'full_model': HierarchicalTHERAPI(...),
       'no_hierarchy': FlatTHERAPI(...),  # Original THERAPI
       'fixed_uniform_weights': HierarchicalTHERAPI(uniform_weights=True),
       'no_temperature': HierarchicalTHERAPI(learnable_temp=False),
       'hard_routing': HierarchicalTHERAPI(gumbel_hard=True)
   }
   ```

5. **Track interpretability metrics**:
   - Which tissues contribute to which drugs?
   - Do breast cancer patients route to breast cell lines?
   - How does routing change for metastatic (HMF) samples?

#### DON'T ❌:

1. **Don't use patient outcome labels** during alignment or model selection
2. **Don't modify THERAPI's core modules** - wrap and extend
3. **Don't ignore PDX** - use it for hyperparameter selection
4. **Don't make claims without statistical testing**
5. **Don't forget to handle missing tissue types** - have an "other" category

### Expected File Structure

```
hierarchical_therapi/
├── data/                           # TRANSACT's data
├── models/
│   ├── therapi.py                 # Original THERAPI (copy, don't modify)
│   ├── hierarchical_therapi.py    # Your implementation
│   └── ablation_models.py         # Ablation variants
├── utils/
│   ├── data_loader.py             # Load TRANSACT's data
│   ├── tissue_mapping.py          # Tissue organization
│   └── evaluation_metrics.py      # TRANSACT-style evaluation
├── train_hierarchical.py
├── evaluate_hierarchical.py
├── configs/
│   └── hierarchical_config.yaml
└── notebooks/
    ├── tissue_routing_analysis.ipynb
    ├── computational_efficiency.ipynb
    └── metastatic_analysis.ipynb
```

### Success Criteria

1. **Performance**: Match or exceed THERAPI on TRANSACT's benchmark
2. **Efficiency**: 20-50% faster than flat attention
3. **Interpretability**: >70% correct tissue routing
4. **Robustness**: Better performance on metastatic (HMF) samples
5. **Ablations**: Clear degradation without hierarchical structure

### Critical Notes

- **PDX is key**: Use it to select hyperparameters without touching patient labels
- **HMF is your ace**: Show that hierarchical routing adapts better to metastatic samples
- **Biological story**: Frame as "learning tissue relationships" not just efficiency
- **Temperature annealing**: Start high (explore) then decrease (exploit)
- **Sparse routing**: Consider sparsemax for interpretability

This implementation combines THERAPI's proven architecture with your hierarchical innovation, evaluated on TRANSACT's comprehensive benchmark. The key is showing that one unified model with learnable tissue routing outperforms both flat attention and multiple tissue-specific models.