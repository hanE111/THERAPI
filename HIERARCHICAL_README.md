# Hierarchical THERAPI

Hierarchical THERAPI is an extension of THERAPI that introduces a unified tissue-aware architecture with learnable tissue routing for improved drug response prediction.

## Key Innovation

Instead of training multiple tissue-specific models, Hierarchical THERAPI uses a **single unified model** with:

1. **Tissue Routing Network**: Automatically determines tissue relevance for each patient
2. **Hierarchical Attention**: Computes attention within each tissue separately
3. **Adaptive Weighting**: Combines tissue-specific representations with learned weights

## Architecture Overview

```
Patient Sample
      ↓
Patient Encoder
      ↓
Tissue Router → [Tissue Weights]
      ↓
┌─────────┬─────────┬─────────┐
│ Tissue 1│ Tissue 2│ Tissue 3│ ... (Hierarchical Processing)
│ Attn    │ Attn    │ Attn    │
└─────────┴─────────┴─────────┘
      ↓
Weighted Combination
      ↓
Drug Response Predictor
      ↓
Prediction
```

## Directory Structure

```
THERAPI/
├── models/
│   ├── hierarchical_therapi.py   # Main hierarchical model
│   └── ablation_models.py        # Ablation study variants
├── utils/
│   ├── data_loader.py            # TRANSACT data loading
│   └── tissue_mapping.py         # Tissue hierarchies
├── configs/
│   └── hierarchical_config.yaml  # Configuration
├── notebooks/
│   └── tissue_routing_analysis.ipynb  # Analysis notebooks
├── train_hierarchical.py         # Training script
├── evaluate_hierarchical.py      # Evaluation script
└── src/                          # Original THERAPI code
```

## Installation

```bash
# Clone repository
git clone https://github.com/Sunginyoung/THERAPI.git
cd THERAPI

# Create environment
conda create -n hierarchical_therapi python=3.9
conda activate hierarchical_therapi

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Hierarchical THERAPI works with TRANSACT's data structure:

```
data/
├── mini_cancer_genes.csv
├── mini_cancer_lookup_genes.csv
├── GDSC/
│   ├── model_list_20191104.csv
│   ├── rnaseq/GDSC_rnaseq_data.pkl
│   └── response/GDSC[1,2]_fitted_dose_response_*.xlsx
├── PDXE/
│   ├── pancancer_biospecimen.csv
│   ├── fpkm/PDXE_fpkm_data.pkl
│   └── response/response.csv
├── TCGA/
│   ├── pancancer_sample_spec.csv
│   ├── rnaseq/TCGA_rnaseq_data.pkl
│   └── response/response.csv
└── HMF/  # Metastatic data (when available)
```

## Usage

### 1. Train Hierarchical Aligner

```bash
python train_hierarchical.py \
    --source GDSC \
    --target TCGA \
    --config configs/hierarchical_config.yaml \
    --device cuda:0
```

### 2. Evaluate Model

```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target TCGA \
    --output_dir output/
```

### 3. Analyze Results

```bash
jupyter notebook notebooks/tissue_routing_analysis.ipynb
```

## Configuration

Edit `configs/hierarchical_config.yaml` to customize:

```yaml
model:
  latent_dim: 128
  routing_strategy: 'gumbel'  # or 'softmax', 'sparsemax'

training:
  batch_size: 128
  learning_rate: 0.001
  n_epochs: 200

  loss_weights:
    center: 0.8
    routing: 0.4
    entropy: 0.2
    diversity: 0.1
```

## Ablation Studies

Run ablation experiments to validate each component:

```python
from models.ablation_models import create_ablation_model

# Create different variants
flat_model = create_ablation_model('flat', n_genes=n_genes)
uniform_model = create_ablation_model('uniform_weights', n_genes=n_genes, n_tissues=n_tissues)
hard_routing = create_ablation_model('hard_routing', n_genes=n_genes, n_tissues=n_tissues)
```

Available variants:
- `full_model`: Complete hierarchical architecture
- `flat`: No hierarchy (original THERAPI-style)
- `uniform_weights`: Fixed uniform tissue weights
- `hard_routing`: Hard tissue assignments (one-hot)
- `single_head`: Single-head attention
- `fixed_temperature`: Non-learnable temperature

## Key Features

### 1. Learnable Tissue Routing

The tissue router learns to determine tissue relevance:

```python
tissue_weights = model.tissue_router(patient_encoding)
# Returns: [batch_size, n_tissues] soft weights
```

### 2. Hierarchical Attention

Attention computed separately within each tissue:

```python
for tissue_idx in range(n_tissues):
    tissue_cells = cell_lines[tissue_mask[tissue_idx]]
    tissue_repr = attention(patient_enc, tissue_cells)
    weighted_repr = tissue_weights[:, tissue_idx] * tissue_repr
```

### 3. Multiple Routing Strategies

- **Softmax**: Standard soft routing
- **Gumbel-Softmax**: Differentiable discrete sampling
- **Sparsemax**: Sparse tissue selection

## Evaluation Metrics

### TRANSACT-Style Evaluation

1. **Mann-Whitney U test** per drug (primary metric)
2. **AUROC** as effect size
3. **Multiple testing correction** (Benjamini-Hochberg)

### Hierarchical-Specific Metrics

1. **Tissue Routing Accuracy**: % correct tissue predictions
2. **Routing Entropy**: Measure of routing decisiveness
3. **Computational Efficiency**: Speedup vs. flat attention
4. **Attention Interpretability**: Tissue-level attention analysis

## Expected Results

Based on the plan, Hierarchical THERAPI should demonstrate:

1. **Performance**: Match or exceed original THERAPI
2. **Efficiency**: 20-50% faster inference than flat attention
3. **Interpretability**: >70% tissue routing accuracy
4. **Robustness**: Better performance on metastatic (HMF) samples

## Biological Interpretation

### Tissue Routing Analysis

The tissue router learns biologically meaningful patterns:

```python
# Breast cancer patients should route to breast cell lines
breast_patients = data[data['tissue'] == 'breast']
routing_weights = model.tissue_router(breast_patients)
# Expected: High weights for breast tissue index
```

### Cross-Tissue Analysis

Examine routing for metastatic samples:

```python
# Metastatic samples may route to multiple tissues
hmf_data = load_hmf_data()
routing_weights = model.tissue_router(hmf_data)
# Analyze multi-tissue routing patterns
```

## Advantages over Original THERAPI

1. **Single Unified Model**: No need for tissue-specific models
2. **Learnable Routing**: Adapts to tissue relationships
3. **Better Scalability**: Efficient for large datasets
4. **Interpretability**: Explicit tissue relevance scores
5. **Metastatic Handling**: Naturally handles multi-tissue origin

## Comparison to Tissue-Specific Models

| Aspect | Tissue-Specific | Hierarchical |
|--------|----------------|--------------|
| Models needed | N (one per tissue) | 1 (unified) |
| Training time | N × T | T |
| Inference time | ~T | ~0.5T |
| Tissue routing | Manual | Learned |
| Metastatic samples | Ambiguous | Natural |
| Interpretability | Limited | High |

## Citation

If you use Hierarchical THERAPI, please cite:

```bibtex
@article{therapi2024,
  title={THERAPI: Tumor Heterogeneity-aware Embedding for Response Adaptation and Patient Inference},
  author={...},
  journal={...},
  year={2024}
}

@article{hierarchical_therapi2024,
  title={Hierarchical THERAPI: A Unified Tissue-Aware Architecture for Drug Response Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in config
   - Use `mixed_precision: true`

2. **Poor Routing Accuracy**
   - Increase `loss_weights.routing`
   - Decrease `target_temperature`
   - Train for more epochs

3. **Low Routing Entropy (too uniform)**
   - Increase `loss_weights.entropy`
   - Use `routing_strategy: 'sparsemax'`

4. **Data Loading Errors**
   - Check data paths in config
   - Verify data format matches TRANSACT structure
   - Ensure common genes exist across datasets

## Development

### Adding New Features

1. **New Routing Strategy**:
   ```python
   # In models/hierarchical_therapi.py
   def forward(self, ..., strategy='my_strategy'):
       if strategy == 'my_strategy':
           return self._my_strategy(logits)
   ```

2. **Custom Loss**:
   ```python
   # In train_hierarchical.py
   def compute_hierarchical_losses(...):
       my_loss = compute_my_loss(...)
       total_loss += config['loss_weights']['my_loss'] * my_loss
   ```

3. **New Ablation**:
   ```python
   # In models/ablation_models.py
   class MyAblationTHERAPI(HierarchicalTHERAPI):
       def __init__(self, ...):
           super().__init__(...)
           # Modify architecture
   ```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

See LICENSE file in repository root.

## Contact

For questions about Hierarchical THERAPI:
- Email: [your-email@domain.com]
- Issues: [GitHub Issues](https://github.com/Sunginyoung/THERAPI/issues)

For questions about original THERAPI:
- Email: inyoung.sung@snu.ac.kr
