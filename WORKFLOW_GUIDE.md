# Hierarchical THERAPI - Complete Workflow Guide

## Overview

This guide provides step-by-step instructions for using Hierarchical THERAPI, from data preparation to final analysis.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation](#evaluation)
5. [Analysis](#analysis)
6. [Comparison with Original THERAPI](#comparison)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### Environment Setup

```bash
# Create conda environment
conda create -n hierarchical_therapi python=3.9
conda activate hierarchical_therapi

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
```

---

## 2. Data Preparation

### Required Data Structure

```
data/
├── mini_cancer_genes.csv              # Cancer gene list
├── mini_cancer_lookup_genes.csv       # Gene mapping
├── GDSC/
│   ├── model_list_20191104.csv        # Cell line metadata
│   ├── rnaseq/
│   │   └── GDSC_rnaseq_data.pkl      # Expression data
│   ├── response/
│   │   ├── GDSC1_fitted_dose_response_27Oct23.xlsx
│   │   └── GDSC2_fitted_dose_response_27Oct23.xlsx
│   ├── GDSC_perturbation_float16.npy
│   ├── GDSC_perturbation_compound_float16.npy
│   ├── GDSC_rankrepresentation.csv
│   └── GDSC_split/                    # Optional: CV fold indices
│       ├── fold_0_indices.pkl
│       ├── fold_1_indices.pkl
│       └── ...
├── TCGA/
│   ├── pancancer_sample_spec.csv      # Sample metadata
│   ├── rnaseq/
│   │   ├── TCGA_rnaseq_data.pkl
│   │   └── TCGA_rnaseq_sample_annot.pkl
│   ├── response/
│   │   └── response.csv
│   ├── TCGA_perturbation_float16.npy
│   ├── TCGA_perturbation_compound_float16.npy
│   └── TCGA_rankrepresentation.csv
├── PDXE/                              # Optional: PDX data
│   ├── pancancer_biospecimen.csv
│   ├── fpkm/
│   │   └── PDXE_fpkm_data.pkl
│   └── response/
│       └── response.csv
└── HMF/                               # Optional: Metastatic data
```

### Data Validation

```bash
# Check if data files exist
python -c "
import os
required_files = [
    'data/GDSC/rnaseq/GDSC_rnaseq_data.pkl',
    'data/TCGA/rnaseq/TCGA_rnaseq_data.pkl',
    'data/mini_cancer_genes.csv'
]
for f in required_files:
    status = '✓' if os.path.exists(f) else '✗'
    print(f'{status} {f}')
"
```

---

## 3. Training Pipeline

### Quick Start: Run Complete Pipeline

```bash
# Make script executable
chmod +x run_hierarchical_pipeline.sh

# Run complete pipeline
./run_hierarchical_pipeline.sh
```

This will:
1. Train hierarchical aligner (GDSC → TCGA)
2. Train drug response predictor (10-fold CV)
3. Test on TCGA
4. Generate evaluation metrics

### Manual Step-by-Step Training

#### Step 1: Train Hierarchical Aligner

```bash
python train_hierarchical.py \
    --source GDSC \
    --target TCGA \
    --config configs/hierarchical_config.yaml \
    --device cuda:0 \
    --seed 42
```

**Expected output:**
```
Start training HierarchicalTHERAPI_aligner_GDSC_TCGA model
Using 1817 common genes
Epoch 10/200 | Total: 2.3456 | Center: 1.2345 | Routing: 0.3456 | ...
Epoch 200/200 | Total: 0.8765 | Center: 0.4321 | Routing: 0.1234 | ...
Model saved to ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt
```

**Time estimate:** ~2-4 hours on GPU, ~10-20 hours on CPU

#### Step 2: Train Drug Response Predictor

```bash
python train_hierarchical_predictor.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --config configs/hierarchical_config.yaml \
    --device cuda:0 \
    --seed 42
```

**Expected output:**
```
Loading aligner from ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt
Computing patient representations...
Training fold 1/10
Epoch 1, Train loss: 0.5234, Valid loss: 0.5123
...
All folds training completed!
```

**Time estimate:** ~1-2 hours per fold on GPU

#### Step 3: Test on TCGA

```bash
python test_hierarchical_TCGA.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --n_folds 10 \
    --device cuda:0 \
    --output_dir output/
```

**Expected output:**
```
Testing Hierarchical THERAPI on TCGA
Loading aligner...
Computing TCGA patient representations...
Loading 10 predictor models...
Evaluating on TCGA...
Fold 0: AUC=0.7234, AUPRC=0.6789
...
Results saved to output/HierarchicalTHERAPI_test_TCGA.csv
```

---

## 4. Evaluation

### Standard Evaluation

```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target TCGA \
    --output_dir output/ \
    --device cuda:0
```

**Outputs:**
- `output/tissue_routing_confusion_TCGA.csv` - Routing confusion matrix
- `output/tissue_routing_summary.csv` - Summary statistics

### TRANSACT-Style Evaluation

For Mann-Whitney U test per drug:

```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --predictor_path ckpts/HierarchicalTHERAPI_predictor_CV0.pt \
    --target TCGA \
    --output_dir output/
```

### Evaluate on PDX (Intermediate Validation)

```bash
python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target PDX \
    --output_dir output/
```

---

## 5. Analysis

### Interactive Analysis

```bash
# Start Jupyter
jupyter notebook notebooks/tissue_routing_analysis.ipynb
```

The notebook includes:
1. Model and data loading
2. Tissue routing weight visualization
3. Routing accuracy analysis
4. Attention pattern analysis
5. Summary statistics

### Generate Figures

```python
# In notebook or script
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv('output/HierarchicalTHERAPI_test_TCGA.csv')

# Plot
plt.figure(figsize=(10, 6))
results[:-2].plot(kind='bar')  # Exclude mean/std rows
plt.title('Hierarchical THERAPI Performance by Fold')
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('output/performance_by_fold.png')
```

### Tissue Routing Analysis

```python
# Load model
checkpoint = torch.load('ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Analyze routing
with torch.no_grad():
    output = model.hierarchical_attention_forward(
        patient_expr,
        cell_line_exprs,
        return_attention=True
    )

# Get tissue weights
tissue_weights = output['tissue_weights'].numpy()
top_tissues = tissue_weights.argmax(axis=1)

# Print distribution
print("Tissue routing distribution:")
print(pd.Series(top_tissues).value_counts())
```

---

## 6. Comparison with Original THERAPI

### Run Both Versions

#### Original THERAPI

```bash
cd src/
python train_aligner.py --source GDSC --target TCGA --device cuda:0
python train_predictor.py --device cuda:0
python test_TCGA.py --device cuda:0
cd ..
```

#### Hierarchical THERAPI

```bash
./run_hierarchical_pipeline.sh
```

### Compare Results

```python
import pandas as pd

# Load results
orig_results = pd.read_csv('output/THERAPI_test_TCGA.csv')
hier_results = pd.read_csv('output/HierarchicalTHERAPI_test_TCGA.csv')

# Compare
comparison = pd.DataFrame({
    'Original': orig_results.loc[orig_results.index[-2], :],  # Mean row
    'Hierarchical': hier_results.loc[hier_results.index[-2], :]
})

print(comparison)

# Statistical test
from scipy.stats import ttest_rel
for metric in ['AUC', 'AUPRC']:
    orig_vals = orig_results.iloc[:-2][metric].values
    hier_vals = hier_results.iloc[:-2][metric].values
    stat, pval = ttest_rel(orig_vals, hier_vals)
    print(f'{metric}: t={stat:.4f}, p={pval:.4f}')
```

---

## 7. Troubleshooting

### Common Issues

#### Out of Memory

**Symptom:** `RuntimeError: CUDA out of memory`

**Solution:**
```yaml
# In configs/hierarchical_config.yaml
training:
  batch_size: 64  # Reduce from 128
```

Or use CPU:
```bash
python train_hierarchical.py --device cpu
```

#### Poor Routing Accuracy

**Symptom:** Routing accuracy < 50%

**Solutions:**
1. Increase routing loss weight:
```yaml
training:
  loss_weights:
    routing: 0.8  # Increase from 0.4
```

2. Train for more epochs:
```yaml
training:
  n_epochs: 300  # Increase from 200
```

3. Decrease temperature faster:
```yaml
training:
  target_temperature: 0.3  # Decrease from 0.5
```

#### Data Loading Errors

**Symptom:** `FileNotFoundError`

**Solutions:**
1. Check data paths:
```bash
ls -la data/GDSC/rnaseq/
ls -la data/TCGA/rnaseq/
```

2. Verify data format:
```python
import pickle
with open('data/GDSC/rnaseq/GDSC_rnaseq_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(type(data), data.shape if hasattr(data, 'shape') else len(data))
```

#### Slow Training

**Symptom:** Training takes too long

**Solutions:**
1. Use GPU:
```bash
python train_hierarchical.py --device cuda:0
```

2. Reduce number of epochs for testing:
```yaml
training:
  n_epochs: 50  # Quick test
```

3. Use smaller gene set:
```python
# Modify data_loader.py to use top 500 genes
```

#### Model Not Converging

**Symptom:** Loss not decreasing

**Solutions:**
1. Check learning rate:
```yaml
training:
  learning_rate: 0.0001  # Decrease from 0.001
```

2. Check data normalization:
```python
# In data_loader.py
data = (data - data.mean()) / data.std()
```

3. Verify tissue mapping:
```python
tissue_stats = tissue_mapper.get_tissue_statistics(cell_line_tissues)
print(tissue_stats)
# Ensure at least 5 samples per tissue
```

---

## Additional Resources

- **Full Documentation**: [HIERARCHICAL_README.md](HIERARCHICAL_README.md)
- **Quick Reference**: [QUICK_START.md](QUICK_START.md)
- **Architecture Details**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Comparison**: [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## Support

For issues or questions:
1. Check [THERAPI_COMPARISON.md](THERAPI_COMPARISON.md) for common questions
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Open an issue on GitHub
4. Contact: inyoung.sung@snu.ac.kr (original THERAPI)
