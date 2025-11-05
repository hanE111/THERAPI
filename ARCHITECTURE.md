# Hierarchical THERAPI Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hierarchical THERAPI                          │
│                                                                  │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │ Patient      │                    │ Cell Line    │          │
│  │ Expression   │                    │ Expression   │          │
│  │ [batch, G]   │                    │ [N, G]       │          │
│  └──────┬───────┘                    └──────┬───────┘          │
│         │                                   │                   │
│         ▼                                   ▼                   │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │ Patient      │                    │ Cell Line    │          │
│  │ Encoder      │                    │ Encoder      │          │
│  │              │                    │              │          │
│  │ Linear(G→128)│                    │ Linear(G→128)│          │
│  │ LayerNorm    │                    │ LayerNorm    │          │
│  │ ReLU         │                    │ ReLU         │          │
│  └──────┬───────┘                    └──────┬───────┘          │
│         │                                   │                   │
│         │    [batch, 128]                   │  [N, 128]         │
│         │                                   │                   │
│         ▼                                   │                   │
│  ┌──────────────────────┐                  │                   │
│  │   Tissue Router      │                  │                   │
│  │                      │                  │                   │
│  │  Linear(128→512)     │                  │                   │
│  │  BatchNorm           │                  │                   │
│  │  ReLU, Dropout       │                  │                   │
│  │  Linear(512→256)     │                  │                   │
│  │  BatchNorm           │                  │                   │
│  │  ReLU, Dropout       │                  │                   │
│  │  Linear(256→T)       │                  │                   │
│  │                      │                  │                   │
│  │  Gumbel-Softmax      │                  │                   │
│  │  (temp=learned)      │                  │                   │
│  └──────┬───────────────┘                  │                   │
│         │                                   │                   │
│         │  [batch, T]                       │                   │
│         │  Tissue Weights                   │                   │
│         │                                   │                   │
│    ┌────┴────┬────────┬────────┬───────────┴─────┐            │
│    │         │        │        │                  │            │
│    ▼         ▼        ▼        ▼                  ▼            │
│  Tissue 1  Tissue 2 Tissue 3  ...              Tissue T        │
│  ┌────────┐ ┌────────┐ ┌────────┐              ┌────────┐     │
│  │ Mask   │ │ Mask   │ │ Mask   │              │ Mask   │     │
│  │ Cells  │ │ Cells  │ │ Cells  │    ...       │ Cells  │     │
│  └────┬───┘ └────┬───┘ └────┬───┘              └────┬───┘     │
│       │          │          │                        │         │
│       ▼          ▼          ▼                        ▼         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Multi-Head Attention (per tissue)                │ │
│  │                                                           │ │
│  │  Query: Patient Encoding    [batch, 128]                │ │
│  │  Keys:  Tissue Cell Lines   [n_tissue_cells, 128]       │ │
│  │  Values: Tissue Cell Lines  [n_tissue_cells, 128]       │ │
│  │                                                           │ │
│  │  Attention = Softmax(QK^T / √d)                          │ │
│  │  Output = Attention × V                                  │ │
│  └────┬─────────────────────────────────────────────────────┘ │
│       │                                                        │
│       │  [batch, 128] per tissue                              │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────┐                 │
│  │  Weight by Tissue Relevance              │                 │
│  │                                           │                 │
│  │  Weighted_Repr_i = Tissue_Weight_i × Repr_i               │
│  └────┬─────────────────────────────────────┘                 │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────┐                 │
│  │  Sum Across Tissues                      │                 │
│  │                                           │                 │
│  │  Final_Repr = Σ Weighted_Repr_i          │                 │
│  └────┬─────────────────────────────────────┘                 │
│       │                                                        │
│       │  [batch, 128]                                         │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────────────────────────┐                 │
│  │   Drug Response Predictor                │                 │
│  │                                           │                 │
│  │   Input: [Patient_Repr, Gene_Feat, Drug] │                 │
│  │   Output: Response Prediction            │                 │
│  └───────────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Patient Encoder

```python
nn.Sequential(
    nn.Linear(n_genes, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 128)
)
```

**Purpose**: Encode patient gene expression into latent space

### 2. Cell Line Encoder

```python
nn.Sequential(
    nn.Linear(n_genes, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
```

**Purpose**: Encode cell line gene expression into same latent space

### 3. Tissue Router (Key Innovation)

```python
nn.Sequential(
    nn.Linear(128, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, n_tissues)
)
```

**Output**: Tissue weights via Gumbel-Softmax
- Soft routing: All tissues contribute
- Differentiable: Enables end-to-end training
- Temperature-controlled: Learnable τ parameter

### 4. Hierarchical Attention

For each tissue independently:

```python
# Multi-head attention
Q = Linear(patient_enc)  # Query
K = Linear(tissue_cells) # Keys
V = Linear(tissue_cells) # Values

Attention = Softmax(QK^T / √d)
Output = Attention × V
```

**Key Feature**: Attention computed **within** each tissue, not across all cells

### 5. Weighted Combination

```python
weighted_repr = []
for tissue_idx in range(n_tissues):
    tissue_repr = attention(patient, tissue_cells[tissue_idx])
    weighted = tissue_weights[:, tissue_idx] * tissue_repr
    weighted_repr.append(weighted)

final_repr = sum(weighted_repr)
```

**Purpose**: Combine tissue-specific representations weighted by relevance

## Data Flow Example

### Input
- Patient: Breast cancer, 1817 genes
- Cell Lines: 673 lines across 24 tissues

### Step 1: Encoding
- Patient → [1, 128]
- Cell Lines → [673, 128]

### Step 2: Routing
- Tissue Router(patient_enc) → [1, 24]
- Example: [0.6 breast, 0.2 ovary, 0.1 lung, 0.1 other...]

### Step 3: Hierarchical Attention
- Breast tissue (50 cells): Attention → [1, 128]
- Ovary tissue (30 cells): Attention → [1, 128]
- Lung tissue (100 cells): Attention → [1, 128]
- ...

### Step 4: Weighting
- Breast_weighted = 0.6 × [1, 128]
- Ovary_weighted = 0.2 × [1, 128]
- Lung_weighted = 0.1 × [1, 128]
- ...

### Step 5: Combination
- Final = sum(all weighted) → [1, 128]

### Output
- Final representation for drug response prediction

## Loss Functions

```python
total_loss = (
    α × center_loss         # Cluster by tissue
  + β × routing_loss        # Route to correct tissue
  + γ × entropy_loss        # Decisive routing
  + δ × diversity_loss      # Use different tissues
  + ε × temperature_loss    # Anneal temperature
)
```

### Center Loss
```python
# Pull same-tissue samples together
distance = ||patient_enc - tissue_center[label]||²
```

### Routing Loss
```python
# Cross-entropy with true tissue
loss = -log(tissue_weights[true_tissue])
```

### Entropy Loss
```python
# Encourage decisive routing
entropy = -Σ(p × log(p))
loss = mean(entropy)
```

### Diversity Loss
```python
# Different patients use different tissues
batch_dist = mean(tissue_weights, dim=0)
diversity = -Σ(batch_dist × log(batch_dist))
loss = -diversity  # Maximize diversity
```

## Comparison: Flat vs. Hierarchical

### Flat (Original THERAPI)
```
Patient → Encoder → Attention(all 673 cells) → Repr
                    O(673) attention operations
```

### Hierarchical (This Work)
```
Patient → Encoder → Router → [24 tissue groups]
                              ↓
                    Attention(~28 cells per tissue)
                    24 × O(28) = O(672) with parallelism
                    Effective: O(28) with proper implementation
```

**Speedup**: ~24× theoretical, ~1.5-2× practical (due to overhead)

## Tissue Groups (24)

```
1.  breast          7.  pancreas       13. prostate       19. uterus
2.  lung            8.  ovary          14. bladder        20. head_neck
3.  colon           9.  kidney         15. thyroid        21. biliary
4.  blood          10.  liver          16. bone           22. testis
5.  brain          11.  stomach        17. soft_tissue    23. adrenal
6.  skin           12.  esophagus      18. cervix         24. other
```

## Key Advantages

1. **Single Model**: One model handles all tissues
2. **Interpretable**: Tissue weights show reasoning
3. **Efficient**: Parallel tissue processing
4. **Adaptive**: Learns tissue relationships
5. **Robust**: Handles metastatic/mixed samples

## Training Process

```
Epoch 1:
  Temperature: 1.0 (explore)
  Routing: Uniform-ish
  Attention: Dispersed

Epoch 50:
  Temperature: 0.7 (annealing)
  Routing: Getting specific
  Attention: More focused

Epoch 200:
  Temperature: 0.5 (exploit)
  Routing: Decisive
  Attention: Tissue-specific
```

## Evaluation Metrics

### 1. Prediction Performance
- AUROC (drug response)
- Mann-Whitney U (TRANSACT)
- Multiple testing correction

### 2. Routing Quality
- Accuracy (% correct tissue)
- Entropy (decisiveness)
- Confusion matrix

### 3. Computational
- Inference time
- Memory usage
- Speedup vs. flat

### 4. Interpretability
- Tissue weight analysis
- Attention heatmaps
- Biological validation

## Implementation Highlights

### Gumbel-Softmax Temperature

```python
# Start high (explore), end low (exploit)
τ_initial = 1.0
τ_final = 0.5
τ_t = τ_initial × decay_rate^t

# Or learnable
τ = nn.Parameter(torch.ones(1))
```

### Masking for Tissues

```python
# Binary mask [n_tissues, n_cell_lines]
tissue_mask[i, j] = 1 if cell_j in tissue_i else 0

# Use in attention
valid_cells = cell_lines[tissue_mask[tissue_idx].bool()]
```

### Multi-Head Attention

```python
# 4 heads × 32 dimensions = 128 total
n_heads = 4
head_dim = 128 // 4 = 32

# Parallel attention across heads
# Then concatenate and project
```

## Code Entry Points

### Training
```python
train_hierarchical.py → train_hierarchical_aligner()
    → HierarchicalTHERAPI.forward()
        → tissue_router()
        → hierarchical_attention_forward()
```

### Inference
```python
model.eval()
output = model.hierarchical_attention_forward(
    patient_expr,
    cell_line_exprs,
    return_attention=True
)
# Returns: representation, tissue_weights, attention_breakdown
```

## Future Extensions

1. **Multi-Modal**: Add mutations, CNV
2. **Uncertainty**: Bayesian routing
3. **Hierarchical Tissues**: Sub-tissue grouping
4. **Transfer Learning**: Pre-train on large datasets
5. **Active Learning**: Query uncertain samples

---

For implementation details, see source code in `models/hierarchical_therapi.py`
