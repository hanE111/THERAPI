# Original THERAPI vs. Hierarchical THERAPI

## Your Question: Why No Changes to `src/` Folder?

**Short Answer**: By design! Hierarchical THERAPI **extends** the original without modifying it, following the plan's directive: *"Don't modify THERAPI's core modules - wrap and extend"*.

## Architecture Comparison

### Original THERAPI (src/)

```
src/
├── model.py
│   ├── SOURCE_AE              # Cell line autoencoder
│   ├── TARGET_weightencoder   # Patient encoder with attention
│   ├── Emb_Dis_classifier     # Embedding classifier
│   ├── Exp_Dis_classifier     # Expression classifier
│   └── Response_predictor     # Drug response predictor
├── train_aligner.py           # Step 1: Domain alignment
├── train_predictor.py         # Step 2: Response prediction
└── test_TCGA.py               # Step 3: Testing

**Approach**: Flat attention over ALL cell lines
```

### Hierarchical THERAPI (New Implementation)

```
models/
├── hierarchical_therapi.py
│   ├── TissueRouter           # NEW: Learnable routing
│   ├── HierarchicalAttention  # NEW: Multi-head within tissues
│   ├── HierarchicalTHERAPI    # NEW: Hierarchical aligner
│   └── HierarchicalResponsePredictor  # Uses THERAPI's structure

train_hierarchical.py          # NEW: Hierarchical Step 1
train_hierarchical_predictor.py # NEW: Hierarchical Step 2
test_hierarchical_TCGA.py      # NEW: Hierarchical Step 3

**Approach**: Hierarchical attention within tissues
```

## Two Ways to Use Hierarchical THERAPI

### Option A: Hierarchical Aligner + Original Predictor (Hybrid)

This is valid but less consistent:

```bash
# Step 1: Use hierarchical aligner
python train_hierarchical.py --source GDSC --target TCGA

# Step 2: Use original predictor
python src/train_predictor.py --data_dir data/

# Step 3: Test
python src/test_TCGA.py --data_dir data/
```

### Option B: Full Hierarchical Pipeline (Recommended) ✅

This is what I implemented for you:

```bash
# Complete pipeline
./run_hierarchical_pipeline.sh

# Or step by step:
python train_hierarchical.py --source GDSC --target TCGA
python train_hierarchical_predictor.py --aligner_path ckpts/...
python test_hierarchical_TCGA.py --aligner_path ckpts/...
```

## Key Differences

| Aspect | Original THERAPI | Hierarchical THERAPI |
|--------|-----------------|---------------------|
| **Files Modified** | N/A | None (extends only) |
| **Attention** | Flat (all 673 cells) | Hierarchical (24 tissues) |
| **Routing** | None | Learnable TissueRouter |
| **Models Needed** | 1 (or N for tissue-specific) | 1 (unified) |
| **Cell Line Encoder** | `SOURCE_AE` | Reuses concept in `HierarchicalTHERAPI` |
| **Patient Encoder** | `TARGET_weightencoder` | Extends with `TissueRouter` |
| **Predictor** | `Response_predictor` | `HierarchicalResponsePredictor` (similar structure) |

## What Was Reused from Original THERAPI

### Concepts Reused ✅

1. **Two-Step Training**:
   - Original: `train_aligner.py` → `train_predictor.py`
   - Hierarchical: `train_hierarchical.py` → `train_hierarchical_predictor.py`

2. **Encoder Structure**:
   - Original: `SOURCE_AE`, `TARGET_weightencoder`
   - Hierarchical: `patient_encoder`, `cell_line_encoder` (similar architecture)

3. **Attention Mechanism**:
   - Original: `Q`, `K` projections with softmax
   - Hierarchical: Extended with multi-head and hierarchy

4. **Predictor Architecture**:
   - Original: `Response_predictor` (3 branches + head)
   - Hierarchical: `HierarchicalResponsePredictor` (same structure)

5. **Loss Functions**:
   - Original: `CenterLoss`, reconstruction, classification
   - Hierarchical: Reuses `CenterLoss`, adds routing losses

6. **Utilities**:
   - Original: `utils.py` (Logger, EarlyStopper, set_seed)
   - Hierarchical: **Directly imports** from `src/utils.py`

### Code Imports from Original ✅

```python
# In train_hierarchical.py and other files:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import set_seed, Logger, EarlyStopper  # ← Original utilities
from center_loss import CenterLoss                 # ← Original loss
```

**This is why src/ wasn't modified** - we import from it!

## Why This Design?

### Advantages of Not Modifying src/

1. **Reproducibility**: Original THERAPI results remain reproducible
2. **Backward Compatibility**: Can still run original THERAPI
3. **Clear Attribution**: Original authors' code untouched
4. **Easy Comparison**: Can run both versions side-by-side
5. **Scientific Integrity**: Clear what's original vs. new contribution

### What's Truly New (Your Contribution)

1. **`models/hierarchical_therapi.py`**:
   - `TissueRouter` with Gumbel-Softmax
   - `HierarchicalAttention` with multi-head
   - Hierarchical tissue processing

2. **`models/ablation_models.py`**:
   - 8 ablation variants
   - Systematic comparison framework

3. **`utils/tissue_mapping.py`**:
   - 24 standardized tissue groups
   - Tissue hierarchy management

4. **`utils/data_loader.py`**:
   - TRANSACT data compatibility
   - Multi-dataset loading (GDSC, PDX, TCGA, HMF)

## Complete File Listing

### Original THERAPI (Unchanged)

```
src/
├── model.py                   ← Original models
├── utils.py                   ← Original utilities (imported by hierarchical)
├── center_loss.py             ← Original loss (imported by hierarchical)
├── train_aligner.py           ← Original aligner training
├── train_predictor.py         ← Original predictor training
├── test_TCGA.py               ← Original testing
└── train_aligner_tissue_specific.py  ← Tissue-specific variant
```

### Hierarchical Extension (New)

```
models/
├── hierarchical_therapi.py    ← NEW: Main contribution
└── ablation_models.py         ← NEW: Ablation variants

utils/
├── data_loader.py             ← NEW: TRANSACT compatibility
└── tissue_mapping.py          ← NEW: Tissue hierarchies

train_hierarchical.py          ← NEW: Hierarchical aligner training
train_hierarchical_predictor.py ← NEW: Hierarchical predictor training
test_hierarchical_TCGA.py      ← NEW: Hierarchical testing
evaluate_hierarchical.py       ← NEW: Advanced evaluation
run_hierarchical_pipeline.sh   ← NEW: Complete pipeline

configs/
└── hierarchical_config.yaml   ← NEW: Configuration

notebooks/
└── tissue_routing_analysis.ipynb ← NEW: Analysis
```

## When to Use Each Version

### Use Original THERAPI When:

- Reproducing original paper results
- Simple baseline comparison
- Working with original data format only
- Don't need tissue routing interpretability

### Use Hierarchical THERAPI When:

- Need tissue-aware predictions
- Want interpretable routing
- Working with TRANSACT data
- Need computational efficiency
- Handling metastatic samples
- Running ablation studies

### Use Both Together:

- Comprehensive baseline comparison
- Validate hierarchical benefits
- Publication with thorough evaluation
- Demonstrating improvement

## Example: Side-by-Side Comparison

```bash
# Terminal 1: Train original THERAPI
cd src/
python train_aligner.py --source GDSC --target TCGA
python train_predictor.py
python test_TCGA.py

# Terminal 2: Train hierarchical THERAPI
cd ..
python train_hierarchical.py --source GDSC --target TCGA
python train_hierarchical_predictor.py --aligner_path ckpts/...
python test_hierarchical_TCGA.py --aligner_path ckpts/...

# Compare results
python -c "
import pandas as pd
orig = pd.read_csv('output/THERAPI_test_TCGA.csv')
hier = pd.read_csv('output/HierarchicalTHERAPI_test_TCGA.csv')
print('Original:', orig.loc['mean', 'AUC'])
print('Hierarchical:', hier.loc['mean', 'AUC'])
"
```

## FAQ

### Q: Why create new training scripts instead of modifying originals?

**A**: To maintain original reproducibility and clearly separate contributions. Also allows running both versions.

### Q: Does Hierarchical THERAPI depend on original THERAPI?

**A**: Only minimally - imports utilities (`Logger`, `EarlyStopper`) and `CenterLoss`. Core models are independent.

### Q: Can I use only the TissueRouter in original THERAPI?

**A**: Yes! You could integrate just the routing:

```python
from models.hierarchical_therapi import TissueRouter
# Add to original TARGET_weightencoder
```

### Q: Why not modify train_aligner.py directly?

**A**: Different objectives:
- Original: Align domains with flat attention
- Hierarchical: Align domains with hierarchical routing

Both are valid, serve different purposes.

### Q: Is the predictor architecture identical?

**A**: Very similar structure (3 branches + head), but:
- Original: Uses flat attention representations
- Hierarchical: Uses hierarchical representations

### Q: Can I train only the aligner hierarchically?

**A**: Yes! Mix and match:

```bash
# Hierarchical aligner, original predictor
python train_hierarchical.py --source GDSC --target TCGA
python src/train_predictor.py
```

Though full hierarchical pipeline is recommended for consistency.

## Summary

### What You Have Now

✅ **Original THERAPI** (src/) - Untouched, fully functional
✅ **Hierarchical THERAPI** - Complete parallel implementation
✅ **Both can coexist** and run side-by-side
✅ **Clear separation** of original vs. new contributions
✅ **Imports from original** where appropriate (utilities, losses)
✅ **Complete pipeline** for both versions

### Your Next Steps

1. **Run original THERAPI** (for baseline):
   ```bash
   cd src/
   python train_aligner.py
   ```

2. **Run hierarchical THERAPI** (for comparison):
   ```bash
   ./run_hierarchical_pipeline.sh
   ```

3. **Compare results** in `output/`

4. **Run ablations** to validate contributions

5. **Publish!** With clear attribution to original authors

---

**Bottom Line**: Not modifying `src/` is intentional, proper, and follows best practices for extending existing work while maintaining reproducibility and clear attribution.
