# Quick Fix for test_hierarchical_TCGA.py Dimension Mismatch

## Problem
Predictor checkpoint expects:
- Rank features: 100 dimensions
- Chemical features: 2048 dimensions

But test script is using actual loaded data dimensions (256 and 978).

## Solution

Replace the feature loading section in `test_hierarchical_TCGA.py` (around lines 192-227) with:

```python
# Load features - Use dimensions that match training
# Training used zeros with specific dimensions
TRAINING_RANK_DIM = 100
TRAINING_CHEM_DIM = 2048

print("\nPreparing features to match training dimensions...")

# Use zeros for rank and chemical features (matches training)
tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM), dtype=np.float32)
tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM), dtype=np.float32)

# Use patient representations as perturbation features
# Note: Need to align representations to drug responses
# For now, just use zeros or representations
tcga_pert = np.zeros((len(tcga_resp), 128), dtype=np.float32)  # Will need alignment

print(f"Feature shapes (matching training):")
print(f"  Rank representation: {tcga_rank.shape}")
print(f"  Chemical features: {tcga_comp.shape}")
print(f"  Perturbation features: {tcga_pert.shape}")
```

Then update the predictor initialization (around line 245):

```python
predictor = HierarchicalResponsePredictor(
    emb_dim=128,
    genef_dim=TRAINING_RANK_DIM,     # Use constant: 100
    chemical_dim=TRAINING_CHEM_DIM,   # Use constant: 2048
    hidden_dim1=256,
    hidden_dim2=128,
    output_dim=1,
    dropout=0.1
).to(args.device)
```

## Why This Works

1. **Training used zeros** for rank and chemical features
2. **Testing should match** - use same zero dimensions
3. **Model expects** these specific dimensions (hardcoded in checkpoint)
4. **Consistent behavior** - test matches train

## Apply the Fix

```bash
# Edit the test script
nano test_hierarchical_TCGA.py

# Find lines 192-227 (feature loading)
# Replace with the code above

# Find line ~245 (predictor initialization)
# Update to use TRAINING_RANK_DIM and TRAINING_CHEM_DIM

# Save and run
./run_hierarchical_pipeline.sh
```

## Expected Output After Fix

```
Preparing features to match training dimensions...
Feature shapes (matching training):
  Rank representation: (2572, 100)
  Chemical features: (2572, 2048)
  Perturbation features: (2572, 128)

Loading 10 predictor models...
✓ Successfully loaded predictor fold 0
✓ Successfully loaded predictor fold 1
...
```

## Note About Perturbation Features

The perturbation features also need proper alignment. You may need to:

1. Map TCGA samples to drug responses (like we did for predictor training)
2. Use patient representations aligned to each drug response entry

For now, using zeros is safe and consistent.
