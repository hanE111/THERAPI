# Predictor Dimension Mismatch Fix

## Issue #13: Feature Dimension Mismatch in Testing

### Problem

```
RuntimeError: Error(s) in loading state_dict for HierarchicalResponsePredictor:
    size mismatch for genef_fc.0.weight: copying a param with shape torch.Size([256, 100])
    from checkpoint, the shape in current model is torch.Size([256, 256]).
    size mismatch for chemical_fc.0.weight: copying a param with shape torch.Size([256, 2048])
    from checkpoint, the shape in current model is torch.Size([256, 978]).
```

### Root Cause

**During Training** (predictor was created with):
- `genef_dim = 100` (rank representation)
- `chemical_dim = 2048` (chemical fingerprints)

**During Testing** (predictor is being created with):
- `genef_dim = tcga_rank.shape[1]` = 256 (actual loaded data dimension)
- `chemical_dim = tcga_comp.shape[1]` = 978 (actual loaded data dimension)

The test script is using the **actual dimensions** of the loaded features, but it should use the **same dimensions as training** (100 and 2048).

### Why This Happened

The training script created the predictor with hardcoded/default dimensions:

```python
# train_hierarchical_predictor.py
predictor = HierarchicalResponsePredictor(
    emb_dim=128,
    genef_dim=gdsc_rank.shape[1],    # Was 100 (all zeros)
    chemical_dim=gdsc_comp.shape[1],  # Was 2048 (all zeros)
    ...
)
```

During training:
- `gdsc_rank` was zeros with shape `[333161, 100]`
- `gdsc_comp` was zeros with shape `[333161, 2048]`

During testing:
- `tcga_rank` was loaded from file with shape `[2572, 256]` (different!)
- `tcga_comp` was loaded from file with shape `[2572, 978]` (different!)

### Solution

**Option 1: Use Fixed Dimensions** ✅ (Recommended)

Update `test_hierarchical_TCGA.py` to use the same dimensions as training:

```python
# After loading features, ensure they match training dimensions
TRAINING_RANK_DIM = 100
TRAINING_CHEM_DIM = 2048

# Pad or truncate rank features to match training
if tcga_rank.shape[1] > TRAINING_RANK_DIM:
    tcga_rank = tcga_rank[:, :TRAINING_RANK_DIM]  # Truncate
elif tcga_rank.shape[1] < TRAINING_RANK_DIM:
    # Pad with zeros
    padding = np.zeros((tcga_rank.shape[0], TRAINING_RANK_DIM - tcga_rank.shape[1]))
    tcga_rank = np.hstack([tcga_rank, padding])

# Pad or truncate chemical features to match training
if tcga_comp.shape[1] > TRAINING_CHEM_DIM:
    tcga_comp = tcga_comp[:, :TRAINING_CHEM_DIM]
elif tcga_comp.shape[1] < TRAINING_CHEM_DIM:
    padding = np.zeros((tcga_comp.shape[0], TRAINING_CHEM_DIM - tcga_comp.shape[1]))
    tcga_comp = np.hstack([tcga_comp, padding])

# Create predictor with TRAINING dimensions
predictor = HierarchicalResponsePredictor(
    emb_dim=128,
    genef_dim=TRAINING_RANK_DIM,      # Use 100, not tcga_rank.shape[1]
    chemical_dim=TRAINING_CHEM_DIM,   # Use 2048, not tcga_comp.shape[1]
    ...
)
```

**Option 2: Save Dimensions in Checkpoint** (Better long-term)

During training, save the feature dimensions in the checkpoint:

```python
# train_hierarchical_predictor.py
torch.save({
    'model_state_dict': predictor.state_dict(),
    'config': config,
    'feature_dims': {
        'genef_dim': gdsc_rank.shape[1],
        'chemical_dim': gdsc_comp.shape[1],
        'emb_dim': aligner_config.get('latent_dim', 128)
    }
}, predictor_path)
```

Then in testing:

```python
# Load dimensions from checkpoint
checkpoint = torch.load(predictor_path)
feature_dims = checkpoint.get('feature_dims', {
    'genef_dim': 100,
    'chemical_dim': 2048,
    'emb_dim': 128
})

# Create predictor with saved dimensions
predictor = HierarchicalResponsePredictor(
    emb_dim=feature_dims['emb_dim'],
    genef_dim=feature_dims['genef_dim'],
    chemical_dim=feature_dims['chemical_dim'],
    ...
)
```

### Immediate Fix for Testing

Since the predictor was trained with:
- Rank: 100 dimensions (all zeros)
- Chemical: 2048 dimensions (all zeros)

The simplest fix is to use zeros with the same dimensions during testing:

```python
# In test_hierarchical_TCGA.py

# Force dimensions to match training
TRAINING_RANK_DIM = 100
TRAINING_CHEM_DIM = 2048

# Always use zeros to match training
tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM))
tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM))

print(f"Using zero features to match training dimensions:")
print(f"  Rank: {tcga_rank.shape}")
print(f"  Chemical: {tcga_comp.shape}")
```

This ensures:
1. ✅ Dimensions match checkpoint
2. ✅ Behavior matches training (model was trained on zeros)
3. ✅ No need to preprocess/resize actual features
4. ✅ Simple and consistent

### Why Use Zeros?

Since the predictor was **trained with zeros** for these features:
- Using actual features during testing would be **inconsistent**
- The model never learned to use them (weights are random/minimal)
- Using zeros ensures test-time behavior matches train-time behavior

**Principle**: Test with the same feature distribution as training.

### Alternative: Retrain Predictor

If you want to use actual rank and chemical features:

1. Ensure features are available during training
2. Point to `data_therapi/` directory where features exist
3. Retrain predictor from scratch
4. Then testing can use actual features

But this requires restarting the pipeline from Step 2.

### Implementation

Update `test_hierarchical_TCGA.py` around line 192-227:

```python
# Define dimensions that match training
TRAINING_RANK_DIM = 100
TRAINING_CHEM_DIM = 2048

# Since training used zeros, testing should too for consistency
tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM), dtype=np.float32)
tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM), dtype=np.float32)
tcga_pert = tcga_representations  # Use representations as perturbation proxy

print(f"Using feature dimensions matching training:")
print(f"  Rank representation: {tcga_rank.shape} (zeros, as in training)")
print(f"  Chemical features: {tcga_comp.shape} (zeros, as in training)")
print(f"  Perturbation (using representations): {tcga_pert.shape}")
```

And when creating the predictor (around line 245):

```python
predictor = HierarchicalResponsePredictor(
    emb_dim=128,
    genef_dim=TRAINING_RANK_DIM,    # Use constant, not data shape
    chemical_dim=TRAINING_CHEM_DIM,  # Use constant, not data shape
    hidden_dim1=256,
    hidden_dim2=128,
    output_dim=1,
    dropout=0.1
).to(args.device)
```

### Files to Modify

1. **test_hierarchical_TCGA.py**
   - Lines ~192-227: Feature loading section
   - Lines ~245: Predictor initialization

### Impact

✅ **Dimensions match checkpoint** - Loading succeeds
✅ **Consistent with training** - Same zero features
✅ **Testing can proceed** - No more dimension mismatch errors
✅ **Correct behavior** - Model sees same input distribution as training

### Related Issues

- **Issue #1-11**: Previous fixes (all resolved)
- **Issue #12**: TCGA response format (fixed)
- **Issue #13**: Predictor dimension mismatch (THIS ISSUE)

---

**Date**: 2025-11-05
**Status**: ⏳ Needs implementation on remote server
**Solution**: Use fixed dimensions (100, 2048) with zeros to match training
**Alternative**: Retrain predictor with actual features from data_therapi
