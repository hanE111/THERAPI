# Manual Fix for test_hierarchical_TCGA.py

## Quick Summary
The predictor expects specific dimensions (100 for rank, 2048 for chemical) but the test script is using actual data dimensions. This causes a size mismatch error.

## Automated Fix (Easiest)

```bash
cd /data/project/haeun/THERAPI
python3 apply_test_fix.py
```

This will automatically apply all the necessary changes.

## Manual Fix (If automated doesn't work)

### Step 1: Find the Feature Loading Section

Open `test_hierarchical_TCGA.py` and find this section (around line 192):

```python
    # Load features - check both data_therapi and data directories
    # Try data_therapi first (original THERAPI format)
    rank_path = os.path.join(os.path.dirname(args.data_dir), 'data_therapi/TCGA_rankrepresentation.csv')
    ...
    [many lines of feature loading code]
    ...
```

### Step 2: Replace Entire Feature Loading Section

Replace everything from `# Load features` until `# Get labels` with:

```python
    # Use fixed dimensions that match training
    # During training, the predictor was trained with:
    # - Rank features: 100 dimensions (all zeros)
    # - Chemical features: 2048 dimensions (all zeros)
    # We must use the SAME dimensions during testing
    TRAINING_RANK_DIM = 100
    TRAINING_CHEM_DIM = 2048

    print("\nUsing feature dimensions matching training:")
    tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM), dtype=np.float32)
    tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM), dtype=np.float32)
    tcga_pert = np.zeros((len(tcga_resp), 128), dtype=np.float32)

    print(f"  Rank features: {tcga_rank.shape} (zeros, as in training)")
    print(f"  Chemical features: {tcga_comp.shape} (zeros, as in training)")
    print(f"  Perturbation features: {tcga_pert.shape}")

    # Get labels
```

### Step 3: Fix Predictor Initialization

Find this section (around line 245):

```python
        predictor = HierarchicalResponsePredictor(
            emb_dim=128,
            genef_dim=tcga_rank.shape[1],      # WRONG!
            chemical_dim=tcga_comp.shape[1],    # WRONG!
            hidden_dim1=256,
```

Replace with:

```python
        predictor = HierarchicalResponsePredictor(
            emb_dim=128,
            genef_dim=TRAINING_RANK_DIM,        # Use constant: 100
            chemical_dim=TRAINING_CHEM_DIM,     # Use constant: 2048
            hidden_dim1=256,
```

### Step 4: Save and Test

```bash
# Save the file
# Then run:
./run_hierarchical_pipeline.sh
```

## Expected Output After Fix

```
Drug response records: 2572

Using feature dimensions matching training:
  Rank features: (2572, 100) (zeros, as in training)
  Chemical features: (2572, 2048) (zeros, as in training)
  Perturbation features: (2572, 128)

Loading 10 predictor models...
✓ Loaded predictor fold 0
✓ Loaded predictor fold 1
...
```

## Why This Fix Works

1. **Training used specific dimensions**: The predictor was saved with layers expecting rank=100, chemical=2048
2. **These were zeros**: Training didn't have actual features, used zeros
3. **Test must match**: We need the same dimensions during testing
4. **Hardcoded constants**: Use `TRAINING_RANK_DIM` and `TRAINING_CHEM_DIM` instead of `.shape[1]`

## Verification

After applying the fix, check that:
- ✓ File has `TRAINING_RANK_DIM = 100`
- ✓ File has `TRAINING_CHEM_DIM = 2048`
- ✓ Predictor uses these constants, not `.shape[1]`
- ✓ Features are created with `np.zeros()` using these dimensions

## If It Still Fails

Check the exact line numbers in your file - they may be different if the file was edited. The key is:
1. Define `TRAINING_RANK_DIM = 100` and `TRAINING_CHEM_DIM = 2048`
2. Use these constants everywhere instead of `tcga_rank.shape[1]` or `tcga_comp.shape[1]`
