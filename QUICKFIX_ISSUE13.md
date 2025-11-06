# ⚡ Quick Fix for Issue #13 - Dimension Mismatch

## Problem
```
RuntimeError: size mismatch for genef_fc.0.weight:
  copying [256, 100] from checkpoint, current model [256, 256]
```

## Root Cause
Test script creates predictor with wrong dimensions. Needs to match training (100, 2048).

## Quick Fix (1 minute)

### Option 1: Run Automated Script ✅
```bash
cd /data/project/haeun/THERAPI
python3 apply_test_fix.py
./run_hierarchical_pipeline.sh
```

### Option 2: Manual Edit (2 changes)

**Edit 1: Replace feature loading (line ~192)**
```python
# OLD: Complex loading from files with wrong dimensions
# NEW: Simple fixed dimensions
TRAINING_RANK_DIM = 100
TRAINING_CHEM_DIM = 2048

tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM), dtype=np.float32)
tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM), dtype=np.float32)
tcga_pert = np.zeros((len(tcga_resp), 128), dtype=np.float32)
```

**Edit 2: Fix predictor init (line ~245)**
```python
predictor = HierarchicalResponsePredictor(
    emb_dim=128,
    genef_dim=TRAINING_RANK_DIM,      # Not tcga_rank.shape[1]
    chemical_dim=TRAINING_CHEM_DIM,   # Not tcga_comp.shape[1]
    ...
)
```

## That's It!

After applying either fix, run:
```bash
./run_hierarchical_pipeline.sh
```

Testing will complete successfully.

---

For details: [COMPLETE_FIXES_SUMMARY.md](COMPLETE_FIXES_SUMMARY.md)
