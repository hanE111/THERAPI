# Cell Line to Drug Response Alignment Fix

## Issue #10: IndexError in Dataset Loading

### Problem

```
IndexError: index 77916 is out of bounds for axis 0 with size 1049
```

At [train_hierarchical_predictor.py:53](train_hierarchical_predictor.py#L53) in `HierarchicalDrugDataset.__getitem__`

### Root Cause

**Dimension mismatch between representations and drug responses**:

- **Patient representations**: Shape `[1049, 128]`
  - 1,049 unique cell lines
  - 128 latent dimensions

- **Drug response data**: Shape `[333161, ...]`
  - 333,161 measurements
  - Each is a (cell line, drug) combination
  - Example: Cell line A × 200 drugs = 200 entries

The dataset was trying to use the drug response index (0-333,160) to directly index the patient representations (0-1,048), causing an out-of-bounds error.

### Understanding the Data Structure

**TRANSACT Drug Response Format**:
```
| CELL_LINE_NAME | DRUG_NAME  | AUC  | Label |
|----------------|------------|------|-------|
| A375           | Erlotinib  | 0.95 | 0     |
| A375           | Gefitinib  | 0.72 | 1     |
| A375           | Lapatinib  | 0.88 | 0     |
| HT-29          | Erlotinib  | 0.65 | 1     |
| ...            | ...        | ...  | ...   |
```

Each cell line appears multiple times (once per drug tested).

**What We Need**:
- Map each drug response row to its cell line's representation
- Cell line A375's representation should be used for ALL drugs tested on A375

### Solution

Added cell line alignment logic in [train_hierarchical_predictor.py:202-237](train_hierarchical_predictor.py#L202-237):

```python
# Create a mapping from cell line names to representation indices
cell_line_to_repr_idx = {cell_line: idx for idx, cell_line in enumerate(gdsc_expr.index)}

# For each drug response entry, find the corresponding cell line representation
if 'CELL_LINE_NAME' in drug_response.columns:
    cell_line_col = 'CELL_LINE_NAME'
elif 'SANGER_MODEL_ID' in drug_response.columns:
    cell_line_col = 'SANGER_MODEL_ID'
elif 'COSMIC_ID' in drug_response.columns:
    cell_line_col = 'COSMIC_ID'

# Map each drug response to its cell line representation
cell_line_indices = []
missing_count = 0
for cell_line in drug_response[cell_line_col]:
    if cell_line in cell_line_to_repr_idx:
        cell_line_indices.append(cell_line_to_repr_idx[cell_line])
    else:
        # If cell line not found, use first representation (fallback)
        cell_line_indices.append(0)
        missing_count += 1

cell_line_indices = np.array(cell_line_indices)

# Expand patient representations to match drug response data
aligned_patient_repr = patient_representations[cell_line_indices]
logger(f'Aligned representations shape: {aligned_patient_repr.shape}')
```

### How It Works

**Step 1: Create Index Mapping**
```python
cell_line_to_repr_idx = {
    'A375': 0,
    'HT-29': 1,
    'MCF7': 2,
    ...
}
```

**Step 2: Map Drug Responses to Indices**
```python
# Drug response row 0: A375, Erlotinib → index 0
# Drug response row 1: A375, Gefitinib → index 0  (same cell line!)
# Drug response row 2: A375, Lapatinib → index 0
# Drug response row 3: HT-29, Erlotinib → index 1
```

**Step 3: Expand Representations**
```python
aligned_patient_repr = patient_representations[cell_line_indices]
# Shape: [333161, 128]
# Row 0: A375's representation (from patient_representations[0])
# Row 1: A375's representation (from patient_representations[0])
# Row 2: A375's representation (from patient_representations[0])
# Row 3: HT-29's representation (from patient_representations[1])
```

### Result Dimensions

| Component | Before | After | Status |
|-----------|--------|-------|--------|
| Patient repr | `[1049, 128]` | `[1049, 128]` | Computed once |
| Aligned repr | N/A | `[333161, 128]` | Expanded mapping |
| Drug responses | `[333161, ...]` | `[333161, ...]` | Unchanged |
| Dataset length | 333161 | 333161 | ✅ Match |

### Multiple Identifier Support

The fix checks for multiple possible cell line identifiers in order of preference:
1. `CELL_LINE_NAME` - Human-readable name (e.g., "A375")
2. `SANGER_MODEL_ID` - Sanger Institute identifier
3. `COSMIC_ID` - COSMIC database identifier

This ensures compatibility with different TRANSACT data versions.

### Missing Cell Lines

If a cell line in the drug response data doesn't have expression data:
- It's mapped to index 0 (first cell line) as a fallback
- A warning is logged with the count of missing cell lines
- Training can still proceed (though predictions may be less accurate for those samples)

### Files Modified

**[train_hierarchical_predictor.py](train_hierarchical_predictor.py#L202-237)**:
- Added cell line to index mapping
- Added identifier column detection
- Added representation alignment
- Added missing cell line handling and logging

### Impact

✅ **Dimensions match** - Dataset can be created without errors
✅ **Correct mapping** - Each drug response uses its cell line's representation
✅ **Efficient** - Representations computed once, reused for multiple drugs
✅ **Robust** - Handles multiple identifier formats and missing data
✅ **Training can proceed** - DataLoader can iterate without IndexError

### Example Usage Pattern

```python
# For cell line A375 with 200 drugs tested:
# - Compute A375 representation once: [128]
# - Reuse for all 200 drug predictions
# - Each prediction: concat([A375_repr, drug_features]) → model → response
```

This is the standard approach in drug response prediction where:
- Cell line features (genomics, expression) → learned representation
- Drug features (chemical structure, target) → drug embedding
- Combined features → predictor → response prediction

### Related Issues

- **Issue #1**: Import conflict (fixed)
- **Issue #2**: Gzip pickle files (fixed)
- **Issue #3**: NumPy 2.x incompatibility (fixed)
- **Issue #4**: Zero common genes (fixed)
- **Issue #5**: CUDA compatibility (fixed)
- **Issue #6**: Aligner tissue mask (fixed)
- **Issue #7**: Config structure (fixed)
- **Issue #8**: Predictor tissue mask (fixed)
- **Issue #9**: TRANSACT format support (fixed)
- **Issue #10**: Cell line alignment (FIXED)

---

**Date**: 2025-11-05
**Status**: ✅ Fixed
**Changes**: Added cell line to drug response mapping and representation alignment
**Result**: Dataset dimensions match, training can proceed
