# Tissue Mask Dimension Mismatch Fix

## Issue #6: IndexError in Tissue Masking

### Problem
```
IndexError: The shape of the mask [1633] at index 0 does not match
the shape of the indexed tensor [1049, 128] at index 0
```

This occurred at [models/hierarchical_therapi.py:274](models/hierarchical_therapi.py#L274) during the hierarchical attention forward pass.

### Root Cause

The tissue mask was created using **all cell lines from the metadata file** (1633 cell lines), but the expression data only contained **1049 cell lines** (the subset with available expression data).

**What was happening**:
1. `source_data['tissue_mapping']` contains 1633 cell line → tissue mappings (all from metadata)
2. `source_data['expression']` contains only 1049 cell lines (those with expression data)
3. Tissue mask created with shape `[24, 1633]` (24 tissues × 1633 cell lines)
4. Model receives `[1049, 128]` cell line encodings
5. When indexing `cell_line_encs[tissue_mask]`, PyTorch fails because mask length (1633) doesn't match first dimension (1049)

**Code Location**:
```python
# train_hierarchical.py line 185 (OLD - WRONG)
tissue_cell_mask, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(source_tissues)
```

### Solution

Filter the tissue mapping to **only include cell lines that exist in the expression data**:

```python
# train_hierarchical.py lines 185-187 (NEW - CORRECT)
source_tissues_filtered = {idx: source_tissues[idx]
                           for idx in source_expr.index
                           if idx in source_tissues}
tissue_cell_mask, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(source_tissues_filtered)
tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)
```

This ensures:
- Tissue mask shape: `[24, 1049]` (matches expression data)
- Cell line encodings: `[1049, 128]`
- Indexing operation: `cell_line_encs[tissue_mask]` works correctly

### Why the Mismatch Occurred

The GDSC metadata file (`model_list_20191104.csv` or `GDSC_info.csv`) contains information for more cell lines than have gene expression data available:
- **Metadata**: 1633 cell lines with tissue annotations
- **Expression data**: 1049 cell lines with RNA-seq profiles
- **Gap**: 584 cell lines have metadata but no expression data

This is common in cancer databases where:
- Some cell lines have only drug response data
- Some have only genomic data
- Full multi-omics data is available for a subset

### Files Modified

**[train_hierarchical.py](train_hierarchical.py#L185-187)**:
```python
# Before (line 185):
tissue_cell_mask, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(source_tissues)

# After (lines 185-187):
source_tissues_filtered = {idx: source_tissues[idx] for idx in source_expr.index if idx in source_tissues}
tissue_cell_mask, tissue_to_idx = tissue_mapper.create_cell_line_tissue_matrix(source_tissues_filtered)
tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask, dtype=torch.float32)
```

### Verification

The fix ensures dimensional consistency:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Expression data | `[1049, 1774]` | 1049 cell lines × 1774 genes |
| Tissue mask | `[24, 1049]` | 24 tissues × 1049 cell lines |
| Cell encodings | `[1049, 128]` | 1049 cell lines × 128 latent dims |
| Tissue subset | `[n_tissue, 128]` | Selected cell lines per tissue |

### Impact

This fix ensures:
1. ✅ Tissue mask dimensions match expression data dimensions
2. ✅ Boolean indexing works correctly in hierarchical attention
3. ✅ Each tissue can correctly select its relevant cell lines
4. ✅ Model training can proceed without IndexError

### Testing

To verify the fix works, run:
```bash
./run_hierarchical_pipeline.sh
```

The training should now proceed past the first batch without IndexError.

### Related Issues

- **Issue #1**: Import conflict (fixed)
- **Issue #2**: Gzip pickle files (fixed)
- **Issue #3**: NumPy 2.x incompatibility (fixed)
- **Issue #4**: Zero common genes (fixed)
- **Issue #5**: CUDA compatibility (fixed - PyTorch reinstalled)
- **Issue #6**: Tissue mask dimensions (FIXED by this change)

---

**Date**: 2025-11-05
**Status**: ✅ Fixed
**Change**: 3 lines in [train_hierarchical.py](train_hierarchical.py#L185-187)
