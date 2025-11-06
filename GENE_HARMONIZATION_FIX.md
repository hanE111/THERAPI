# Gene Harmonization Fix

## Issue #4: Zero Common Genes Error

### Problem
When running the Hierarchical THERAPI pipeline, the training script failed with:
```
[18:39:31] Using 0 common genes
...
ZeroDivisionError: float division by zero
```

This occurred at [train_hierarchical.py:203](train_hierarchical.py#L203) when initializing the `HierarchicalTHERAPI` model with `n_genes=0`, which then failed when trying to create a `Linear` layer with zero input features.

### Root Cause
The `harmonize_genes()` method in [hierarchical_utils/data_loader.py](hierarchical_utils/data_loader.py) was looking for a column named `'gene_symbol'` in the cancer genes CSV file, but the actual column name was `'Hugo'`.

**Data Structure:**
- `data/mini_cancer_genes.csv` has columns: `['Unnamed: 0', 'Hugo']`
- The method only checked for `'gene_symbol'`, not `'Hugo'`
- This caused it to fall back to `iloc[:, 0]` which selected the index column instead of gene names
- Result: No genes matched between GDSC and TCGA datasets

### Actual Data
Debug analysis revealed:
- **GDSC expression**: 1049 samples × 1780 genes (gene symbols as column names)
- **TCGA expression**: 11093 samples × 1774 genes (gene symbols as column names)
- **Cancer genes file**: 1815 genes in 'Hugo' column
- **Actual overlap**: 1774 common genes exist (GDSC ∩ TCGA ∩ Cancer genes)

### Solution

#### 1. Enhanced Column Name Detection
Updated `harmonize_genes()` method to check multiple common column name variants:

```python
# Try multiple common column names for cancer genes
if self.cancer_genes is not None:
    gene_col = None
    for col_name in ['Hugo', 'gene_symbol', 'gene', 'symbol', 'Gene', 'SYMBOL']:
        if col_name in self.cancer_genes.columns:
            gene_col = col_name
            break

    if gene_col:
        common_genes = set(self.cancer_genes[gene_col])
    else:
        # Fallback to first column if no recognized column name
        common_genes = set(self.cancer_genes.iloc[:, 0])
```

#### 2. Fixed DtypeWarning
Added `low_memory=False` parameter when loading tissue info CSV to prevent mixed type warnings:

```python
tissue_info = pd.read_csv(tissue_path, low_memory=False)
```

### Result
✅ **Gene harmonization now correctly finds 1774 common genes**

Test output:
```
RESULT: Found 1774 common genes

First 10 common genes:
  1. AATK
  2. ABCA1
  3. ABL1
  4. ABL2
  5. ACTR2
  ...
```

### Files Modified
- [hierarchical_utils/data_loader.py](hierarchical_utils/data_loader.py) (lines 292-339)
  - Enhanced `harmonize_genes()` method with robust column name detection
  - Added `low_memory=False` to tissue_info CSV loading

### Testing
Created test scripts to verify the fix:
- `debug_genes_quick.py` - Quick data structure inspection
- `test_harmonize_fix.py` - Comprehensive harmonization test

### Impact
This fix ensures that:
1. Gene harmonization works with various gene annotation formats (Hugo, gene_symbol, etc.)
2. The model can now be initialized with the correct number of genes (1774)
3. Training can proceed without ZeroDivisionError
4. Code is more robust to different data formats and column naming conventions

### Related Issues
- **Issue #1**: Import conflict (fixed by renaming `utils/` → `hierarchical_utils/`)
- **Issue #2**: Gzip pickle files (fixed by adding auto-detection)
- **Issue #3**: NumPy 2.x incompatibility (fixed by pinning to numpy<2.0)
- **Issue #4**: Zero common genes (FIXED by this change)

---

**Date**: 2025-11-05
**Status**: ✅ Fixed and tested
