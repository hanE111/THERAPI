# Data Format Fix - Gzip-Compressed Pickle Files

## Issue

```
_pickle.UnpicklingError: invalid load key, '\x1f'.
```

This error occurs when trying to load gzip-compressed pickle files with standard `pickle.load()`.

## Root Cause

The pickle files in TRANSACT data are **gzip-compressed** (`.pkl.gz` stored as `.pkl`).

The magic bytes `\x1f\x8b` indicate a gzip-compressed file.

## Solution Applied

### Added `load_pickle_file()` Helper Function

Located in `hierarchical_utils/data_loader.py`:

```python
def load_pickle_file(filepath: str):
    """
    Load pickle file, handling both compressed and uncompressed formats.

    Automatically detects gzip compression by checking magic bytes.
    """
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b'\x1f\x8b':  # Gzip magic number
            with gzip.open(filepath, 'rb') as gz:
                return pickle.load(gz)
        else:
            return pickle.load(f)
```

### Updated All Pickle Loading

Replaced all `pickle.load(f)` calls with `load_pickle_file(path)`:

**Files Updated:**
- ✅ `load_gdsc_data()` - GDSC expression data
- ✅ `load_tcga_data()` - TCGA expression and annotations
- ✅ `load_pdx_data()` - PDX expression data

## How It Works

1. **Open file in binary mode**
2. **Read first 2 bytes** (magic number)
3. **Check if gzip** (`\x1f\x8b`)
4. **Load accordingly**:
   - If gzipped: Use `gzip.open()` then `pickle.load()`
   - If not: Use standard `pickle.load()`

## Benefits

- ✅ **Automatic detection** - No manual specification needed
- ✅ **Backward compatible** - Works with uncompressed pickles too
- ✅ **Transparent** - No changes needed in calling code
- ✅ **Efficient** - Only reads 2 bytes to detect format

## Usage

```python
from hierarchical_utils.data_loader import load_pickle_file

# Works with both compressed and uncompressed
data = load_pickle_file('data.pkl')  # Auto-detects format
```

## Testing

```python
# Test with your data
from hierarchical_utils.data_loader import load_pickle_file
import pandas as pd

# Load GDSC data
gdsc_expr = load_pickle_file('data/GDSC/rnaseq/GDSC_rnaseq_data.pkl')
print(f"✓ Loaded: {type(gdsc_expr)}, shape: {gdsc_expr.shape}")

# Load TCGA data
tcga_expr = load_pickle_file('data/TCGA/rnaseq/TCGA_rnaseq_data.pkl')
print(f"✓ Loaded: {type(tcga_expr)}, shape: {tcga_expr.shape}")
```

## Common Pickle Formats

| Magic Bytes | Format | Handler |
|-------------|--------|---------|
| `\x1f\x8b` | Gzip | `gzip.open()` |
| `\x80\x02` | Pickle protocol 2 | `pickle.load()` |
| `\x80\x03` | Pickle protocol 3 | `pickle.load()` |
| `\x80\x04` | Pickle protocol 4 | `pickle.load()` |
| `\x80\x05` | Pickle protocol 5 | `pickle.load()` |

Our function handles all of these automatically!

## Alternative: Manual Gzip

If you prefer explicit handling:

```python
import gzip
import pickle

# Explicit gzip
with gzip.open('data.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# Or use our helper (recommended)
data = load_pickle_file('data.pkl.gz')  # Same result!
```

## Status

✅ **FIXED** - All pickle loading now handles gzip compression automatically

## Related Issues

- Import conflict: Fixed in [FINAL_FIX.md](FINAL_FIX.md)
- General troubleshooting: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Date**: November 2024
**Issue**: Gzip-compressed pickle files
**Solution**: Auto-detecting `load_pickle_file()` helper
