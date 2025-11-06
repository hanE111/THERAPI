# NumPy 2.x Compatibility Issue - Fix

## Issue

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash.
...
AttributeError: _ARRAY_API not found
```

## Root Cause

Your environment has **NumPy 2.0.2** installed, but PyTorch was compiled against **NumPy 1.x**.

NumPy 2.0 introduced breaking changes that are incompatible with packages built for NumPy 1.x.

## Solution

### Option 1: Downgrade NumPy (Recommended - Fastest)

```bash
conda activate therapi
pip install "numpy<2"
```

This will install the latest NumPy 1.x version (likely 1.26.4).

### Option 2: Upgrade PyTorch

```bash
conda activate therapi
pip install --upgrade torch torchvision torchaudio
```

This installs the latest PyTorch that may support NumPy 2.x.

### Option 3: Clean Install (Most Reliable)

```bash
# Deactivate and remove environment
conda deactivate
conda env remove -n therapi

# Recreate with correct versions
conda create -n therapi python=3.9
conda activate therapi

# Install PyTorch first
pip install torch==2.0.1 torchvision torchaudio

# Install NumPy 1.x explicitly
pip install "numpy<2"

# Install other requirements
pip install -r requirements.txt
```

## Quick Fix (Just NumPy)

```bash
# In your therapi environment
pip install --force-reinstall "numpy<2.0"
```

## Verify Fix

After fixing, verify:

```bash
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; import numpy; print('✓ Both work together')"
```

Expected output:
```
NumPy: 1.26.4  (or similar 1.x version)
PyTorch: 2.0.1 (or similar)
✓ Both work together
```

## Run Pipeline Again

After fixing NumPy:

```bash
./run_hierarchical_pipeline.sh
```

## Why This Happens

**NumPy 2.0 Release (June 2024)**:
- Major breaking changes
- New C-API
- Incompatible with packages built for NumPy 1.x

**Affected Packages**:
- PyTorch (if older version)
- TensorFlow
- scikit-learn (older versions)
- pandas (older versions)

## Long-term Solution

Update `requirements.txt` to pin NumPy version:

```txt
# Add to requirements.txt
numpy>=1.24.4,<2.0
```

## Status After Fix

- ✅ NumPy 1.x installed
- ✅ Compatible with PyTorch
- ✅ Ready to train

---

**Recommendation**: Use Option 1 (downgrade NumPy) - it's fastest and most reliable.

```bash
pip install "numpy<2"
```
