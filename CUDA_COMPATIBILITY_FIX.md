# CUDA Compatibility Fix for RTX 3090

## Issue #5: CUDA Kernel Not Available for RTX 3090

### Problem
```
RuntimeError: CUDA error: no kernel image is available for execution on the device

UserWarning: NVIDIA GeForce RTX 3090 with CUDA capability sm_86 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
```

### Root Cause
Your PyTorch installation was compiled for older CUDA compute capabilities and doesn't include support for the RTX 3090's architecture (compute capability 8.6 / sm_86).

### System Info
- **GPU**: NVIDIA GeForce RTX 3090 (compute capability sm_86)
- **CUDA Version**: 12.2 (driver version 535.183.01)
- **Current PyTorch**: Compiled for sm_37, sm_50, sm_60, sm_70 only

### Solution

#### Option 1: Reinstall PyTorch with CUDA 11.8 (Recommended)

```bash
# Activate your environment
conda activate therapi

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 11.8 support (includes sm_86 for RTX 3090)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

#### Option 2: Install PyTorch with CUDA 12.1

```bash
# Activate your environment
conda activate therapi

# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.1 support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

#### Option 3: Use Conda (Alternative)

```bash
conda activate therapi

# Uninstall pip version
pip uninstall torch torchvision torchaudio -y

# Install via conda with CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Verification

After reinstalling, verify the installation:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch version: 2.0.1 (or 2.1.0)
CUDA available: True
CUDA version: 11.8 (or 12.1)
GPU: NVIDIA GeForce RTX 3090
```

### Test CUDA Computation

```bash
python -c "import torch; x = torch.randn(3, 3).cuda(); print(f'✓ CUDA tensor created on {torch.cuda.get_device_name(0)}'); print(x @ x.T)"
```

### Why This Happens

PyTorch wheels are compiled for specific CUDA architectures to keep download sizes manageable. The version you installed was likely:
- An older PyTorch version (< 1.8)
- A CPU-only build
- Built for older GPUs only

The RTX 3090 (Ampere architecture, launched 2020) requires compute capability 8.6, which is only included in PyTorch builds from version 1.8+ with proper CUDA support.

### GPU Selection in Your System

You have multiple GPUs:
- GPU 0: RTX 2080 Ti (sm_75) - 11GB
- GPU 1: RTX 3090 (sm_86) - 24GB ⚠️ Currently in use
- GPU 2: RTX 2080 Ti (sm_75) - 11GB
- GPU 3: RTX 2080 Ti (sm_75) - 11GB
- GPU 4: RTX 3090 (sm_86) - 24GB
- GPU 5: RTX 3090 (sm_86) - 24GB
- GPU 6: RTX 2080 Ti (sm_75) - 11GB
- GPU 7: RTX 3090 (sm_86) - 24GB

**Temporary Workaround**: If you can't reinstall PyTorch immediately, use a RTX 2080 Ti instead:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_hierarchical_pipeline.sh
# or
CUDA_VISIBLE_DEVICES=2 ./run_hierarchical_pipeline.sh
```

### Related Issues
- **Issue #1**: Import conflict (fixed)
- **Issue #2**: Gzip pickle files (fixed)
- **Issue #3**: NumPy 2.x incompatibility (fixed)
- **Issue #4**: Zero common genes (fixed)
- **Issue #5**: CUDA compatibility (THIS ISSUE)

---

**Date**: 2025-11-05
**Status**: ⏳ Requires PyTorch reinstallation
