#!/bin/bash

# Fix CUDA compatibility for RTX 3090
# This script reinstalls PyTorch with proper CUDA support

echo "=========================================="
echo "  CUDA Compatibility Fix for RTX 3090"
echo "=========================================="
echo

# Check if conda environment is active
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "❌ No conda environment active!"
    echo "Please activate your environment first:"
    echo "  conda activate therapi"
    exit 1
fi

echo "✓ Environment: ${CONDA_DEFAULT_ENV}"
echo

# Show current PyTorch version
echo "Current PyTorch installation:"
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "  PyTorch not found or error"
echo

# Prompt user for CUDA version
echo "Select CUDA version to install:"
echo "  1) CUDA 11.8 (Recommended - most stable)"
echo "  2) CUDA 12.1 (Newer, if you need latest features)"
echo "  3) Cancel"
echo
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        CUDA_VERSION="cu118"
        TORCH_VERSION="2.0.1"
        TORCHVISION_VERSION="0.15.2"
        TORCHAUDIO_VERSION="2.0.2"
        echo
        echo "Installing PyTorch ${TORCH_VERSION} with CUDA 11.8..."
        ;;
    2)
        CUDA_VERSION="cu121"
        TORCH_VERSION="2.1.0"
        TORCHVISION_VERSION="0.16.0"
        TORCHAUDIO_VERSION="2.1.0"
        echo
        echo "Installing PyTorch ${TORCH_VERSION} with CUDA 12.1..."
        ;;
    3)
        echo "Cancelled."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Uninstall current PyTorch
echo
echo "Step 1: Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio -y

# Install new PyTorch
echo
echo "Step 2: Installing PyTorch with proper CUDA support..."
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Verify installation
echo
echo "=========================================="
echo "  Verification"
echo "=========================================="
echo
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print()
    print('Testing CUDA computation...')
    x = torch.randn(3, 3).cuda()
    result = x @ x.T
    print('✓ CUDA computation successful!')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
else:
    print('❌ CUDA not available!')
"

echo
echo "=========================================="
echo "  Installation Complete"
echo "=========================================="
echo
echo "You can now run the pipeline:"
echo "  ./run_hierarchical_pipeline.sh"
echo
