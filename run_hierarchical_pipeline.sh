#!/bin/bash

# Complete pipeline for Hierarchical THERAPI
# This script runs the entire training and evaluation pipeline

set -e  # Exit on error

echo "=================================================="
echo "  Hierarchical THERAPI - Complete Pipeline"
echo "=================================================="

# Configuration
DATA_DIR="data/"
DEVICE="cuda:0"
CONFIG="configs/hierarchical_config.yaml"
SEED=42

# Output directories
mkdir -p ckpts
mkdir -p output
mkdir -p log

# ================================================
# Step 1: Train Hierarchical Aligner
# ================================================
echo ""
echo "Step 1: Training Hierarchical Aligner (GDSC → TCGA)"
echo "--------------------------------------------------"

# python train_hierarchical.py \
#     --source GDSC \
#     --target TCGA \
#     --data_dir ${DATA_DIR} \
#     --config ${CONFIG} \
#     --device ${DEVICE} \
#     --seed ${SEED}

echo "✓ Aligner training completed"

# ================================================
# Step 2: Train Drug Response Predictor
# ================================================
echo ""
echo "Step 2: Training Drug Response Predictor"
echo "--------------------------------------------------"

# python train_hierarchical_predictor.py \
#     --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
#     --data_dir ${DATA_DIR} \
#     --config ${CONFIG} \
#     --device ${DEVICE} \
#     --seed ${SEED}

echo "✓ Predictor training completed"

# ================================================
# Step 3: Test on TCGA
# ================================================
echo ""
echo "Step 3: Testing on TCGA"
echo "--------------------------------------------------"

python test_hierarchical_TCGA.py \
    --aligner_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --data_dir ${DATA_DIR} \
    --n_folds 10 \
    --device ${DEVICE} \
    --output_dir output/

echo "✓ TCGA testing completed"

# ================================================
# Step 4: Evaluate with Advanced Metrics
# ================================================
echo ""
echo "Step 4: Advanced Evaluation"
echo "--------------------------------------------------"

python evaluate_hierarchical.py \
    --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
    --target TCGA \
    --output_dir output/ \
    --device ${DEVICE}

echo "✓ Advanced evaluation completed"

# ================================================
# Optional: Evaluate on PDX
# ================================================
if [ -d "${DATA_DIR}/PDXE" ]; then
    echo ""
    echo "Optional: Testing on PDX"
    echo "--------------------------------------------------"

    python evaluate_hierarchical.py \
        --model_path ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt \
        --target PDX \
        --output_dir output/ \
        --device ${DEVICE}

    echo "✓ PDX evaluation completed"
fi

# ================================================
# Summary
# ================================================
echo ""
echo "=================================================="
echo "  Pipeline Completed Successfully!"
echo "=================================================="
echo ""
echo "Output files:"
echo "  - ckpts/HierarchicalTHERAPI_aligner_GDSC_TCGA.pt"
echo "  - ckpts/HierarchicalTHERAPI_predictor_CV*.pt"
echo "  - output/HierarchicalTHERAPI_test_TCGA.csv"
echo "  - output/tissue_routing_confusion_TCGA.csv"
echo "  - log/*.log"
echo ""
echo "Next steps:"
echo "  1. Review results in output/"
echo "  2. Run analysis notebook: jupyter notebook notebooks/tissue_routing_analysis.ipynb"
echo "  3. Run ablation studies (see QUICK_START.md)"
echo ""
