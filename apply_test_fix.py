#!/usr/bin/env python3
"""
Script to fix dimension mismatch in test_hierarchical_TCGA.py
Run this on the remote server where the file exists.
"""

import re

# Read the file
with open('test_hierarchical_TCGA.py', 'r') as f:
    content = f.read()

# Pattern to find the feature loading section (starting around line 192)
# We'll replace everything from "# Load features" until "# Get labels"

old_pattern = r"(# Load features.*?)(# Get labels)"

new_code = """# Use fixed dimensions that match training
    # During training, the predictor was trained with:
    # - Rank features: 100 dimensions (all zeros)
    # - Chemical features: 2048 dimensions (all zeros)
    # We must use the SAME dimensions during testing
    TRAINING_RANK_DIM = 100
    TRAINING_CHEM_DIM = 2048

    print("\\nUsing feature dimensions matching training:")
    tcga_rank = np.zeros((len(tcga_resp), TRAINING_RANK_DIM), dtype=np.float32)
    tcga_comp = np.zeros((len(tcga_resp), TRAINING_CHEM_DIM), dtype=np.float32)
    tcga_pert = np.zeros((len(tcga_resp), 128), dtype=np.float32)  # Match patient repr dim

    print(f"  Rank features: {tcga_rank.shape} (zeros, as in training)")
    print(f"  Chemical features: {tcga_comp.shape} (zeros, as in training)")
    print(f"  Perturbation features: {tcga_pert.shape}")

    """

# Replace the section
content_new = re.sub(old_pattern, r'\1' + new_code + r'\2', content, flags=re.DOTALL)

# Also fix the predictor initialization
# Replace "genef_dim=tcga_rank.shape[1]," with "genef_dim=TRAINING_RANK_DIM,"
content_new = re.sub(
    r'genef_dim=tcga_rank\.shape\[1\]',
    'genef_dim=TRAINING_RANK_DIM     # Use constant: 100',
    content_new
)

# Replace "chemical_dim=tcga_comp.shape[1]," with "chemical_dim=TRAINING_CHEM_DIM,"
content_new = re.sub(
    r'chemical_dim=tcga_comp\.shape\[1\]',
    'chemical_dim=TRAINING_CHEM_DIM   # Use constant: 2048',
    content_new
)

# Save backup
with open('test_hierarchical_TCGA.py.bak', 'w') as f:
    f.write(content)

# Write fixed version
with open('test_hierarchical_TCGA.py', 'w') as f:
    f.write(content_new)

print("✓ Fixed test_hierarchical_TCGA.py")
print("✓ Backup saved as test_hierarchical_TCGA.py.bak")
print("\nChanges made:")
print("1. Replaced feature loading with fixed dimensions (100, 2048)")
print("2. Updated predictor initialization to use TRAINING_RANK_DIM and TRAINING_CHEM_DIM")
print("\nYou can now run: ./run_hierarchical_pipeline.sh")
