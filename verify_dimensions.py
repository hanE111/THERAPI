"""
Verify that tissue mask dimensions match expression data
"""
import os
import sys
import torch

# Add paths
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_utils.data_loader import TransactDataLoader
from hierarchical_utils.tissue_mapping import TissueMapper

print("=" * 80)
print("Verifying Tissue Mask Dimensions")
print("=" * 80)

# Load data
data_loader = TransactDataLoader(data_root='data/')
tissue_mapper = TissueMapper()

# Load GDSC
print("\n1. Loading GDSC data...")
gdsc_data = data_loader.load_gdsc_data()
source_expr = gdsc_data['expression']
source_tissues = gdsc_data['tissue_mapping']

print(f"   Expression shape: {source_expr.shape}")
print(f"   Expression samples: {len(source_expr.index)}")
print(f"   Tissue mapping entries: {len(source_tissues)}")
print(f"   Sample expression indices: {list(source_expr.index)[:5]}")
print(f"   Sample tissue keys: {list(source_tissues.keys())[:5]}")

# Load TCGA
print("\n2. Loading TCGA data...")
tcga_data = data_loader.load_tcga_data()
target_expr = tcga_data['expression']

print(f"   Expression shape: {target_expr.shape}")

# Harmonize genes
print("\n3. Harmonizing genes...")
common_genes = data_loader.harmonize_genes(source_expr, target_expr)
print(f"   Common genes: {len(common_genes)}")

source_expr = source_expr[common_genes]
target_expr = target_expr[common_genes]

# Create tissue mask - OLD WAY (WRONG)
print("\n4. Creating tissue mask - OLD WAY (using all metadata)...")
tissue_cell_mask_old, _ = tissue_mapper.create_cell_line_tissue_matrix(source_tissues)
print(f"   ❌ Mask shape: {tissue_cell_mask_old.shape}")
print(f"   ❌ Expected to match: [24, {len(source_expr)}]")
print(f"   ❌ Actual: [24, {tissue_cell_mask_old.shape[1]}]")
print(f"   ❌ MISMATCH: {tissue_cell_mask_old.shape[1]} != {len(source_expr)}")

# Create tissue mask - NEW WAY (CORRECT)
print("\n5. Creating tissue mask - NEW WAY (filtered to expression data)...")
source_tissues_filtered = {idx: source_tissues[idx] for idx in source_expr.index if idx in source_tissues}
print(f"   Filtered tissue mapping: {len(source_tissues_filtered)} entries")
tissue_cell_mask_new, _ = tissue_mapper.create_cell_line_tissue_matrix(source_tissues_filtered)
print(f"   ✓ Mask shape: {tissue_cell_mask_new.shape}")
print(f"   ✓ Expected: [24, {len(source_expr)}]")
print(f"   ✓ Match: {tissue_cell_mask_new.shape[1]} == {len(source_expr)}")

# Verify tensor operations will work
print("\n6. Testing tensor operations...")
tissue_cell_mask_tensor = torch.tensor(tissue_cell_mask_new, dtype=torch.float32)
batch_size = 32
dummy_cell_line_encs = torch.randn(len(source_expr), 128)  # [n_cell_lines, n_latent]

print(f"   Mask tensor shape: {tissue_cell_mask_tensor.shape}")
print(f"   Cell line encodings shape: {dummy_cell_line_encs.shape}")

# Test tissue masking
for tissue_idx in range(tissue_mapper.n_tissue_groups):
    tissue_mask = tissue_cell_mask_tensor[tissue_idx].bool()
    if tissue_mask.any():
        tissue_cell_encs = dummy_cell_line_encs[tissue_mask]
        print(f"   ✓ Tissue {tissue_idx}: mask size {tissue_mask.sum().item()}, selected {tissue_cell_encs.shape[0]} cells")
        break

print("\n" + "=" * 80)
print("✓ All dimension checks passed!")
print("=" * 80)
