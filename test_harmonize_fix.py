"""
Test the gene harmonization fix
"""
import os
import sys

# Add hierarchical_utils to path
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_utils.data_loader import TransactDataLoader

# Initialize data loader
data_loader = TransactDataLoader(data_root='data/')

print("=" * 80)
print("Testing Gene Harmonization Fix")
print("=" * 80)

# Load data
print("\nLoading GDSC data...")
gdsc_data = data_loader.load_gdsc_data()
print(f"✓ GDSC expression shape: {gdsc_data['expression'].shape}")

print("\nLoading TCGA data...")
tcga_data = data_loader.load_tcga_data()
print(f"✓ TCGA expression shape: {tcga_data['expression'].shape}")

# Test harmonization
print("\nTesting harmonize_genes()...")
common_genes = data_loader.harmonize_genes(
    gdsc_data['expression'],
    tcga_data['expression']
)

print(f"\n{'='*80}")
print(f"RESULT: Found {len(common_genes)} common genes")
print(f"{'='*80}")

if len(common_genes) > 0:
    print(f"\n✓ SUCCESS! Gene harmonization working correctly")
    print(f"\nFirst 10 common genes:")
    for i, gene in enumerate(common_genes[:10], 1):
        print(f"  {i}. {gene}")
    print(f"\nLast 10 common genes:")
    for i, gene in enumerate(common_genes[-10:], len(common_genes)-9):
        print(f"  {i}. {gene}")
else:
    print(f"\n✗ FAILED! Still finding 0 common genes")

print(f"\n{'='*80}")
