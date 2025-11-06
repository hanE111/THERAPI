"""
Quick debug script to check gene column names
"""
import pandas as pd
import pickle
import gzip
import os

def load_pickle_file(filepath: str):
    """Load pickle file, handling both compressed and uncompressed formats."""
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b'\x1f\x8b':  # Gzip magic number
            with gzip.open(filepath, 'rb') as gz:
                return pickle.load(gz)
        else:
            return pickle.load(f)

# 1. Check cancer genes
print("=" * 80)
print("1. Cancer Genes")
print("=" * 80)
cancer_genes = pd.read_csv('data/mini_cancer_genes.csv')
print(f"Columns: {cancer_genes.columns.tolist()}")
print(f"Shape: {cancer_genes.shape}")
print(f"First 5: {cancer_genes.head()}")

# 2. Check GDSC expression
print("\n" + "=" * 80)
print("2. GDSC Expression")
print("=" * 80)

# Try pickle first
gdsc_pkl = 'data/GDSC/rnaseq/GDSC_rnaseq_data.pkl'
gdsc_csv = 'data/GDSC/GDSC_gex.csv'

if os.path.exists(gdsc_pkl):
    print(f"Loading from {gdsc_pkl}...")
    gdsc_expr = load_pickle_file(gdsc_pkl)
    print(f"Type: {type(gdsc_expr)}")
    print(f"Shape: {gdsc_expr.shape}")
    print(f"First 5 genes: {gdsc_expr.columns[:5].tolist()}")
    print(f"Gene types: {[type(g).__name__ for g in gdsc_expr.columns[:3]]}")
elif os.path.exists(gdsc_csv):
    print(f"Loading from {gdsc_csv}...")
    gdsc_expr = pd.read_csv(gdsc_csv, index_col=0, nrows=10)
    print(f"Shape (first 10 rows): {gdsc_expr.shape}")
    print(f"First 5 genes: {gdsc_expr.columns[:5].tolist()}")
    print(f"Gene types: {[type(g).__name__ for g in gdsc_expr.columns[:3]]}")
else:
    print("GDSC data not found!")
    gdsc_expr = None

# 3. Check TCGA expression
print("\n" + "=" * 80)
print("3. TCGA Expression")
print("=" * 80)

tcga_pkl = 'data/TCGA/rnaseq/TCGA_rnaseq_data.pkl'
tcga_csv = 'data/TCGA/TCGA_unlabeled_gex.csv'

if os.path.exists(tcga_pkl):
    print(f"Loading from {tcga_pkl}...")
    tcga_expr = load_pickle_file(tcga_pkl)
    print(f"Type: {type(tcga_expr)}")
    print(f"Shape: {tcga_expr.shape}")
    print(f"First 5 genes: {tcga_expr.columns[:5].tolist()}")
    print(f"Gene types: {[type(g).__name__ for g in tcga_expr.columns[:3]]}")
elif os.path.exists(tcga_csv):
    print(f"Loading from {tcga_csv}...")
    tcga_expr = pd.read_csv(tcga_csv, index_col=0, nrows=10)
    print(f"Shape (first 10 rows): {tcga_expr.shape}")
    print(f"First 5 genes: {tcga_expr.columns[:5].tolist()}")
    print(f"Gene types: {[type(g).__name__ for g in tcga_expr.columns[:3]]}")
else:
    print("TCGA data not found!")
    tcga_expr = None

# 4. Check overlaps
if gdsc_expr is not None and tcga_expr is not None:
    print("\n" + "=" * 80)
    print("4. Gene Overlaps")
    print("=" * 80)

    gdsc_genes = set(gdsc_expr.columns)
    tcga_genes = set(tcga_expr.columns)
    cancer_gene_names = set(cancer_genes['Hugo'])

    print(f"GDSC genes: {len(gdsc_genes)}")
    print(f"TCGA genes: {len(tcga_genes)}")
    print(f"Cancer genes (Hugo): {len(cancer_gene_names)}")

    gdsc_tcga = gdsc_genes.intersection(tcga_genes)
    print(f"\nGDSC ∩ TCGA: {len(gdsc_tcga)}")
    if len(gdsc_tcga) > 0:
        print(f"Sample: {list(gdsc_tcga)[:10]}")

    gdsc_cancer = gdsc_genes.intersection(cancer_gene_names)
    tcga_cancer = tcga_genes.intersection(cancer_gene_names)
    all_three = gdsc_genes.intersection(tcga_genes).intersection(cancer_gene_names)

    print(f"\nGDSC ∩ Cancer: {len(gdsc_cancer)}")
    print(f"TCGA ∩ Cancer: {len(tcga_cancer)}")
    print(f"GDSC ∩ TCGA ∩ Cancer: {len(all_three)}")
    if len(all_three) > 0:
        print(f"Sample: {list(all_three)[:10]}")
