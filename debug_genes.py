"""
Debug script to check gene harmonization issue
"""
import os
import sys
import pandas as pd
import pickle
import gzip

# Add src to path to use original utils
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Add hierarchical_utils to path
sys.path.insert(0, os.path.dirname(__file__))

from hierarchical_utils.data_loader import TransactDataLoader, load_pickle_file


def check_gene_structures():
    """Check the structure of all gene-related data"""

    data_root = 'data/'
    data_loader = TransactDataLoader(data_root=data_root)

    print("=" * 80)
    print("CHECKING GENE DATA STRUCTURES")
    print("=" * 80)

    # 1. Check cancer genes
    print("\n1. Cancer Genes File:")
    if data_loader.cancer_genes is not None:
        print(f"   Columns: {data_loader.cancer_genes.columns.tolist()}")
        print(f"   Shape: {data_loader.cancer_genes.shape}")
        print(f"   First 5 genes:")
        print(data_loader.cancer_genes.head())
    else:
        print("   Not found!")

    # 2. Check GDSC expression
    print("\n2. GDSC Expression Data:")
    try:
        gdsc_data = data_loader.load_gdsc_data()
        gdsc_expr = gdsc_data['expression']
        print(f"   Type: {type(gdsc_expr)}")
        print(f"   Shape: {gdsc_expr.shape}")
        print(f"   First 5 genes (columns): {gdsc_expr.columns[:5].tolist()}")
        print(f"   First 3 samples (rows): {gdsc_expr.index[:3].tolist()}")

        # Check if genes are numbers or strings
        sample_genes = gdsc_expr.columns[:10].tolist()
        print(f"   Sample gene types: {[type(g).__name__ for g in sample_genes[:3]]}")
        print(f"   Sample genes: {sample_genes}")

    except Exception as e:
        print(f"   Error loading GDSC: {e}")

    # 3. Check TCGA expression
    print("\n3. TCGA Expression Data:")
    try:
        tcga_data = data_loader.load_tcga_data()
        tcga_expr = tcga_data['expression']
        print(f"   Type: {type(tcga_expr)}")
        print(f"   Shape: {tcga_expr.shape}")
        print(f"   First 5 genes (columns): {tcga_expr.columns[:5].tolist()}")
        print(f"   First 3 samples (rows): {tcga_expr.index[:3].tolist()}")

        # Check if genes are numbers or strings
        sample_genes = tcga_expr.columns[:10].tolist()
        print(f"   Sample gene types: {[type(g).__name__ for g in sample_genes[:3]]}")
        print(f"   Sample genes: {sample_genes}")

    except Exception as e:
        print(f"   Error loading TCGA: {e}")

    # 4. Check harmonization
    print("\n4. Gene Harmonization Test:")
    try:
        gdsc_data = data_loader.load_gdsc_data()
        tcga_data = data_loader.load_tcga_data()

        gdsc_expr = gdsc_data['expression']
        tcga_expr = tcga_data['expression']

        # Get gene sets
        gdsc_genes = set(gdsc_expr.columns)
        tcga_genes = set(tcga_expr.columns)

        if data_loader.cancer_genes is not None:
            # Try different column names
            for col in ['Hugo', 'gene_symbol', 'gene', 'symbol']:
                if col in data_loader.cancer_genes.columns:
                    cancer_genes = set(data_loader.cancer_genes[col])
                    print(f"\n   Cancer gene column '{col}': {len(cancer_genes)} genes")
                    print(f"   Sample: {list(cancer_genes)[:5]}")

                    # Check intersections
                    gdsc_cancer_overlap = gdsc_genes.intersection(cancer_genes)
                    tcga_cancer_overlap = tcga_genes.intersection(cancer_genes)
                    all_overlap = gdsc_genes.intersection(tcga_genes).intersection(cancer_genes)

                    print(f"   GDSC ∩ Cancer genes: {len(gdsc_cancer_overlap)}")
                    print(f"   TCGA ∩ Cancer genes: {len(tcga_cancer_overlap)}")
                    print(f"   GDSC ∩ TCGA ∩ Cancer genes: {len(all_overlap)}")

                    if len(all_overlap) > 0:
                        print(f"   Sample common genes: {list(all_overlap)[:10]}")
                    break

        # Direct intersection without cancer genes
        gdsc_tcga_overlap = gdsc_genes.intersection(tcga_genes)
        print(f"\n   Direct GDSC ∩ TCGA: {len(gdsc_tcga_overlap)} genes")
        if len(gdsc_tcga_overlap) > 0:
            print(f"   Sample: {list(gdsc_tcga_overlap)[:10]}")

        # Call actual harmonize_genes
        common_genes = data_loader.harmonize_genes(gdsc_expr, tcga_expr)
        print(f"\n   harmonize_genes() result: {len(common_genes)} genes")
        if len(common_genes) > 0:
            print(f"   Sample: {common_genes[:10]}")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == '__main__':
    check_gene_structures()
