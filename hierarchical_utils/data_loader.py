"""
Data loader for TRANSACT's data format (GDSC + PDX + TCGA + HMF)
This module handles loading and harmonizing gene expression and drug response data
across different sources for Hierarchical THERAPI.
"""

import os
import pandas as pd
import pickle
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings


def load_pickle_file(filepath: str):
    """
    Load pickle file, handling both compressed and uncompressed formats.

    Args:
        filepath: Path to pickle file

    Returns:
        Unpickled data
    """
    # Try gzip-compressed first (check magic number)
    with open(filepath, 'rb') as f:
        magic = f.read(2)
        f.seek(0)

        if magic == b'\x1f\x8b':  # Gzip magic number
            with gzip.open(filepath, 'rb') as gz:
                return pickle.load(gz)
        else:
            return pickle.load(f)


class TransactDataLoader:
    """
    Load and harmonize data from TRANSACT's data structure.

    Data structure:
    - GDSC: Cell line expression and drug responses
    - PDXE: Patient-Derived Xenografts (intermediate validation)
    - TCGA: Primary tumor patient data
    - HMF: Metastatic patient data
    """

    def __init__(self, data_root='data/'):
        self.data_root = data_root

        # Load gene mappings
        cancer_genes_path = os.path.join(data_root, 'mini_cancer_genes.csv')
        gene_lookup_path = os.path.join(data_root, 'mini_cancer_lookup_genes.csv')

        if os.path.exists(cancer_genes_path):
            self.cancer_genes = pd.read_csv(cancer_genes_path)
        else:
            self.cancer_genes = None
            warnings.warn(f"Cancer genes file not found at {cancer_genes_path}")

        if os.path.exists(gene_lookup_path):
            self.gene_mapping = pd.read_csv(gene_lookup_path)
        else:
            self.gene_mapping = None
            warnings.warn(f"Gene lookup file not found at {gene_lookup_path}")

    def load_gdsc_data(self) -> Dict:
        """
        Load GDSC cell line data with tissue labels.

        Returns:
            Dictionary containing:
            - expression: Gene expression DataFrame (samples x genes)
            - cell_info: Cell line metadata with tissue types
            - drug_response: Drug response data (GDSC1 + GDSC2)
            - tissue_mapping: Dict mapping cell line names to tissue types
        """
        gdsc_dir = os.path.join(self.data_root, 'GDSC')

        # Load expression data
        expr_path = os.path.join(gdsc_dir, 'rnaseq/GDSC_rnaseq_data.pkl')
        if os.path.exists(expr_path):
            gdsc_expr = load_pickle_file(expr_path)
        else:
            # Fallback to CSV if pickle not available
            expr_csv_path = os.path.join(gdsc_dir, 'GDSC_gex.csv')
            if os.path.exists(expr_csv_path):
                gdsc_expr = pd.read_csv(expr_csv_path, index_col=0)
            else:
                raise FileNotFoundError(f"GDSC expression data not found at {expr_path} or {expr_csv_path}")

        # Load cell line metadata with tissue types
        cell_info_path = os.path.join(gdsc_dir, 'model_list_20191104.csv')
        if not os.path.exists(cell_info_path):
            # Fallback to GDSC_info.csv from original THERAPI format
            cell_info_path = os.path.join(gdsc_dir, 'GDSC_info.csv')

        if os.path.exists(cell_info_path):
            cell_info = pd.read_csv(cell_info_path)
        else:
            raise FileNotFoundError(f"GDSC cell info not found")

        # Load drug responses
        drug_response = self._load_gdsc_responses(gdsc_dir)

        # Extract tissue mapping
        if 'tissue' in cell_info.columns:
            tissue_col = 'tissue'
        elif 'tissue_label' in cell_info.columns:
            tissue_col = 'tissue_label'
        else:
            raise ValueError("No tissue column found in cell_info")

        # Handle model_name vs other identifiers
        if 'model_name' in cell_info.columns:
            id_col = 'model_name'
        elif 'CELL_LINE_NAME' in cell_info.columns:
            id_col = 'CELL_LINE_NAME'
        else:
            id_col = cell_info.columns[0]

        tissue_mapping = dict(zip(cell_info[id_col], cell_info[tissue_col]))

        return {
            'expression': gdsc_expr,
            'cell_info': cell_info,
            'drug_response': drug_response,
            'tissue_mapping': tissue_mapping
        }

    def _load_gdsc_responses(self, gdsc_dir: str) -> pd.DataFrame:
        """Load GDSC drug response data from both GDSC1 and GDSC2."""
        gdsc1_path = os.path.join(gdsc_dir, 'response/GDSC1_fitted_dose_response_27Oct23.xlsx')
        gdsc2_path = os.path.join(gdsc_dir, 'response/GDSC2_fitted_dose_response_27Oct23.xlsx')

        responses = []

        # Try loading from TRANSACT format
        if os.path.exists(gdsc1_path):
            gdsc1_resp = pd.read_excel(gdsc1_path)
            responses.append(gdsc1_resp)

        if os.path.exists(gdsc2_path):
            gdsc2_resp = pd.read_excel(gdsc2_path)
            responses.append(gdsc2_resp)

        # Fallback to original THERAPI format
        if not responses:
            csv_path = os.path.join(gdsc_dir, 'GDSC_Drug_SMILES_Response.csv')
            if os.path.exists(csv_path):
                responses.append(pd.read_csv(csv_path))
            else:
                warnings.warn("No GDSC drug response files found")
                return pd.DataFrame()

        return pd.concat(responses, ignore_index=True) if responses else pd.DataFrame()

    def load_tcga_data(self) -> Dict:
        """
        Load TCGA patient data with tissue labels.

        Returns:
            Dictionary containing:
            - expression: Gene expression DataFrame
            - sample_annot: Sample annotations
            - drug_response: Clinical drug responses
            - tissue_info: Tissue type information
        """
        tcga_dir = os.path.join(self.data_root, 'TCGA')

        # Load expression
        expr_path = os.path.join(tcga_dir, 'rnaseq/TCGA_rnaseq_data.pkl')
        if os.path.exists(expr_path):
            tcga_expr = load_pickle_file(expr_path)
        else:
            # Fallback to CSV
            expr_csv_path = os.path.join(tcga_dir, 'TCGA_unlabeled_gex.csv')
            if os.path.exists(expr_csv_path):
                tcga_expr = pd.read_csv(expr_csv_path, index_col=0)
            else:
                raise FileNotFoundError(f"TCGA expression data not found")

        # Load sample annotations
        annot_path = os.path.join(tcga_dir, 'rnaseq/TCGA_rnaseq_sample_annot.pkl')
        if os.path.exists(annot_path):
            sample_annot = load_pickle_file(annot_path)
        else:
            sample_annot = None
            warnings.warn("TCGA sample annotations not found")

        # Load clinical responses
        resp_path = os.path.join(tcga_dir, 'response/response.csv')
        if os.path.exists(resp_path):
            tcga_resp = pd.read_csv(resp_path)
        else:
            # Fallback to THERAPI format
            resp_path = os.path.join(tcga_dir, 'TCGA_Drug_SMILES_Response.csv')
            if os.path.exists(resp_path):
                tcga_resp = pd.read_csv(resp_path)
            else:
                warnings.warn("TCGA drug response not found")
                tcga_resp = pd.DataFrame()

        # Load tissue types
        tissue_path = os.path.join(tcga_dir, 'pancancer_sample_spec.csv')
        if os.path.exists(tissue_path):
            tissue_info = pd.read_csv(tissue_path, low_memory=False)
        else:
            # Fallback to info file
            tissue_path = os.path.join(tcga_dir, 'TCGA_unlabeled_info.csv')
            if os.path.exists(tissue_path):
                tissue_info = pd.read_csv(tissue_path, low_memory=False)
            else:
                warnings.warn("TCGA tissue info not found")
                tissue_info = pd.DataFrame()

        return {
            'expression': tcga_expr,
            'sample_annot': sample_annot,
            'drug_response': tcga_resp,
            'tissue_info': tissue_info
        }

    def load_pdx_data(self) -> Dict:
        """
        Load PDX data as intermediate validation (no patient labels used!).

        Returns:
            Dictionary containing:
            - expression: Gene expression DataFrame
            - biospecimen: PDX to tissue mapping
            - drug_response: Drug response data
        """
        pdx_dir = os.path.join(self.data_root, 'PDXE')

        if not os.path.exists(pdx_dir):
            warnings.warn(f"PDX directory not found at {pdx_dir}")
            return {'expression': None, 'biospecimen': None, 'drug_response': None}

        # Load expression
        expr_path = os.path.join(pdx_dir, 'fpkm/PDXE_fpkm_data.pkl')
        if os.path.exists(expr_path):
            pdx_expr = load_pickle_file(expr_path)
        else:
            warnings.warn(f"PDX expression data not found at {expr_path}")
            pdx_expr = None

        # Load biospecimen (tissue mapping)
        bio_path = os.path.join(pdx_dir, 'pancancer_biospecimen.csv')
        if os.path.exists(bio_path):
            biospecimen = pd.read_csv(bio_path)
        else:
            warnings.warn(f"PDX biospecimen not found at {bio_path}")
            biospecimen = None

        # Load drug responses
        resp_path = os.path.join(pdx_dir, 'response/response.csv')
        if os.path.exists(resp_path):
            pdx_resp = pd.read_csv(resp_path)
        else:
            warnings.warn(f"PDX response data not found at {resp_path}")
            pdx_resp = None

        return {
            'expression': pdx_expr,
            'biospecimen': biospecimen,
            'drug_response': pdx_resp
        }

    def load_hmf_data(self) -> Dict:
        """
        Load HMF (Hartwig Medical Foundation) metastatic patient data.

        Returns:
            Dictionary containing metastatic patient expression and response data
        """
        hmf_dir = os.path.join(self.data_root, 'HMF')

        if not os.path.exists(hmf_dir):
            warnings.warn(f"HMF directory not found at {hmf_dir}. This is expected if data not yet available.")
            return {'expression': None, 'drug_response': None, 'metadata': None}

        # HMF data structure to be populated when available
        return {
            'expression': None,
            'drug_response': None,
            'metadata': None
        }

    def harmonize_genes(self, *datasets) -> List[str]:
        """
        Ensure all datasets use the same gene set.
        Uses cancer gene panel for focused analysis.

        Args:
            *datasets: Variable number of gene expression DataFrames

        Returns:
            List of common gene names/IDs
        """
        # Try multiple common column names for cancer genes
        if self.cancer_genes is not None:
            gene_col = None
            for col_name in ['Hugo', 'gene_symbol', 'gene', 'symbol', 'Gene', 'SYMBOL']:
                if col_name in self.cancer_genes.columns:
                    gene_col = col_name
                    break

            if gene_col:
                common_genes = set(self.cancer_genes[gene_col])
            else:
                # Fallback to first column if no recognized column name
                common_genes = set(self.cancer_genes.iloc[:, 0])
        else:
            # If no cancer genes provided, use intersection of all datasets
            common_genes = None

        for dataset in datasets:
            if dataset is None:
                continue

            if isinstance(dataset, pd.DataFrame):
                available_genes = set(dataset.columns)
            elif isinstance(dataset, dict) and 'expression' in dataset:
                if isinstance(dataset['expression'], pd.DataFrame):
                    available_genes = set(dataset['expression'].columns)
                else:
                    continue
            else:
                continue

            if common_genes is None:
                common_genes = available_genes
            else:
                common_genes = common_genes.intersection(available_genes)

        return sorted(list(common_genes)) if common_genes else []

    def normalize_expression(self, expr_df: pd.DataFrame, method='standard') -> pd.DataFrame:
        """
        Normalize gene expression data.

        Args:
            expr_df: Expression DataFrame (samples x genes)
            method: Normalization method ('standard', 'minmax', 'robust')

        Returns:
            Normalized expression DataFrame
        """
        if method == 'standard':
            scaler = StandardScaler()
            normalized = scaler.fit_transform(expr_df)
            return pd.DataFrame(normalized, index=expr_df.index, columns=expr_df.columns)
        else:
            raise NotImplementedError(f"Normalization method {method} not implemented")

    def create_aligned_datasets(self, gdsc_data: Dict, target_data: Dict,
                               common_genes: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create aligned gene expression datasets for source and target domains.

        Args:
            gdsc_data: GDSC data dictionary
            target_data: Target domain data dictionary (TCGA, PDX, etc.)
            common_genes: List of genes to include

        Returns:
            Tuple of (source_expr, target_expr) DataFrames with aligned genes
        """
        # Extract expression data
        source_expr = gdsc_data['expression']
        target_expr = target_data['expression']

        # Select common genes
        source_expr_aligned = source_expr[common_genes]
        target_expr_aligned = target_expr[common_genes]

        return source_expr_aligned, target_expr_aligned


def create_perturbation_features(expression_data: pd.DataFrame,
                                 drug_response: pd.DataFrame) -> np.ndarray:
    """
    Create drug-induced perturbation features.
    This is a placeholder - actual implementation would compute
    differential expression signatures for each drug.

    Args:
        expression_data: Gene expression DataFrame
        drug_response: Drug response DataFrame

    Returns:
        Perturbation feature array
    """
    # This would need to be implemented based on THERAPI's perturbation calculation
    # For now, return placeholder
    warnings.warn("Perturbation feature calculation not fully implemented")
    return np.zeros((len(drug_response), expression_data.shape[1]))
