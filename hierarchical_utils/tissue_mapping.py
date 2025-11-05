"""
Tissue mapping and hierarchical organization utilities for Hierarchical THERAPI.
Handles tissue type standardization and creation of tissue-cell line matrices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings


class TissueMapper:
    """
    Map cell lines to tissue types and create hierarchical structure.
    Standardizes tissue categories across different data sources.
    """

    # Standardized tissue categories (consolidate similar tissues)
    TISSUE_GROUPS = {
        'breast': ['breast', 'breast cancer', 'mammary', 'breast carcinoma'],
        'lung': ['lung', 'lung cancer', 'nsclc', 'sclc', 'lung_nsclc', 'lung_sclc',
                'non-small cell lung cancer', 'small cell lung cancer'],
        'colon': ['colon', 'colorectal', 'large intestine', 'colon cancer',
                 'colorectal cancer', 'large_intestine'],
        'blood': ['leukemia', 'lymphoma', 'myeloma', 'blood', 'haematopoietic',
                 'haematopoietic_lymphoid', 'aml', 'all', 'cml', 'cll',
                 'acute myeloid leukemia', 'acute lymphoblastic leukemia'],
        'brain': ['glioma', 'glioblastoma', 'cns', 'brain', 'central nervous system',
                 'neuroblastoma'],
        'skin': ['melanoma', 'skin', 'skin cancer'],
        'pancreas': ['pancreas', 'pancreatic', 'pancreatic cancer', 'pdac'],
        'ovary': ['ovary', 'ovarian', 'ovarian cancer'],
        'kidney': ['kidney', 'renal', 'kidney cancer', 'renal cancer'],
        'liver': ['liver', 'hepatocellular', 'liver cancer', 'hepatocellular carcinoma'],
        'stomach': ['stomach', 'gastric', 'stomach cancer', 'gastric cancer'],
        'esophagus': ['esophagus', 'esophageal', 'oesophagus', 'oesophageal'],
        'prostate': ['prostate', 'prostate cancer'],
        'bladder': ['bladder', 'urinary bladder', 'bladder cancer'],
        'thyroid': ['thyroid', 'thyroid cancer'],
        'bone': ['bone', 'osteosarcoma', 'ewing sarcoma', 'sarcoma'],
        'soft_tissue': ['soft tissue', 'soft_tissue', 'fibrosarcoma'],
        'cervix': ['cervix', 'cervical', 'cervical cancer'],
        'uterus': ['uterus', 'endometrial', 'uterine', 'endometrial cancer'],
        'head_neck': ['head and neck', 'head_and_neck', 'oral cavity', 'pharynx',
                     'larynx', 'upper aerodigestive'],
        'biliary': ['biliary', 'biliary tract', 'cholangiocarcinoma'],
        'testis': ['testis', 'testicular', 'germ cell'],
        'adrenal': ['adrenal', 'adrenal gland', 'adrenocortical'],
    }

    def __init__(self):
        """Initialize tissue mapper with standardized groups."""
        self.tissue_to_group = self._create_tissue_to_group_mapping()
        self.tissue_groups_list = list(self.TISSUE_GROUPS.keys()) + ['other']
        self.n_tissue_groups = len(self.tissue_groups_list)
        self.tissue_to_idx = {tissue: i for i, tissue in enumerate(self.tissue_groups_list)}

    def _create_tissue_to_group_mapping(self) -> Dict[str, str]:
        """
        Create mapping from specific tissue names to standardized groups.

        Returns:
            Dictionary mapping tissue names (lowercase) to group names
        """
        mapping = {}
        for group, tissues in self.TISSUE_GROUPS.items():
            for tissue in tissues:
                mapping[tissue.lower()] = group
        return mapping

    def get_tissue_group(self, tissue_name: str) -> str:
        """
        Map specific tissue name to standardized group.

        Args:
            tissue_name: Original tissue name from data

        Returns:
            Standardized tissue group name
        """
        if pd.isna(tissue_name):
            return 'other'
        return self.tissue_to_group.get(tissue_name.lower().strip(), 'other')

    def get_tissue_idx(self, tissue_name: str) -> int:
        """
        Get tissue group index for a tissue name.

        Args:
            tissue_name: Original tissue name

        Returns:
            Integer index of tissue group
        """
        tissue_group = self.get_tissue_group(tissue_name)
        return self.tissue_to_idx[tissue_group]

    def create_cell_line_tissue_matrix(self, cell_line_tissues: Dict[str, str]) -> Tuple[np.ndarray, Dict]:
        """
        Create binary matrix indicating which cell lines belong to which tissues.

        Args:
            cell_line_tissues: Dictionary mapping cell line IDs to tissue types

        Returns:
            Tuple of:
            - matrix: Binary matrix [n_tissues x n_cell_lines]
            - tissue_to_idx: Dictionary mapping tissue groups to indices
        """
        n_cell_lines = len(cell_line_tissues)
        matrix = np.zeros((self.n_tissue_groups, n_cell_lines), dtype=np.float32)

        # Create list to maintain order
        cell_line_list = list(cell_line_tissues.keys())

        for i, cell_line in enumerate(cell_line_list):
            tissue = cell_line_tissues[cell_line]
            tissue_group = self.get_tissue_group(tissue)
            tissue_idx = self.tissue_to_idx[tissue_group]
            matrix[tissue_idx, i] = 1.0

        return matrix, self.tissue_to_idx

    def get_tissue_statistics(self, cell_line_tissues: Dict[str, str]) -> pd.DataFrame:
        """
        Get statistics about tissue distribution.

        Args:
            cell_line_tissues: Dictionary mapping cell line IDs to tissue types

        Returns:
            DataFrame with tissue counts and percentages
        """
        tissue_groups = [self.get_tissue_group(t) for t in cell_line_tissues.values()]
        counts = pd.Series(tissue_groups).value_counts()

        stats = pd.DataFrame({
            'count': counts,
            'percentage': (counts / len(tissue_groups) * 100).round(2)
        })

        return stats.sort_values('count', ascending=False)

    def create_tissue_label_array(self, sample_tissues: List[str]) -> np.ndarray:
        """
        Create array of tissue labels for samples.

        Args:
            sample_tissues: List of tissue names for each sample

        Returns:
            Array of tissue indices
        """
        return np.array([self.get_tissue_idx(t) for t in sample_tissues])

    def get_tissue_name_from_idx(self, idx: int) -> str:
        """
        Get tissue group name from index.

        Args:
            idx: Tissue index

        Returns:
            Tissue group name
        """
        for tissue, tissue_idx in self.tissue_to_idx.items():
            if tissue_idx == idx:
                return tissue
        return 'other'

    def filter_cell_lines_by_tissue(self, cell_line_tissues: Dict[str, str],
                                   tissue_group: str) -> List[str]:
        """
        Get list of cell lines belonging to a specific tissue group.

        Args:
            cell_line_tissues: Dictionary mapping cell line IDs to tissue types
            tissue_group: Target tissue group

        Returns:
            List of cell line IDs in the tissue group
        """
        return [cl for cl, tissue in cell_line_tissues.items()
                if self.get_tissue_group(tissue) == tissue_group]

    def create_tissue_masks(self, cell_line_tissues: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Create boolean masks for each tissue type.

        Args:
            cell_line_tissues: Dictionary mapping cell line IDs to tissue types

        Returns:
            Dictionary mapping tissue groups to boolean masks
        """
        cell_line_list = list(cell_line_tissues.keys())
        n_cell_lines = len(cell_line_list)
        masks = {}

        for tissue_group in self.tissue_groups_list:
            mask = np.zeros(n_cell_lines, dtype=bool)
            for i, cell_line in enumerate(cell_line_list):
                if self.get_tissue_group(cell_line_tissues[cell_line]) == tissue_group:
                    mask[i] = True
            masks[tissue_group] = mask

        return masks

    def validate_tissue_coverage(self, cell_line_tissues: Dict[str, str],
                                min_samples_per_tissue: int = 5) -> Dict[str, bool]:
        """
        Check if each tissue has sufficient cell lines for training.

        Args:
            cell_line_tissues: Dictionary mapping cell line IDs to tissue types
            min_samples_per_tissue: Minimum required samples per tissue

        Returns:
            Dictionary indicating which tissues have sufficient coverage
        """
        stats = self.get_tissue_statistics(cell_line_tissues)
        coverage = {}

        for tissue in self.tissue_groups_list:
            if tissue in stats.index:
                coverage[tissue] = stats.loc[tissue, 'count'] >= min_samples_per_tissue
            else:
                coverage[tissue] = False

        # Warn about tissues with insufficient coverage
        insufficient = [t for t, covered in coverage.items() if not covered and t != 'other']
        if insufficient:
            warnings.warn(f"Tissues with <{min_samples_per_tissue} samples: {insufficient}")

        return coverage


class HierarchicalTissueEncoder:
    """
    Encode tissue hierarchies for use in neural networks.
    Supports one-hot encoding and learned embeddings.
    """

    def __init__(self, tissue_mapper: TissueMapper):
        self.tissue_mapper = tissue_mapper
        self.n_tissues = tissue_mapper.n_tissue_groups

    def one_hot_encode(self, tissue_indices: np.ndarray) -> np.ndarray:
        """
        Create one-hot encoding for tissue indices.

        Args:
            tissue_indices: Array of tissue indices

        Returns:
            One-hot encoded array [n_samples x n_tissues]
        """
        n_samples = len(tissue_indices)
        encoding = np.zeros((n_samples, self.n_tissues))
        encoding[np.arange(n_samples), tissue_indices] = 1
        return encoding

    def create_tissue_similarity_matrix(self) -> np.ndarray:
        """
        Create tissue similarity matrix based on biological relationships.
        This can be used to initialize tissue embeddings or as a prior.

        Returns:
            Similarity matrix [n_tissues x n_tissues]
        """
        # Initialize with identity (each tissue similar to itself)
        similarity = np.eye(self.n_tissues)

        # Define biological relationships (can be extended)
        related_groups = [
            ['lung', 'breast', 'colon'],  # Carcinomas
            ['blood', 'bone'],  # Hematological
            ['brain', 'neuroblastoma'],  # Neurological
            ['kidney', 'bladder'],  # Urological
            ['stomach', 'esophagus', 'colon'],  # GI tract
            ['ovary', 'cervix', 'uterus'],  # Gynecological
        ]

        # Add similarities for related groups
        for group in related_groups:
            for tissue1 in group:
                for tissue2 in group:
                    if tissue1 != tissue2 and tissue1 in self.tissue_mapper.tissue_to_idx and tissue2 in self.tissue_mapper.tissue_to_idx:
                        idx1 = self.tissue_mapper.tissue_to_idx[tissue1]
                        idx2 = self.tissue_mapper.tissue_to_idx[tissue2]
                        similarity[idx1, idx2] = 0.3  # Moderate similarity
                        similarity[idx2, idx1] = 0.3

        return similarity
