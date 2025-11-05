"""
Ablation models for Hierarchical THERAPI.

These variants help demonstrate the contribution of each component:
1. No hierarchy - Flat attention (original THERAPI-style)
2. Uniform tissue weights - Fixed weights instead of learned routing
3. Hard routing - Hard assignments instead of soft weighting
4. No multi-head - Single-head attention
5. No temperature learning - Fixed temperature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from models.hierarchical_therapi import (
    HierarchicalTHERAPI,
    TissueRouter,
    HierarchicalAttention
)


class FlatTHERAPI(nn.Module):
    """
    Flat attention baseline (similar to original THERAPI).
    No tissue hierarchy - attends to all cell lines equally.
    """

    def __init__(self, n_genes: int, n_latent: int = 128):
        super().__init__()

        self.n_genes = n_genes
        self.n_latent = n_latent

        # Patient encoder
        self.patient_encoder = nn.Sequential(
            nn.Linear(n_genes, n_latent),
            nn.LayerNorm(n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )

        # Cell line encoder
        self.cell_line_encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_latent)
        )

        # Simple attention (no hierarchy)
        self.key_proj = nn.Linear(n_latent, n_latent, bias=False)

    def forward(self, patient_expr: torch.Tensor,
                cell_line_exprs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Flat attention forward pass.

        Args:
            patient_expr: Patient gene expression [batch_size, n_genes]
            cell_line_exprs: Cell line expressions [n_cell_lines, n_genes]

        Returns:
            Dictionary with representation (no tissue weights)
        """
        # Encode
        patient_enc = self.patient_encoder(patient_expr)
        cell_line_encs = self.cell_line_encoder(cell_line_exprs)

        # Compute attention over ALL cell lines
        keys = self.key_proj(cell_line_encs)
        scores = torch.matmul(patient_enc, keys.t()) / np.sqrt(self.n_latent)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        representation = torch.matmul(attn_weights, cell_line_encs)

        return {
            'representation': representation,
            'tissue_weights': None,  # No tissue routing
            'attention_weights': attn_weights
        }


class UniformTissueWeightTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model with uniform (fixed) tissue weights.
    Tests whether learned routing is necessary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override tissue router to return uniform weights
        self._uniform_weights = True

    def compute_tissue_routing(self, patient_enc: torch.Tensor,
                              hard: bool = False) -> torch.Tensor:
        """
        Return uniform weights instead of learned routing.
        """
        batch_size = patient_enc.size(0)
        uniform_weights = torch.ones(batch_size, self.n_tissues, device=patient_enc.device)
        uniform_weights = uniform_weights / self.n_tissues
        return uniform_weights


class HardRoutingTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model with hard tissue assignments.
    Routes each patient to exactly ONE tissue (one-hot).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hard_routing = True

    def compute_tissue_routing(self, patient_enc: torch.Tensor,
                              hard: bool = True) -> torch.Tensor:
        """
        Force hard routing (one-hot assignments).
        """
        # Get soft weights
        logits = self.tissue_router.router(patient_enc)

        # Convert to hard one-hot
        max_idx = logits.argmax(dim=-1, keepdim=True)
        hard_weights = torch.zeros_like(logits)
        hard_weights.scatter_(1, max_idx, 1.0)

        return hard_weights


class SingleHeadTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model with single-head attention.
    Tests contribution of multi-head mechanism.
    """

    def __init__(self, *args, **kwargs):
        # Force single-head
        kwargs['use_multi_head'] = False
        super().__init__(*args, **kwargs)

        # Override attention with simple version
        self.attention = None  # Will use simple attention fallback


class FixedTemperatureTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model with fixed (non-learnable) temperature.
    Tests whether temperature learning is beneficial.
    """

    def __init__(self, *args, fixed_temp: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace learnable temperature with fixed value
        self.tissue_router.temperature = nn.Parameter(
            torch.tensor(fixed_temp),
            requires_grad=False  # Fixed, not learnable
        )


class NoEntropyRegTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model trained without entropy regularization.
    Tests whether entropy regularization improves routing decisiveness.
    """
    # This is more of a training configuration than architecture change
    # Included for completeness
    pass


class RandomRoutingTHERAPI(HierarchicalTHERAPI):
    """
    Hierarchical model with random tissue routing.
    Worst-case baseline to show learned routing is better than random.
    """

    def compute_tissue_routing(self, patient_enc: torch.Tensor,
                              hard: bool = False) -> torch.Tensor:
        """
        Return random tissue weights.
        """
        batch_size = patient_enc.size(0)
        # Random weights from Dirichlet distribution
        random_weights = torch.rand(batch_size, self.n_tissues, device=patient_enc.device)
        random_weights = random_weights / random_weights.sum(dim=-1, keepdim=True)
        return random_weights


class TrueOracleRoutingTHERAPI(HierarchicalTHERAPI):
    """
    Oracle model that uses true tissue labels for routing.
    Upper bound on what perfect routing could achieve.
    Only for analysis, not fair comparison.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oracle_mode = True

    def forward_with_oracle(self, patient_expr: torch.Tensor,
                          cell_line_exprs: torch.Tensor,
                          true_tissue_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass using oracle tissue labels.

        Args:
            patient_expr: Patient gene expression
            cell_line_exprs: Cell line expressions
            true_tissue_labels: True tissue indices [batch_size]

        Returns:
            Dictionary with results using oracle routing
        """
        batch_size = patient_expr.size(0)

        # Create one-hot tissue weights from true labels
        oracle_weights = torch.zeros(batch_size, self.n_tissues, device=patient_expr.device)
        oracle_weights.scatter_(1, true_tissue_labels.unsqueeze(1), 1.0)

        # Get patient encoding
        patient_enc = self.patient_encoder(patient_expr)
        cell_line_encs = self.cell_line_encoder(cell_line_exprs)

        # Process with oracle weights (simplified version)
        weighted_representations = []

        for tissue_idx in range(self.n_tissues):
            if self.tissue_cell_mask is not None:
                tissue_mask = self.tissue_cell_mask[tissue_idx].bool()
                if not tissue_mask.any():
                    continue
                tissue_cell_encs = cell_line_encs[tissue_mask]
            else:
                tissue_cell_encs = cell_line_encs

            # Simple attention within tissue
            scores = torch.matmul(patient_enc, tissue_cell_encs.t())
            attn_weights = F.softmax(scores / np.sqrt(self.n_latent), dim=-1)
            tissue_repr = torch.matmul(attn_weights, tissue_cell_encs)

            # Weight by oracle tissue relevance
            tissue_weight = oracle_weights[:, tissue_idx:tissue_idx+1]
            weighted_repr = tissue_weight * tissue_repr
            weighted_representations.append(weighted_repr)

        final_representation = torch.stack(weighted_representations, dim=0).sum(dim=0)

        return {
            'representation': final_representation,
            'tissue_weights': oracle_weights,
            'oracle': True
        }


class SimplifiedHierarchicalTHERAPI(nn.Module):
    """
    Simplified hierarchical model with fewer parameters.
    Tests whether full complexity is needed.
    """

    def __init__(self, n_genes: int, n_tissues: int, n_latent: int = 64,
                 tissue_cell_mask: Optional[torch.Tensor] = None):
        super().__init__()

        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.n_latent = n_latent

        # Simpler encoders
        self.patient_encoder = nn.Sequential(
            nn.Linear(n_genes, n_latent),
            nn.ReLU()
        )

        self.cell_line_encoder = nn.Sequential(
            nn.Linear(n_genes, n_latent),
            nn.ReLU()
        )

        # Simpler router
        self.tissue_router = nn.Sequential(
            nn.Linear(n_latent, n_tissues),
            nn.Softmax(dim=-1)
        )

        if tissue_cell_mask is not None:
            self.register_buffer('tissue_cell_mask', tissue_cell_mask)
        else:
            self.tissue_cell_mask = None

    def forward(self, patient_expr: torch.Tensor,
                cell_line_exprs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simplified forward pass."""
        patient_enc = self.patient_encoder(patient_expr)
        cell_line_encs = self.cell_line_encoder(cell_line_exprs)
        tissue_weights = self.tissue_router(patient_enc)

        # Simple weighted attention per tissue
        weighted_representations = []

        for tissue_idx in range(self.n_tissues):
            if self.tissue_cell_mask is not None:
                tissue_mask = self.tissue_cell_mask[tissue_idx].bool()
                if not tissue_mask.any():
                    continue
                tissue_cell_encs = cell_line_encs[tissue_mask]
            else:
                tissue_cell_encs = cell_line_encs

            # Simple mean pooling (no attention)
            tissue_repr = tissue_cell_encs.mean(dim=0, keepdim=True).expand(patient_enc.size(0), -1)

            tissue_weight = tissue_weights[:, tissue_idx:tissue_idx+1]
            weighted_repr = tissue_weight * tissue_repr
            weighted_representations.append(weighted_repr)

        final_representation = torch.stack(weighted_representations, dim=0).sum(dim=0)

        return {
            'representation': final_representation,
            'tissue_weights': tissue_weights
        }


# Factory function for creating ablation models
def create_ablation_model(variant: str, **kwargs):
    """
    Factory function to create ablation model variants.

    Args:
        variant: Name of the ablation variant
        **kwargs: Model parameters

    Returns:
        Ablation model instance
    """
    variant = variant.lower()

    if variant == 'full_model' or variant == 'hierarchical':
        return HierarchicalTHERAPI(**kwargs)
    elif variant == 'flat' or variant == 'no_hierarchy':
        # Remove tissue-specific kwargs
        kwargs.pop('n_tissues', None)
        kwargs.pop('tissue_cell_mask', None)
        return FlatTHERAPI(
            n_genes=kwargs['n_genes'],
            n_latent=kwargs.get('n_latent', 128)
        )
    elif variant == 'uniform_weights':
        return UniformTissueWeightTHERAPI(**kwargs)
    elif variant == 'hard_routing':
        return HardRoutingTHERAPI(**kwargs)
    elif variant == 'single_head' or variant == 'no_multi_head':
        return SingleHeadTHERAPI(**kwargs)
    elif variant == 'fixed_temperature':
        fixed_temp = kwargs.pop('fixed_temp', 1.0)
        return FixedTemperatureTHERAPI(**kwargs, fixed_temp=fixed_temp)
    elif variant == 'random_routing':
        return RandomRoutingTHERAPI(**kwargs)
    elif variant == 'oracle':
        return TrueOracleRoutingTHERAPI(**kwargs)
    elif variant == 'simplified':
        return SimplifiedHierarchicalTHERAPI(
            n_genes=kwargs['n_genes'],
            n_tissues=kwargs['n_tissues'],
            n_latent=kwargs.get('n_latent', 64),
            tissue_cell_mask=kwargs.get('tissue_cell_mask')
        )
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
