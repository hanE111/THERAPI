"""
Hierarchical THERAPI: A unified tissue-aware architecture for drug response prediction.

This model extends THERAPI with a hierarchical attention mechanism that:
1. Routes patient samples to relevant tissue-specific cell lines
2. Computes attention within each tissue separately
3. Combines tissue-specific representations with learned weights

Key innovation: Single unified model with learnable tissue routing instead of
multiple separate tissue-specific models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class TissueRouter(nn.Module):
    """
    Learnable tissue routing network.
    Determines which tissue types are most relevant for a given patient sample.
    """

    def __init__(self, input_dim: int, n_tissues: int, hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()

        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, n_tissues)
        )

        # Learnable temperature for Gumbel-Softmax
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, patient_encoding: torch.Tensor,
                strategy: str = 'gumbel',
                hard: bool = False) -> torch.Tensor:
        """
        Compute tissue relevance weights.

        Args:
            patient_encoding: Encoded patient features [batch_size, input_dim]
            strategy: Routing strategy ('softmax', 'gumbel', 'sparsemax')
            hard: Whether to use hard (one-hot) assignments in Gumbel-Softmax

        Returns:
            Tissue weights [batch_size, n_tissues]
        """
        logits = self.router(patient_encoding)

        if strategy == 'softmax':
            return F.softmax(logits, dim=-1)
        elif strategy == 'gumbel':
            # Gumbel-Softmax for differentiable discrete sampling
            return F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        elif strategy == 'sparsemax':
            # Sparse routing (would need sparsemax implementation)
            return self._sparsemax(logits)
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    def _sparsemax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sparsemax activation for sparse tissue selection.
        Simplified implementation - use dedicated library for production.
        """
        # Sort logits
        sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)

        # Find threshold
        cumsum = torch.cumsum(sorted_logits, dim=-1)
        k = torch.arange(1, logits.size(-1) + 1, device=logits.device).float()
        threshold = (cumsum - 1) / k

        # Find support size
        support = (sorted_logits > threshold).sum(dim=-1, keepdim=True)

        # Compute tau
        tau = threshold.gather(-1, support - 1)

        # Apply threshold
        output = torch.clamp(logits - tau, min=0)
        return output


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism.
    Computes attention within each tissue separately, then combines.
    """

    def __init__(self, latent_dim: int, n_heads: int = 4):
        super().__init__()

        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // n_heads

        assert self.head_dim * n_heads == latent_dim, "latent_dim must be divisible by n_heads"

        # Multi-head attention components
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        self.value_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention within a tissue.

        Args:
            query: Patient encoding [batch_size, latent_dim]
            keys: Cell line encodings [n_cell_lines, latent_dim]
            values: Cell line encodings [n_cell_lines, latent_dim]
            mask: Optional mask for valid cell lines [n_cell_lines]

        Returns:
            Tuple of (weighted_repr, attention_weights)
        """
        batch_size = query.size(0)
        n_cell_lines = keys.size(0)

        # Project and reshape for multi-head attention
        Q = self.query_proj(query).view(batch_size, self.n_heads, self.head_dim)
        K = self.key_proj(keys).view(n_cell_lines, self.n_heads, self.head_dim)
        V = self.value_proj(values).view(n_cell_lines, self.n_heads, self.head_dim)

        # Compute attention scores
        scores = torch.einsum('bhd,nhd->bhn', Q, K) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.view(1, 1, -1).expand(batch_size, self.n_heads, -1)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.einsum('bhn,nhd->bhd', attn_weights, V)
        output = output.reshape(batch_size, self.latent_dim)
        output = self.out_proj(output)

        # Average attention weights across heads for analysis
        avg_attn_weights = attn_weights.mean(dim=1)

        return output, avg_attn_weights


class HierarchicalTHERAPI(nn.Module):
    """
    Hierarchical THERAPI: Unified tissue-aware drug response prediction.

    Combines THERAPI's proven alignment and prediction with hierarchical
    tissue routing for improved performance and interpretability.
    """

    def __init__(self, n_genes: int, n_tissues: int, n_latent: int = 128,
                 tissue_cell_mask: Optional[torch.Tensor] = None,
                 routing_strategy: str = 'gumbel',
                 use_multi_head: bool = True):
        super().__init__()

        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.n_latent = n_latent
        self.routing_strategy = routing_strategy

        # Patient encoder (similar to THERAPI's TARGET_weightencoder Q network)
        self.patient_encoder = nn.Sequential(
            nn.Linear(n_genes, n_latent),
            nn.LayerNorm(n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent)
        )

        # Cell line encoder (similar to THERAPI's SOURCE_AE encoder)
        self.cell_line_encoder = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, n_latent)
        )

        # Tissue router (NEW: key contribution)
        self.tissue_router = TissueRouter(
            input_dim=n_latent,
            n_tissues=n_tissues,
            hidden_dim=512,
            dropout=0.3
        )

        # Hierarchical attention
        if use_multi_head:
            self.attention = HierarchicalAttention(n_latent, n_heads=4)
        else:
            self.attention = None

        # Store tissue-cell line mapping as buffer (not trained)
        if tissue_cell_mask is not None:
            self.register_buffer('tissue_cell_mask', tissue_cell_mask)
        else:
            self.tissue_cell_mask = None

    def compute_tissue_routing(self, patient_enc: torch.Tensor,
                              hard: bool = False) -> torch.Tensor:
        """
        Compute tissue routing weights for patient samples.

        Args:
            patient_enc: Encoded patient features [batch_size, n_latent]
            hard: Whether to use hard assignments

        Returns:
            Tissue weights [batch_size, n_tissues]
        """
        return self.tissue_router(patient_enc, strategy=self.routing_strategy, hard=hard)

    def hierarchical_attention_forward(self, patient_expr: torch.Tensor,
                                      cell_line_exprs: torch.Tensor,
                                      return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Core hierarchical attention mechanism.

        Args:
            patient_expr: Patient gene expression [batch_size, n_genes]
            cell_line_exprs: Cell line expressions [n_cell_lines, n_genes]
            return_attention: Whether to return detailed attention info

        Returns:
            Dictionary containing:
            - representation: Final patient representation [batch_size, n_latent]
            - tissue_weights: Tissue routing weights [batch_size, n_tissues]
            - attention_breakdown: (optional) Detailed attention per tissue
        """
        batch_size = patient_expr.size(0)

        # Encode patient and cell lines
        patient_enc = self.patient_encoder(patient_expr)  # [batch, n_latent]
        cell_line_encs = self.cell_line_encoder(cell_line_exprs)  # [n_cells, n_latent]

        # Compute tissue routing weights
        tissue_weights = self.compute_tissue_routing(patient_enc)  # [batch, n_tissues]

        # Initialize storage for weighted representations
        weighted_representations = []
        attention_breakdown = {} if return_attention else None

        # Process each tissue separately
        for tissue_idx in range(self.n_tissues):
            # Get mask for this tissue's cell lines
            if self.tissue_cell_mask is not None:
                tissue_mask = self.tissue_cell_mask[tissue_idx].bool()  # [n_cell_lines]

                # Skip if no cell lines for this tissue
                if not tissue_mask.any():
                    continue

                # Select cell lines for this tissue
                tissue_cell_encs = cell_line_encs[tissue_mask]  # [n_tissue_cells, n_latent]
            else:
                # If no mask, use all cell lines (fallback)
                tissue_cell_encs = cell_line_encs
                tissue_mask = None

            # Compute attention within tissue
            if self.attention is not None:
                tissue_repr, attn_weights = self.attention(
                    query=patient_enc,
                    keys=tissue_cell_encs,
                    values=tissue_cell_encs,
                    mask=None
                )
            else:
                # Fallback to simple attention (like original THERAPI)
                scores = torch.matmul(patient_enc, tissue_cell_encs.t())  # [batch, n_tissue_cells]
                attn_weights = F.softmax(scores / np.sqrt(self.n_latent), dim=-1)
                tissue_repr = torch.matmul(attn_weights, tissue_cell_encs)  # [batch, n_latent]

            # Weight by tissue relevance
            tissue_weight = tissue_weights[:, tissue_idx:tissue_idx+1]  # [batch, 1]
            weighted_repr = tissue_weight * tissue_repr  # [batch, n_latent]
            weighted_representations.append(weighted_repr)

            # Store attention info if requested
            if return_attention:
                attention_breakdown[tissue_idx] = {
                    'attention_weights': attn_weights.detach(),
                    'n_cells': tissue_cell_encs.size(0),
                    'tissue_weight': tissue_weight.detach()
                }

        # Combine all tissue representations
        if weighted_representations:
            final_representation = torch.stack(weighted_representations, dim=0).sum(dim=0)
        else:
            # Fallback if no valid tissues
            final_representation = patient_enc

        result = {
            'representation': final_representation,
            'tissue_weights': tissue_weights
        }

        if return_attention:
            result['attention_breakdown'] = attention_breakdown

        return result

    def forward(self, patient_expr: torch.Tensor,
                cell_line_exprs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical alignment.

        Args:
            patient_expr: Patient gene expression
            cell_line_exprs: Cell line gene expressions

        Returns:
            Dictionary with representation and tissue weights
        """
        return self.hierarchical_attention_forward(patient_expr, cell_line_exprs)


class HierarchicalResponsePredictor(nn.Module):
    """
    Drug response predictor using hierarchical representations.
    Extends THERAPI's Response_predictor with hierarchical features.
    """

    def __init__(self, emb_dim: int, genef_dim: int, chemical_dim: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()

        # Feature-specific branches (like original THERAPI)
        self.emb_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.genef_fc = nn.Sequential(
            nn.Linear(genef_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.chemical_fc = nn.Sequential(
            nn.Linear(chemical_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim1 * 3, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, emb: torch.Tensor, genef: torch.Tensor,
                chemical: torch.Tensor) -> torch.Tensor:
        """
        Predict drug response.

        Args:
            emb: Patient embedding from hierarchical alignment
            genef: Gene features (perturbation/rank representation)
            chemical: Drug chemical features

        Returns:
            Response prediction [batch_size, output_dim]
        """
        emb_out = self.emb_fc(emb)
        genef_out = self.genef_fc(genef)
        chemical_out = self.chemical_fc(chemical)

        combined = torch.cat([emb_out, genef_out, chemical_out], dim=1)
        output = self.pred_head(combined)

        return output


class HierarchicalTHERAPIFull(nn.Module):
    """
    Complete Hierarchical THERAPI model: Alignment + Prediction.
    """

    def __init__(self, n_genes: int, n_tissues: int, genef_dim: int,
                 chemical_dim: int, n_latent: int = 128,
                 tissue_cell_mask: Optional[torch.Tensor] = None):
        super().__init__()

        # Hierarchical aligner
        self.aligner = HierarchicalTHERAPI(
            n_genes=n_genes,
            n_tissues=n_tissues,
            n_latent=n_latent,
            tissue_cell_mask=tissue_cell_mask
        )

        # Response predictor
        self.predictor = HierarchicalResponsePredictor(
            emb_dim=n_latent,
            genef_dim=genef_dim,
            chemical_dim=chemical_dim,
            hidden_dim1=256,
            hidden_dim2=128,
            output_dim=1
        )

    def forward(self, patient_expr: torch.Tensor,
                cell_line_exprs: torch.Tensor,
                genef: torch.Tensor,
                chemical: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: alignment + prediction.

        Args:
            patient_expr: Patient gene expression
            cell_line_exprs: Cell line expressions
            genef: Gene features
            chemical: Drug features
            return_attention: Whether to return attention details

        Returns:
            Dictionary with prediction and optionally attention breakdown
        """
        # Get hierarchical representation
        alignment_output = self.aligner.hierarchical_attention_forward(
            patient_expr, cell_line_exprs, return_attention=return_attention
        )

        # Predict response
        prediction = self.predictor(
            alignment_output['representation'],
            genef,
            chemical
        )

        result = {
            'prediction': prediction,
            'tissue_weights': alignment_output['tissue_weights'],
            'representation': alignment_output['representation']
        }

        if return_attention:
            result['attention_breakdown'] = alignment_output['attention_breakdown']

        return result
