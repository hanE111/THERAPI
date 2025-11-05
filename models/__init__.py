"""
Hierarchical THERAPI models package.

This package contains:
- Hierarchical THERAPI main models (hierarchical_therapi.py)
- Ablation study variants (ablation_models.py)
"""

from .hierarchical_therapi import (
    TissueRouter,
    HierarchicalAttention,
    HierarchicalTHERAPI,
    HierarchicalResponsePredictor,
    HierarchicalTHERAPIFull,
)

from .ablation_models import (
    FlatTHERAPI,
    UniformTissueWeightTHERAPI,
    HardRoutingTHERAPI,
    SingleHeadTHERAPI,
    FixedTemperatureTHERAPI,
    RandomRoutingTHERAPI,
    TrueOracleRoutingTHERAPI,
    SimplifiedHierarchicalTHERAPI,
    create_ablation_model,
)

__all__ = [
    # Main models
    'TissueRouter',
    'HierarchicalAttention',
    'HierarchicalTHERAPI',
    'HierarchicalResponsePredictor',
    'HierarchicalTHERAPIFull',
    # Ablation models
    'FlatTHERAPI',
    'UniformTissueWeightTHERAPI',
    'HardRoutingTHERAPI',
    'SingleHeadTHERAPI',
    'FixedTemperatureTHERAPI',
    'RandomRoutingTHERAPI',
    'TrueOracleRoutingTHERAPI',
    'SimplifiedHierarchicalTHERAPI',
    'create_ablation_model',
]
