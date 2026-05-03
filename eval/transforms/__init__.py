"""
Dataset-specific transformation functions for BiasScan evaluation.
Each transform implements the exact logic from EVAL_DATASETS.md "How to Use" column.

Importing this package registers all transforms in the global registry.
"""

# Import all transform modules to trigger registration
from eval.transforms import b01_confirmation
from eval.transforms import b03_certainty
from eval.transforms import b04_overgeneralization
from eval.transforms import b05_framing
from eval.transforms import b08_causal
from eval.transforms import libra_hedging
from eval.transforms import lens_discourse

from eval.transforms.registry import (
    EvalSample,
    DatasetSpec,
    TRANSFORM_REGISTRY,
    get_all_specs,
    get_spec,
)

__all__ = [
    "EvalSample",
    "DatasetSpec",
    "TRANSFORM_REGISTRY",
    "get_all_specs",
    "get_spec",
]
