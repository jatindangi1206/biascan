"""
Dataset Transform Registry — STEP 00

Parses the DATASET_SPEC table, extracts "How to Use" logic,
and registers each as a callable transform_<DATASET_ID>() function.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EvalSample:
    """A single evaluation sample with provenance."""
    id: str                              # unique ID: "{dataset}_{index}"
    dataset_id: str                      # e.g. "B01_scifact"
    bias_type: str                       # e.g. "B01", "B03", "B08"
    agent: str                           # which agent should catch this: "argus", "libra", "lens"
    input_text: str                      # the text to feed BiasScan
    label: str                           # "biased" or "control"
    bias_present: bool                   # True = biased, False = control
    original_fields: dict = field(default_factory=dict)   # raw dataset fields for debugging
    transform_description: str = ""      # what transform was applied


@dataclass
class DatasetSpec:
    """Spec for one evaluation dataset."""
    dataset_id: str                      # e.g. "B01_scifact"
    bias_type: str
    dataset_name: str
    agent: str
    how_to_use: str                      # from the table
    transform_fn: Optional[Callable] = None  # transform_<id>() function
    sample_count: int = 0


# Global registry
TRANSFORM_REGISTRY: dict[str, DatasetSpec] = {}


def register_transform(spec: DatasetSpec) -> None:
    """Register a dataset transform."""
    TRANSFORM_REGISTRY[spec.dataset_id] = spec


def get_all_specs() -> list[DatasetSpec]:
    """Get all registered dataset specs."""
    return list(TRANSFORM_REGISTRY.values())


def get_spec(dataset_id: str) -> DatasetSpec:
    """Get a specific dataset spec."""
    return TRANSFORM_REGISTRY[dataset_id]
