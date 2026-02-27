"""HuggingFace dataset metadata extraction tools.

This module provides tools for extracting metadata from HuggingFace datasets
including README content, configurations, and dataset statistics.
"""

from .hf_tool import hf_dataset_metadata

__all__ = ["hf_dataset_metadata"]
