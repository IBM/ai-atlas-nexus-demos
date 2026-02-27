"""Metadata extraction tools for benchmark identifiers.

This module provides tools for extracting HuggingFace repository names,
paper URLs, and other metadata from UnitXT configurations.
"""

from .extractor_tool import extract_ids

__all__ = ["extract_ids"]
