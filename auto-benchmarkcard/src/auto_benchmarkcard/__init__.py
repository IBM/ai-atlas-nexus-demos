"""BenchmarkCard - Comprehensive benchmark metadata extraction and validation.

This package provides tools for extracting, validating, and enhancing AI benchmark
metadata through LLM-powered analysis, risk assessment, and factual verification.

Individual tools can be used standalone or as part of the full workflow.

Example (standalone tool usage):
    >>> from auto_benchmarkcard import unitxt_benchmark_lookup
    >>> metadata = unitxt_benchmark_lookup("glue")
    >>> print(metadata.name, metadata.description)

Example (full workflow):
    >>> from auto_benchmarkcard import build_workflow, OutputManager
    >>> workflow = build_workflow()
    >>> output_manager = OutputManager("glue")
    >>> state = workflow.invoke({"query": "glue", "output_manager": output_manager, ...})
"""

__version__ = "0.1.0"
__author__ = "Aris Hofmann"

# Core workflow components
from auto_benchmarkcard.config import Config
from auto_benchmarkcard.workflow import OutputManager, build_workflow

# Individual tools for standalone use
from auto_benchmarkcard.tools.unitxt import UnitxtMetadata, unitxt_benchmark_lookup
from auto_benchmarkcard.tools.hf import hf_dataset_metadata
from auto_benchmarkcard.tools.extractor import extract_ids
from auto_benchmarkcard.tools.rag import RAGRetriever, MetadataIndexer, atomize_benchmark_card
from auto_benchmarkcard.tools.composer import compose_benchmark_card
from auto_benchmarkcard.tools.docling import extract_paper_with_docling

__all__ = [
    # Workflow
    "Config",
    "build_workflow",
    "OutputManager",
    # UnitXT
    "unitxt_benchmark_lookup",
    "UnitxtMetadata",
    # HuggingFace
    "hf_dataset_metadata",
    # Extractor
    "extract_ids",
    # RAG
    "RAGRetriever",
    "MetadataIndexer",
    "atomize_benchmark_card",
    # Composer
    "compose_benchmark_card",
    # Docling
    "extract_paper_with_docling",
]
