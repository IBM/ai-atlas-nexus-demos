"""Benchmark card composition tool using LLM-based synthesis.

This module provides functionality to compose structured benchmark cards
from heterogeneous metadata sources using large language models. It combines
data from UnitXT, HuggingFace, academic papers, and other sources into
standardized benchmark documentation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Suppress noisy logging from external libraries
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# use the shared llm instance
from auto_benchmarkcard.config import LLM, Config

logger = logging.getLogger(__name__)


# schema for the benchmark card
class BenchmarkDetails(BaseModel):
    """Basic identifying information about a benchmark.

    Attributes:
        name: The official name of the benchmark as it appears in literature.
        overview: A comprehensive 2-3 sentence description explaining what the benchmark measures.
        data_type: The primary data modality (e.g., text, image, audio, multimodal, tabular).
        domains: Specific application domains or subject areas.
        languages: All languages supported in the dataset using full language names.
        similar_benchmarks: Names of closely related or comparable benchmarks.
        resources: URLs to official papers, datasets, leaderboards, and documentation.
    """

    name: str = Field(
        ...,
        description="The official name of the benchmark as it appears in literature",
    )
    overview: str = Field(
        ...,
        description="A comprehensive 2-3 sentence description explaining what the benchmark measures, its key characteristics, and its significance in the field",
    )
    data_type: str = Field(
        ...,
        description="The primary data modality (e.g., text, image, audio, multimodal, tabular)",
    )
    domains: List[str] = Field(
        ...,
        description="Specific application domains or subject areas (e.g., medical, legal, scientific, conversational AI)",
    )
    languages: List[str] = Field(
        ...,
        description="All languages supported in the dataset using full language names (e.g., 'English', 'Chinese', 'Spanish', 'Multilingual')",
    )
    similar_benchmarks: List[str] = Field(
        ...,
        description="Names of closely related or comparable benchmarks that measure similar capabilities",
    )
    resources: List[str] = Field(
        ...,
        description="URLs to official papers, datasets, leaderboards, and documentation",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class PurposeAndIntendedUsers(BaseModel):
    """Purpose, target users, and use case information.

    Attributes:
        goal: The primary objective and research question this benchmark addresses.
        audience: Target user groups for the benchmark.
        tasks: Specific evaluation tasks or subtasks the benchmark covers.
        limitations: Known limitations, biases, or constraints of the benchmark.
        out_of_scope_uses: Explicit examples of inappropriate or unsupported use cases.
    """

    goal: str = Field(
        ...,
        description="The primary objective and research question this benchmark addresses, including what capabilities or behaviors it aims to measure",
    )
    audience: List[str] = Field(
        ...,
        description="Target user groups (e.g., 'AI researchers', 'model developers', 'safety evaluators', 'industry practitioners')",
    )
    tasks: List[str] = Field(
        ...,
        description="Specific evaluation tasks or subtasks the benchmark covers (e.g., 'question answering', 'code generation', 'factual accuracy')",
    )
    limitations: str = Field(
        ...,
        description="Known limitations, biases, or constraints of the benchmark that users should be aware of",
    )
    out_of_scope_uses: List[str] = Field(
        ...,
        description="Explicit examples of inappropriate or unsupported use cases for this benchmark",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class DataInfo(BaseModel):
    """Information about dataset composition and collection.

    Attributes:
        source: Detailed information about data origins and collection methods.
        size: Dataset size with specific numbers.
        format: Data structure, file formats, and organization.
        annotation: Annotation methodology and quality control measures.
    """

    source: str = Field(
        ...,
        description="Detailed information about data origins, collection methods, and any preprocessing steps applied",
    )
    size: str = Field(
        ...,
        description="Dataset size with specific numbers (e.g., '10,000 examples', '50K questions across 3 splits')",
    )
    format: str = Field(
        ...,
        description="Data structure, file formats, and organization (e.g., 'JSON with question-answer pairs', 'CSV with multiple choice options')",
    )
    annotation: str = Field(
        ...,
        description="Annotation methodology, quality control measures, inter-annotator agreement, and any human involvement in labeling",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class Methodology(BaseModel):
    """Evaluation methodology and metric specifications.

    Attributes:
        methods: Evaluation approaches and techniques applied.
        metrics: Specific quantitative metrics used.
        calculation: Detailed explanation of metric computation.
        interpretation: Guidelines for interpreting scores.
        baseline_results: Performance of established models or baselines.
        validation: Quality assurance measures and validation procedures.
    """

    methods: List[str] = Field(
        ...,
        description="Evaluation approaches and techniques applied within the benchmark (e.g., 'zero-shot evaluation', 'few-shot prompting', 'fine-tuning')",
    )
    metrics: List[str] = Field(
        ...,
        description="Specific quantitative metrics used (e.g., 'accuracy', 'F1-score', 'BLEU', 'exact match')",
    )
    calculation: str = Field(
        ...,
        description="Detailed explanation of how metrics are computed, including any normalization or aggregation methods",
    )
    interpretation: str = Field(
        ...,
        description="Guidelines for interpreting scores, including score ranges, what constitutes good performance, and any caveats",
    )
    baseline_results: str = Field(
        ...,
        description="Performance of established models or baselines, with specific numbers and context for comparison",
    )
    validation: str = Field(
        ...,
        description="Quality assurance measures, validation procedures, and steps taken to ensure reliable and reproducible evaluations",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class EthicalAndLegalConsiderations(BaseModel):
    """Ethical and legal aspects of the benchmark.

    Attributes:
        privacy_and_anonymity: Data protection and anonymization measures.
        data_licensing: License terms and usage restrictions.
        consent_procedures: Informed consent processes and participant rights.
        compliance_with_regulations: Adherence to relevant regulations and ethical reviews.
    """

    privacy_and_anonymity: str = Field(
        ...,
        description="Data protection measures, anonymization techniques, and handling of personally identifiable information",
    )
    data_licensing: str = Field(
        ...,
        description="Specific license terms, usage restrictions, and redistribution permissions",
    )
    consent_procedures: str = Field(
        ...,
        description="Details of informed consent processes, participant rights, and withdrawal procedures",
    )
    compliance_with_regulations: str = Field(
        ...,
        description="Adherence to relevant regulations (GDPR, IRB approval, etc.) and ethical review processes",
    )
    provenance: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None,
        description="Source evidence mapping: field_name -> {source, evidence}",
    )


class BenchmarkCard(BaseModel):
    """Complete benchmark card structure.

    Attributes:
        benchmark_details: Basic identifying information.
        purpose_and_intended_users: Purpose and target user information.
        data: Dataset composition and collection details.
        methodology: Evaluation methodology and metrics.
        ethical_and_legal_considerations: Ethical and legal aspects.
    """

    benchmark_details: BenchmarkDetails
    purpose_and_intended_users: PurposeAndIntendedUsers
    data: DataInfo
    methodology: Methodology
    ethical_and_legal_considerations: EthicalAndLegalConsiderations


def extract_provenance(section_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract provenance from section data, returning clean data and provenance separately.

    Args:
        section_data: Section dictionary that may contain a 'provenance' field.

    Returns:
        Tuple of (clean_section_data without provenance, provenance_data).
    """
    # Make a copy to avoid mutating the original
    clean_data = dict(section_data)
    provenance = clean_data.pop("provenance", None) or {}
    return clean_data, provenance


@tool("compose_benchmark_card")
def compose_benchmark_card(
    unitxt_metadata: Dict[str, Any],
    hf_metadata: Optional[Dict[str, Any]] = None,
    extracted_ids: Optional[Dict[str, Any]] = None,
    docling_output: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> Dict[str, Any]:
    """Compose a benchmark card from all the metadata we collected.

    Args:
        unitxt_metadata: Metadata from UnitXT catalog.
        hf_metadata: Optional metadata from HuggingFace.
        extracted_ids: Optional extracted identifier information.
        docling_output: Optional extracted paper content.
        query: Original query string for context.

    Returns:
        Dictionary containing composed benchmark card and composition metadata.
    """

    logger.debug(f"Composing benchmark card for: {query}")

    # Log available data sources
    data_sources = []
    if unitxt_metadata:
        data_sources.append("UnitXT")
    if hf_metadata:
        data_sources.append("HuggingFace")
    if extracted_ids:
        data_sources.append("Extracted IDs")
    if docling_output and docling_output.get("success"):
        data_sources.append("Academic Paper")

    logger.debug(f"Available data sources: {', '.join(data_sources)}")

    # Initialize paper retriever for RAG-lite (index once, retrieve per section)
    paper_retriever = None
    if docling_output and docling_output.get("success"):
        try:
            paper_text = docling_output.get("filtered_text", "")
            if paper_text:
                logger.debug("Initializing paper retriever for RAG-lite")
                # Initialize embeddings based on config
                if Config.DEFAULT_EMBEDDING_MODEL == "bge-large":
                    embeddings = HuggingFaceEmbeddings(
                        model_name="BAAI/bge-large-en-v1.5",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                elif Config.DEFAULT_EMBEDDING_MODEL == "e5-large":
                    embeddings = HuggingFaceEmbeddings(
                        model_name="intfloat/e5-large-v2",
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                else:  # minilm fallback
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )

                # Chunk paper for retrieval (smaller chunks for better precision)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " "]
                )
                chunks = splitter.split_text(paper_text)

                # Create documents
                documents = [
                    Document(page_content=chunk, metadata={"chunk_idx": i})
                    for i, chunk in enumerate(chunks)
                ]

                # Create vectorstore and retriever
                paper_vectorstore = Chroma.from_documents(documents, embeddings)
                paper_retriever = paper_vectorstore.as_retriever(search_kwargs={"k": 3})
                logger.debug(f"Paper indexed: {len(chunks)} chunks ready for retrieval")
        except Exception as e:
            logger.warning(f"Failed to initialize paper retriever: {e}")
            paper_retriever = None

    # define the sections to generate
    sections = [
        ("benchmark_details", BenchmarkDetails),
        ("purpose_and_intended_users", PurposeAndIntendedUsers),
        ("data", DataInfo),
        ("methodology", Methodology),
        ("ethical_and_legal_considerations", EthicalAndLegalConsiderations),
    ]

    # Section-specific query templates for retrieval
    section_queries = {
        "benchmark_details": "benchmark name overview domains languages similar benchmarks resources",
        "data": "dataset size format annotation data collection data statistics",
        "methodology": "evaluation methods metrics calculation baseline results performance",
        "purpose_and_intended_users": "goal purpose motivation audience tasks limitations",
        "ethical_and_legal_considerations": "ethics privacy licensing consent compliance regulations",
    }

    generated_sections = {}
    all_provenance = {}  # Track provenance for all sections

    for section_name, section_class in sections:
        logger.debug("Generating %s", section_name.replace("_", " ").title())

        # Retrieve relevant paper chunks for this section using RAG-lite
        paper_content = "Not available"
        if paper_retriever:
            try:
                query = section_queries.get(section_name, section_name.replace("_", " "))
                relevant_chunks = paper_retriever.get_relevant_documents(query)

                if relevant_chunks:
                    formatted_chunks = []
                    for i, chunk in enumerate(relevant_chunks, 1):
                        formatted_chunks.append(f"[Relevant Paper Section {i}]\n{chunk.page_content}")
                    paper_content = "\n\n".join(formatted_chunks)
                    logger.debug(f"Retrieved {len(relevant_chunks)} paper chunks for {section_name}")
                else:
                    logger.debug(f"No relevant chunks found for {section_name}, using fallback")
                    # Fallback: use first 2000 chars if retrieval fails
                    if docling_output and docling_output.get("filtered_text"):
                        paper_content = docling_output.get("filtered_text", "")[:2000]
            except Exception as e:
                logger.warning(f"Paper retrieval failed for {section_name}: {e}")
                # Fallback: use first 2000 chars
                if docling_output and docling_output.get("filtered_text"):
                    paper_content = docling_output.get("filtered_text", "")[:2000]
        elif docling_output and docling_output.get("success"):
            # No retriever available, use first 2000 chars as fallback
            paper_content = docling_output.get("filtered_text", "Not available")[:2000]

        # Define few-shot examples for each section
        # NOTE: Placeholders like [BENCHMARK_1] are used to prevent the LLM from copying example values
        few_shot_examples = {
            "benchmark_details": {
                "good_example": {
                    "name": "[BENCHMARK_NAME] - use actual name from sources",
                    "overview": "A comprehensive description extracted from the paper abstract or introduction, explaining what the benchmark evaluates and its key characteristics.",
                    "data_type": "text",
                    "domains": [
                        "[DOMAIN_1] - extract from paper",
                        "[DOMAIN_2] - extract from paper",
                    ],
                    "languages": ["[LANGUAGE] - extract from sources"],
                    "similar_benchmarks": ["[BENCHMARK_1] - ONLY if explicitly mentioned in paper", "[BENCHMARK_2] - otherwise use 'Not specified'"],
                    "resources": [
                        "[URL_1] - use actual URLs from sources",
                        "[URL_2] - use actual URLs from sources",
                    ],
                },
                "bad_example": {
                    "name": "prompt_leakage.glue",
                    "overview": "natural language understanding",
                    "data_type": "text",
                    "domains": ["NLP"],
                    "languages": ["en"],
                    "similar_benchmarks": ["D1", "D2"],
                    "resources": ["paper", "dataset"],
                },
            },
            "purpose_and_intended_users": {
                "good_example": {
                    "goal": "Extract the stated purpose/goal from the paper's introduction or abstract. Describe what the benchmark aims to evaluate or achieve.",
                    "audience": [
                        "[AUDIENCE_1] - extract from paper if mentioned",
                        "[AUDIENCE_2] - otherwise use generic ML/NLP audience",
                    ],
                    "tasks": [
                        "[TASK_1] - list actual tasks from sources",
                        "[TASK_2] - list actual tasks from sources",
                    ],
                    "limitations": "Extract limitations explicitly stated in the paper. If none stated, write 'Not specified'",
                    "out_of_scope_uses": [
                        "[USE_1] - extract from paper if mentioned",
                        "Otherwise write 'Not specified'",
                    ],
                }
            },
            "data": {
                "good_example": {
                    "source": "Describe data sources as stated in the paper or HuggingFace metadata",
                    "size": "[NUMBER] examples - USE EXACT COUNT FROM SOURCES (e.g., '1.24 GB' from HuggingFace, or 'Not specified' if not found)",
                    "format": "[FORMAT] - extract from HuggingFace (e.g., 'parquet') or paper, otherwise 'Not specified'",
                    "annotation": "Describe annotation process from paper. If not described, write 'Not specified'",
                },
                "bad_example": {
                    "source": "various sources",
                    "size": "large dataset",
                    "format": "text",
                    "annotation": "manual annotation",
                },
            },
            "methodology": {
                "good_example": {
                    "methods": [
                        "[METHOD_1] - extract evaluation methods from paper",
                        "[METHOD_2] - extract evaluation methods from paper",
                    ],
                    "metrics": [
                        "[METRIC_1] - list metrics explicitly mentioned in sources",
                        "[METRIC_2] - list metrics explicitly mentioned in sources",
                    ],
                    "calculation": "Describe how metrics are calculated IF explicitly stated in paper. Otherwise write 'Not specified'",
                    "interpretation": "Describe score interpretation IF stated in paper. Write 'Not specified' if human baseline not mentioned.",
                    "baseline_results": "[MODEL] achieves [SCORE]% - ONLY include if EXACT numbers appear in paper. Otherwise write 'Not specified'",
                    "validation": "Describe validation approach from paper. If not described, write 'Not specified'",
                }
            },
        }

        section_example = few_shot_examples.get(section_name, {})
        example_text = ""
        if section_example:
            if "good_example" in section_example:
                good_json = (
                    json.dumps(section_example["good_example"], indent=2)
                    .replace("{", "{{")
                    .replace("}", "}}")
                )
                example_text += f"\n\nGOOD EXAMPLE:\n{good_json}"
            if "bad_example" in section_example:
                bad_json = (
                    json.dumps(section_example["bad_example"], indent=2)
                    .replace("{", "{{")
                    .replace("}", "}}")
                )
                example_text += f"\n\nBAD EXAMPLE (avoid this):\n{bad_json}"

        # set up section-specific prompt with enhanced instructions and priority order
        section_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an AI evaluation researcher. Generate a {section_class.__name__} object for the '{section_name}' section.

CRITICAL RULES:
1. Use ONLY information from the provided metadata sources
2. If information is missing, write exactly: "Not specified"
3. Do NOT use your training data or make assumptions
4. Be concise and specific
5. Return only valid JSON

SOURCE PRIORITY (use in this order):
1. Paper Content (HIGHEST PRIORITY - most authoritative source)
2. HuggingFace metadata (official dataset information)
3. UnitXT metadata (catalog metadata)
4. Extracted IDs (for URLs and identifiers)

FORBIDDEN:
- Generic examples (e.g., "BERT-large achieves 80.5%") unless explicitly in sources
- Placeholder names (e.g., "D1", "D2") unless in metadata
- Invented metrics or performance numbers
- Fake URLs or resources
- Rambling or repetitive text
- Copying values from the examples below - they are templates only

FIELD-SPECIFIC RULES (use "Not specified" if not found in sources):
- methodology.baseline_results: ONLY include specific model scores if EXACT numbers appear in paper/sources. Otherwise write "Not specified"
- methodology.interpretation: ONLY include human baseline percentage if paper explicitly states it. Otherwise write "Not specified"
- methodology.calculation: ONLY describe if paper explains how metrics are computed. Otherwise write "Not specified"
- methodology.validation: ONLY describe if paper explains validation approach. Otherwise write "Not specified"
- benchmark_details.similar_benchmarks: ONLY list benchmarks explicitly mentioned/compared in the paper. Otherwise write "Not specified"
- data.size: Use EXACT numbers from sources (e.g., "1.24 GB" from HuggingFace, "10K examples" from paper). Do NOT approximate or invent numbers.
- data.format: Use format from HuggingFace tags (e.g., "parquet") or paper. Otherwise write "Not specified"

PROVENANCE TRACKING (REQUIRED):
For EVERY field you fill in (except "Not specified" values), you MUST add an entry to the "provenance" field.
The provenance field maps each field name to its source and evidence:
{{{{
  "provenance": {{{{
    "field_name": {{{{
      "source": "paper|huggingface|unitxt|extracted_ids",
      "evidence": "exact quote or description from the source"
    }}}}
  }}}}
}}}}
Example: If you set size to "1.24 GB" from HuggingFace, include:
  "provenance": {{{{"size": {{{{"source": "huggingface", "evidence": "Total amount of disk used: 1.24 GB"}}}}}}}}
- Include the EXACT text snippet that supports your value
- Omit fields set to "Not specified" from provenance

{example_text}""",
                ),
                (
                    "user",
                    f"""Query: {{query}}

METADATA SOURCES (in priority order):
1. PAPER CONTENT (highest priority - use this first):
{{paper_content}}

2. HuggingFace Dataset:
{{hf_metadata}}

3. UnitXT Catalog:
{{unitxt_metadata}}

4. Extracted IDs:
{{extracted_ids}}

INSTRUCTIONS:
- Extract information from sources in priority order (1 → 4)
- Paper content is the most authoritative source - use it first
- Only use HuggingFace/UnitXT if information is NOT found in paper
- If a field cannot be found in ANY source, use "Not specified"

Generate {section_name} section using ONLY the metadata above.""",
                ),
            ]
        )

        # configure for structured output
        llm_with_structure = LLM.with_structured_output(section_class)

        # create and run the chain
        chain = section_prompt | llm_with_structure

        # Retry logic for robust generation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Format metadata for prompt (JSON for structured sources)
                hf_formatted = "Not available"
                if hf_metadata:
                    if isinstance(hf_metadata, dict):
                        # Extract most relevant parts from HF metadata
                        hf_parts = []
                        if "card_data" in hf_metadata and hf_metadata["card_data"]:
                            hf_parts.append(f"Card Data:\n{json.dumps(hf_metadata['card_data'], indent=2)}")
                        if "dataset_info" in hf_metadata and hf_metadata["dataset_info"]:
                            hf_parts.append(f"Dataset Info:\n{json.dumps(hf_metadata['dataset_info'], indent=2)}")
                        if hf_parts:
                            hf_formatted = "\n\n".join(hf_parts)
                        else:
                            hf_formatted = json.dumps(hf_metadata, indent=2)
                    else:
                        hf_formatted = str(hf_metadata)

                unitxt_formatted = json.dumps(unitxt_metadata, indent=2) if unitxt_metadata else "Not available"
                extracted_formatted = json.dumps(extracted_ids, indent=2) if extracted_ids else "Not available"

                section_result = chain.invoke(
                    {
                        "query": query,
                        "paper_content": paper_content,
                        "hf_metadata": hf_formatted,
                        "unitxt_metadata": unitxt_formatted,
                        "extracted_ids": extracted_formatted,
                    }
                )

                # Extract provenance from section data
                section_dict = section_result.model_dump()
                clean_section, section_provenance = extract_provenance(section_dict)
                generated_sections[section_name] = clean_section
                if section_provenance:
                    all_provenance[section_name] = section_provenance

                logger.debug("%s completed", section_name.replace("_", " ").title())
                logger.debug("Preview: %s", str(clean_section)[:100] + "...")
                break  # Success, exit retry loop

            except Exception as e:
                attempt_msg = f"(attempt {attempt + 1}/{max_retries})"
                if attempt < max_retries - 1:
                    logger.warning("Failed to generate %s %s: %s", section_name, attempt_msg, e)
                    logger.debug("Retrying %s", section_name)
                    continue
                else:
                    logger.error(
                        "Failed to compose %s section after %d attempts: %s",
                        section_name,
                        max_retries,
                        e,
                    )
                    logger.error(
                        "Failed to generate %s after %d attempts: %s",
                        section_name,
                        max_retries,
                        e,
                    )
                    raise

    # combine all sections into final benchmark card
    logger.debug("Combining all sections into final benchmark card")

    try:
        final_card = BenchmarkCard(
            benchmark_details=BenchmarkDetails(**generated_sections["benchmark_details"]),
            purpose_and_intended_users=PurposeAndIntendedUsers(
                **generated_sections["purpose_and_intended_users"]
            ),
            data=DataInfo(**generated_sections["data"]),
            methodology=Methodology(**generated_sections["methodology"]),
            ethical_and_legal_considerations=EthicalAndLegalConsiderations(
                **generated_sections["ethical_and_legal_considerations"]
            ),
        )

        logger.debug("Final benchmark card assembled successfully")

    except Exception as e:
        logger.error("Failed to assemble final benchmark card: %s", e)
        logger.error("Failed to assemble final card: %s", e)
        raise

    # add metadata about the composition process
    # Exclude provenance from benchmark_card output (it's saved separately)
    benchmark_card_dict = final_card.model_dump(exclude_none=True)
    # Double-check: remove any remaining provenance fields from nested sections
    for section_key in benchmark_card_dict:
        if isinstance(benchmark_card_dict[section_key], dict) and "provenance" in benchmark_card_dict[section_key]:
            del benchmark_card_dict[section_key]["provenance"]

    return {
        "benchmark_card": benchmark_card_dict,
        "provenance": all_provenance if all_provenance else None,
        "composition_metadata": {
            "sources_used": {
                "unitxt": bool(unitxt_metadata),
                "huggingface": bool(hf_metadata),
                "extracted_ids": bool(extracted_ids),
                "docling": bool(docling_output),
            },
            "query": query,
            "composition_timestamp": datetime.now().isoformat(),
            "generation_method": "chunked_sections",
            "model_used": LLM.model_name,
        },
    }
