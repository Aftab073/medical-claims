"""
LangGraph state schema for the medical claims processing pipeline.

Uses Annotated fields with reducer functions so parallel agent outputs
merge cleanly without overwriting each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class PageContent(TypedDict):
    page_number: int          # 1-indexed
    text: str                 # Extracted plain text
    char_count: int           # Quick quality signal
    has_tables: bool          # Detected table presence


class ClassifiedPage(TypedDict):
    page_number: int
    document_type: str        # One of the 9 canonical types
    confidence: float         # 0.0–1.0 from LLM
    reasoning: str            # Brief explanation (helps debuggability)


class ExtractionResult(TypedDict):
    data: dict[str, Any]
    confidence: float         # Aggregate confidence for this extraction
    source_pages: list[int]   # Which pages contributed
    warnings: list[str]       # Non-fatal issues found during extraction


def _merge_optional(a: Optional[ExtractionResult], b: Optional[ExtractionResult]) -> Optional[ExtractionResult]:
    """Last-writer-wins merge — only one agent writes each slot, so this is safe."""
    return b if b is not None else a


class ClaimState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    claim_id: str
    pdf_bytes: bytes                                    # Raw uploaded PDF

    # ── Segregator output ──────────────────────────────────────────────────
    pdf_pages: list[PageContent]                        # Parsed page content
    classified_pages: list[ClassifiedPage]              # Per-page classification
    # Grouped: document_type → list of PageContent
    pages_by_type: dict[str, list[PageContent]]

    # ── Parallel agent outputs (Annotated so LangGraph merges safely) ──────
    patient_info: Annotated[Optional[ExtractionResult], _merge_optional]
    discharge_summary: Annotated[Optional[ExtractionResult], _merge_optional]
    billing: Annotated[Optional[ExtractionResult], _merge_optional]

    # ── Final output ────────────────────────────────────────────────────────
    final_output: Optional[dict[str, Any]]
    processing_errors: Annotated[list[str], operator.add]   # accumulated across nodes