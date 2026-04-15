"""
Aggregator Node — LangGraph node.

Final step in the pipeline. Combines:
  * patient_info     (from ID Agent)
  * discharge_summary (from Discharge Summary Agent)
  * billing           (from Bill Agent)

Produces the canonical response JSON and a pipeline-wide quality summary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.graph.state import ClaimState, ExtractionResult

logger = logging.getLogger(__name__)


def aggregator_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: merges all extraction results into the final response.

    Input state keys consumed : claim_id, classified_pages, patient_info,
                                discharge_summary, billing, processing_errors
    Output state keys written : final_output
    """
    logger.info("[Aggregator] Building final output for claim_id=%s", state["claim_id"])

    patient_info: Optional[ExtractionResult]     = state.get("patient_info")
    discharge_summary: Optional[ExtractionResult] = state.get("discharge_summary")
    billing: Optional[ExtractionResult]           = state.get("billing")
    all_errors: list[str]                         = state.get("processing_errors", [])

    # ── Build per-section output ──────────────────────────────────────────
    patient_section      = _build_section(patient_info,     "patient_info")
    discharge_section    = _build_section(discharge_summary, "discharge_summary")
    billing_section      = _build_section(billing,          "billing")

    # ── Quality summary ───────────────────────────────────────────────────
    quality = _build_quality_summary(
        state.get("classified_pages", []),
        patient_info,
        discharge_summary,
        billing,
        all_errors,
    )

    # ── Final response ────────────────────────────────────────────────────
    final_output: dict[str, Any] = {
        "claim_id":         state["claim_id"],
        "processed_at":     datetime.now(timezone.utc).isoformat(),
        "patient_info":     patient_section,
        "discharge_summary": discharge_section,
        "billing":          billing_section,
        "quality":          quality,
    }

    logger.info(
        "[Aggregator] Done. Overall quality score: %.2f",
        quality["overall_confidence"],
    )

    return {"final_output": final_output}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_section(
    result: Optional[ExtractionResult],
    section_name: str,
) -> dict[str, Any]:
    """
    Convert an ExtractionResult into a response-level section dict.
    Gracefully handles None (agent was skipped or errored).
    """
    if result is None:
        logger.warning("[Aggregator] %s result is None — no data.", section_name)
        return {
            "data":         {},
            "confidence":   0.0,
            "source_pages": [],
            "warnings":     [f"{section_name} extraction was not performed."],
        }

    return {
        "data":         result["data"],
        "confidence":   round(result["confidence"], 3),
        "source_pages": result["source_pages"],
        "warnings":     result["warnings"],
    }


def _build_quality_summary(
    classified_pages: list,
    patient_info: Optional[ExtractionResult],
    discharge_summary: Optional[ExtractionResult],
    billing: Optional[ExtractionResult],
    errors: list[str],
) -> dict[str, Any]:
    """
    Build a pipeline-wide quality report:
    * per-agent confidence scores
    * overall weighted confidence
    * page classification summary
    * pipeline errors collected across all nodes
    """
    pi_conf  = patient_info["confidence"]      if patient_info      else 0.0
    ds_conf  = discharge_summary["confidence"] if discharge_summary  else 0.0
    bil_conf = billing["confidence"]           if billing            else 0.0

    # Weighted average (billing and discharge slightly more critical for claims)
    weights       = [1.0, 1.5, 1.5]
    total_weight  = sum(weights)
    overall_conf  = round(
        (pi_conf * weights[0] + ds_conf * weights[1] + bil_conf * weights[2]) / total_weight,
        3,
    )

    # Classification summary
    type_summary: dict[str, list[int]] = {}
    for cp in classified_pages:
        dtype = cp["document_type"]
        if dtype not in type_summary:
            type_summary[dtype] = []
        type_summary[dtype].append(cp["page_number"])

    avg_class_conf = (
        round(sum(cp["confidence"] for cp in classified_pages) / len(classified_pages), 3)
        if classified_pages else 0.0
    )

    return {
        "overall_confidence":       overall_conf,
        "classification_confidence": avg_class_conf,
        "agent_confidence": {
            "patient_info":      round(pi_conf, 3),
            "discharge_summary": round(ds_conf, 3),
            "billing":           round(bil_conf, 3),
        },
        "pages_classified":    len(classified_pages),
        "document_type_pages": type_summary,
        "pipeline_errors":     errors,
        "error_count":         len(errors),
    }