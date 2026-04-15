"""
Discharge Summary Agent — LangGraph node.

Extracts clinical discharge fields from discharge_summary pages:
  diagnosis, admission_date, discharge_date, physician_name,
  hospital_name, procedure_done, discharge_condition
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import ClaimState, ExtractionResult
from app.utils.llm_utils import DISCHARGE_AGENT_SYSTEM_PROMPT, extract_json
from app.utils.pdf_utils import build_page_context_prompt

logger = logging.getLogger(__name__)

RELEVANT_TYPES = {"discharge_summary"}


def discharge_agent_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: extracts discharge summary data.

    Input state keys consumed : pages_by_type
    Output state keys written : discharge_summary, processing_errors
    """
    logger.info("[Discharge Agent] Starting for claim_id=%s", state["claim_id"])

    pages_by_type = state.get("pages_by_type", {})
    relevant_pages = []
    for dtype in RELEVANT_TYPES:
        relevant_pages.extend(pages_by_type.get(dtype, []))

    relevant_pages = sorted(relevant_pages, key=lambda p: p["page_number"])

    if not relevant_pages:
        logger.warning("[Discharge Agent] No discharge_summary pages found.")
        return {
            "discharge_summary": ExtractionResult(
                data={
                    "diagnosis": None,
                    "admission_date": None,
                    "discharge_date": None,
                    "physician_name": None,
                    "hospital_name": None,
                    "procedure_done": [],
                    "discharge_condition": None,
                },
                confidence=0.0,
                source_pages=[],
                warnings=["No discharge_summary pages found in this PDF."],
            ),
            "processing_errors": ["Discharge Agent: No discharge_summary pages to process."],
        }

    source_pages = [p["page_number"] for p in relevant_pages]
    logger.info("[Discharge Agent] Processing pages: %s", source_pages)

    user_prompt = _build_user_prompt(relevant_pages)
    result = extract_json(
        system_prompt=DISCHARGE_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    if result.get("_error"):
        error_msg = f"Discharge Agent LLM extraction failed: {result['_error']}"
        logger.error(error_msg)
        return {
            "discharge_summary": ExtractionResult(
                data={},
                confidence=0.0,
                source_pages=source_pages,
                warnings=[error_msg],
            ),
            "processing_errors": [error_msg],
        }

    data, warnings = _validate_and_clean(result)
    overall_confidence = float(result.get("overall_confidence", 0.5))

    logger.info(
        "[Discharge Agent] Done. Confidence=%.2f, warnings=%d",
        overall_confidence,
        len(warnings),
    )

    return {
        "discharge_summary": ExtractionResult(
            data=data,
            confidence=overall_confidence,
            source_pages=source_pages,
            warnings=warnings,
        ),
        "processing_errors": [],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(pages: list) -> str:
    parts = [
        "Extract discharge summary information from the following clinical document pages.\n\n",
    ]
    for page in pages:
        parts.append(build_page_context_prompt(page))
    parts.append("\nRespond ONLY with the JSON specified in your instructions.")
    return "\n".join(parts)


def _validate_and_clean(raw: dict) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []

    notes = raw.get("extraction_notes", "")
    if notes:
        warnings.append(f"LLM note: {notes}")

    # Ensure procedure_done is always a list
    procedure_done = raw.get("procedure_done", [])
    if not isinstance(procedure_done, list):
        procedure_done = [str(procedure_done)] if procedure_done else []
        warnings.append("procedure_done was not a list — coerced.")

    data = {
        "diagnosis":           _str_or_none(raw.get("diagnosis")),
        "admission_date":      _str_or_none(raw.get("admission_date")),
        "discharge_date":      _str_or_none(raw.get("discharge_date")),
        "physician_name":      _str_or_none(raw.get("physician_name")),
        "hospital_name":       _str_or_none(raw.get("hospital_name")),
        "procedure_done":      procedure_done,
        "discharge_condition": _str_or_none(raw.get("discharge_condition")),
        "field_confidence": {
            "diagnosis":       float(raw.get("confidence_diagnosis", 0.5)),
            "admission_date":  float(raw.get("confidence_admission_date", 0.5)),
            "discharge_date":  float(raw.get("confidence_discharge_date", 0.5)),
            "physician_name":  float(raw.get("confidence_physician_name", 0.5)),
            "hospital_name":   float(raw.get("confidence_hospital_name", 0.5)),
        },
    }

    # Sanity check: admission before discharge
    adm = data["admission_date"]
    dis = data["discharge_date"]
    if adm and dis:
        try:
            from datetime import date
            adm_d = date.fromisoformat(adm)
            dis_d = date.fromisoformat(dis)
            if adm_d > dis_d:
                warnings.append(
                    f"admission_date ({adm}) is AFTER discharge_date ({dis}) — please verify."
                )
        except ValueError:
            warnings.append("Could not validate admission/discharge date ordering.")

    if data["diagnosis"] is None:
        warnings.append("diagnosis not found — critical field missing.")
    if data["hospital_name"] is None:
        warnings.append("hospital_name not found.")

    return data, warnings


def _str_or_none(val: Any) -> Any:
    if val is None or val == "" or str(val).lower() in ("null", "none", "n/a", "na"):
        return None
    return str(val).strip()