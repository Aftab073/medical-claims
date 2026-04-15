"""
ID Agent — LangGraph node.

Extracts patient identity fields from:
  * identity_document pages  (Aadhar, PAN, Passport, etc.)
  * claim_forms pages        (often contains policy + patient info)

Fields extracted:
  patient_name, date_of_birth, id_number, policy_number, contact_details

Returns an ExtractionResult with data, aggregate confidence, source pages, and warnings.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import ClaimState, ExtractionResult
from app.utils.llm_utils import ID_AGENT_SYSTEM_PROMPT, extract_json
from app.utils.pdf_utils import build_page_context_prompt

logger = logging.getLogger(__name__)

# Document types this agent is authorised to consume
RELEVANT_TYPES = {"identity_document", "claim_forms"}


def id_agent_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: extracts patient identity information.

    Input state keys consumed : pages_by_type
    Output state keys written : patient_info, processing_errors
    """
    logger.info("[ID Agent] Starting extraction for claim_id=%s", state["claim_id"])

    pages_by_type = state.get("pages_by_type", {})
    relevant_pages = _collect_relevant_pages(pages_by_type)

    if not relevant_pages:
        logger.warning("[ID Agent] No relevant pages found (identity_document / claim_forms).")
        return {
            "patient_info": ExtractionResult(
                data={
                    "patient_name": None,
                    "date_of_birth": None,
                    "id_number": None,
                    "policy_number": None,
                    "contact_details": {"phone": None, "email": None, "address": None},
                },
                confidence=0.0,
                source_pages=[],
                warnings=["No identity_document or claim_forms pages found in this PDF."],
            ),
            "processing_errors": ["ID Agent: No relevant pages to process."],
        }

    source_pages = [p["page_number"] for p in relevant_pages]
    logger.info("[ID Agent] Processing pages: %s", source_pages)

    # Build combined page context for the LLM
    user_prompt = _build_user_prompt(relevant_pages)

    # Call LLM
    result = extract_json(
        system_prompt=ID_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # Handle LLM failure
    if result.get("_error"):
        error_msg = f"ID Agent LLM extraction failed: {result['_error']}"
        logger.error(error_msg)
        return {
            "patient_info": ExtractionResult(
                data={},
                confidence=0.0,
                source_pages=source_pages,
                warnings=[error_msg],
            ),
            "processing_errors": [error_msg],
        }

    # Post-process and validate
    data, warnings = _validate_and_clean(result)
    overall_confidence = float(result.get("overall_confidence", 0.5))

    logger.info(
        "[ID Agent] Extraction complete. Confidence=%.2f, warnings=%d",
        overall_confidence,
        len(warnings),
    )

    return {
        "patient_info": ExtractionResult(
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

def _collect_relevant_pages(pages_by_type: dict) -> list:
    """Merge pages from all relevant document types, sorted by page number."""
    pages = []
    for dtype in RELEVANT_TYPES:
        pages.extend(pages_by_type.get(dtype, []))
    # De-duplicate and sort
    seen: set[int] = set()
    unique_sorted = []
    for p in sorted(pages, key=lambda x: x["page_number"]):
        if p["page_number"] not in seen:
            seen.add(p["page_number"])
            unique_sorted.append(p)
    return unique_sorted


def _build_user_prompt(pages: list) -> str:
    """Combine multiple page contexts into a single user prompt."""
    prompt_parts = [
        "Extract patient identity information from the following document pages.\n",
        "If the same field appears on multiple pages, prefer the most detailed / legible version.\n\n",
    ]
    for page in pages:
        prompt_parts.append(build_page_context_prompt(page))
    prompt_parts.append("\nRespond ONLY with the JSON specified in your instructions.")
    return "\n".join(prompt_parts)


def _validate_and_clean(raw: dict) -> tuple[dict[str, Any], list[str]]:
    """
    Extract the core data fields, normalise them, and collect any warnings.
    Returns (clean_data_dict, warnings_list).
    """
    warnings: list[str] = []

    # Extract notes before stripping
    notes = raw.get("extraction_notes", "")
    if notes:
        warnings.append(f"LLM note: {notes}")

    data = {
        "patient_name": _str_or_none(raw.get("patient_name")),
        "date_of_birth": _str_or_none(raw.get("date_of_birth")),
        "id_number": _str_or_none(raw.get("id_number")),
        "policy_number": _str_or_none(raw.get("policy_number")),
        "contact_details": _parse_contact(raw.get("contact_details"), warnings),
        "field_confidence": {
            "patient_name":     float(raw.get("confidence_patient_name", 0.5)),
            "date_of_birth":    float(raw.get("confidence_date_of_birth", 0.5)),
            "id_number":        float(raw.get("confidence_id_number", 0.5)),
            "policy_number":    float(raw.get("confidence_policy_number", 0.5)),
            "contact_details":  float(raw.get("confidence_contact_details", 0.5)),
        },
    }

    # Missing critical field warnings
    if data["patient_name"] is None:
        warnings.append("patient_name not found in any identity document.")
    if data["policy_number"] is None:
        warnings.append("policy_number not found — may need manual lookup.")

    return data, warnings


def _str_or_none(val: Any) -> Any:
    if val is None or val == "" or str(val).lower() in ("null", "none", "n/a", "na"):
        return None
    return str(val).strip()


def _parse_contact(raw_contact: Any, warnings: list[str]) -> dict:
    default = {"phone": None, "email": None, "address": None}
    if not isinstance(raw_contact, dict):
        warnings.append("contact_details not returned as expected object.")
        return default
    return {
        "phone":   _str_or_none(raw_contact.get("phone")),
        "email":   _str_or_none(raw_contact.get("email")),
        "address": _str_or_none(raw_contact.get("address")),
    }