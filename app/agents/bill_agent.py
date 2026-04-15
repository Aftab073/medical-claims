"""
Itemized Bill Agent — LangGraph node.

Extracts structured billing data from itemized_bill pages:
  items (list), subtotal, tax, discount, total_amount, currency, invoice_number

Includes arithmetic validation: verifies that line items sum to subtotal.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import ClaimState, ExtractionResult
from app.utils.llm_utils import BILL_AGENT_SYSTEM_PROMPT, extract_json
from app.utils.pdf_utils import build_page_context_prompt

logger = logging.getLogger(__name__)

RELEVANT_TYPES = {"itemized_bill"}
AMOUNT_TOLERANCE = 0.05   # 5% tolerance for float rounding in totals


def bill_agent_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: extracts itemized billing information.

    Input state keys consumed : pages_by_type
    Output state keys written : billing, processing_errors
    """
    logger.info("[Bill Agent] Starting for claim_id=%s", state["claim_id"])

    pages_by_type = state.get("pages_by_type", {})
    relevant_pages = []
    for dtype in RELEVANT_TYPES:
        relevant_pages.extend(pages_by_type.get(dtype, []))

    relevant_pages = sorted(relevant_pages, key=lambda p: p["page_number"])

    if not relevant_pages:
        logger.warning("[Bill Agent] No itemized_bill pages found.")
        return {
            "billing": ExtractionResult(
                data={
                    "invoice_number": None,
                    "currency": "INR",
                    "items": [],
                    "subtotal": None,
                    "tax": 0.0,
                    "discount": 0.0,
                    "total_amount": None,
                },
                confidence=0.0,
                source_pages=[],
                warnings=["No itemized_bill pages found in this PDF."],
            ),
            "processing_errors": ["Bill Agent: No itemized_bill pages to process."],
        }

    source_pages = [p["page_number"] for p in relevant_pages]
    logger.info("[Bill Agent] Processing pages: %s", source_pages)

    user_prompt = _build_user_prompt(relevant_pages)
    result = extract_json(
        system_prompt=BILL_AGENT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    if result.get("_error"):
        error_msg = f"Bill Agent LLM extraction failed: {result['_error']}"
        logger.error(error_msg)
        return {
            "billing": ExtractionResult(
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
        "[Bill Agent] Done. Items=%d, total=%s, confidence=%.2f",
        len(data.get("items", [])),
        data.get("total_amount"),
        overall_confidence,
    )

    return {
        "billing": ExtractionResult(
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
        "Extract itemized billing information from the following hospital bill pages.\n\n",
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

    # Parse items, ensuring all monetary values are numbers
    raw_items = raw.get("items", [])
    if not isinstance(raw_items, list):
        raw_items = []
        warnings.append("items was not a list — defaulted to empty.")

    items: list[dict] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            warnings.append(f"Item at index {idx} is not a dict — skipped.")
            continue
        clean_item = {
            "description": str(item.get("description", f"Item {idx + 1}")),
            "quantity":    _to_float(item.get("quantity", 1), warnings, f"items[{idx}].quantity"),
            "rate":        _to_float(item.get("rate", 0), warnings, f"items[{idx}].rate"),
            "amount":      _to_float(item.get("amount", 0), warnings, f"items[{idx}].amount"),
            "category":    str(item.get("category", "miscellaneous")),
        }
        # Derived amount check
        expected = round(clean_item["quantity"] * clean_item["rate"], 2)
        actual   = round(clean_item["amount"], 2)
        if expected != 0 and abs(expected - actual) / max(expected, 0.01) > AMOUNT_TOLERANCE:
            warnings.append(
                f"Item '{clean_item['description']}': qty×rate={expected} but amount={actual}."
            )
        items.append(clean_item)

    subtotal     = _to_float_or_none(raw.get("subtotal"), warnings, "subtotal")
    tax          = _to_float(raw.get("tax", 0), warnings, "tax")
    discount     = _to_float(raw.get("discount", 0), warnings, "discount")
    total_amount = _to_float_or_none(raw.get("total_amount"), warnings, "total_amount")

    # Arithmetic validation
    computed_subtotal = round(sum(i["amount"] for i in items), 2)
    if subtotal is not None and items:
        diff = abs(computed_subtotal - subtotal)
        rel  = diff / max(subtotal, 0.01)
        if rel > AMOUNT_TOLERANCE:
            warnings.append(
                f"Line items sum ({computed_subtotal}) differs from extracted subtotal ({subtotal}) by {rel:.1%}."
            )
    elif subtotal is None and items:
        # Infer subtotal from line items
        subtotal = computed_subtotal
        warnings.append(f"subtotal not found; inferred from line items: {subtotal}.")

    if total_amount is not None and subtotal is not None:
        expected_total = round(subtotal + tax - discount, 2)
        if abs(expected_total - total_amount) / max(total_amount, 0.01) > AMOUNT_TOLERANCE:
            warnings.append(
                f"total_amount ({total_amount}) ≠ subtotal+tax-discount ({expected_total})."
            )

    data = {
        "invoice_number": _str_or_none(raw.get("invoice_number")),
        "currency":       str(raw.get("currency", "INR")),
        "items":          items,
        "subtotal":       subtotal,
        "tax":            tax,
        "discount":       discount,
        "total_amount":   total_amount,
        "item_count":     len(items),
    }

    if not items:
        warnings.append("No billing line items could be extracted.")
    if total_amount is None:
        warnings.append("total_amount not found — critical field missing.")

    return data, warnings


def _to_float(val: Any, warnings: list[str], field: str) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        warnings.append(f"{field}: could not parse '{val}' as number, defaulted to 0.")
        return 0.0


def _to_float_or_none(val: Any, warnings: list[str], field: str) -> Any:
    if val is None or str(val).lower() in ("null", "none", "n/a", ""):
        return None
    return _to_float(val, warnings, field)


def _str_or_none(val: Any) -> Any:
    if val is None or val == "" or str(val).lower() in ("null", "none", "n/a", "na"):
        return None
    return str(val).strip()