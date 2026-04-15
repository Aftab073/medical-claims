"""
Segregator Agent — LangGraph node.

Responsibilities
----------------
1. Parse PDF bytes into per-page content (delegates to pdf_utils).
2. Call the LLM for each page to classify its document type.
3. Group pages by document type → pages_by_type.
4. Apply fallback classification when LLM confidence is too low.

This node NEVER passes the full PDF to downstream agents.
It only passes the relevant slices via the pages_by_type state key.
"""

from __future__ import annotations

import logging
from typing import Any

from app.graph.state import ClassifiedPage, ClaimState, PageContent
from app.utils.llm_utils import classify_page_json
from app.utils.pdf_utils import extract_pages

logger = logging.getLogger(__name__)

# Canonical document type set — any LLM output outside this is rejected
VALID_DOCUMENT_TYPES = {
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
}

# Pages with confidence below this threshold get fallback treatment
CONFIDENCE_THRESHOLD = 0.45


def segregator_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: segregates a PDF into typed page groups.

    Input state keys consumed : pdf_bytes
    Output state keys written : pdf_pages, classified_pages, pages_by_type,
                                processing_errors
    """
    logger.info("[Segregator] Starting PDF extraction for claim_id=%s", state["claim_id"])
    errors: list[str] = []

    # ── Step 1: Extract raw text from every page ──────────────────────────
    try:
        pdf_pages: list[PageContent] = extract_pages(state["pdf_bytes"])
    except ValueError as exc:
        logger.error("[Segregator] PDF extraction failed: %s", exc)
        # Return a minimal state so downstream nodes can handle gracefully
        return {
            "pdf_pages": [],
            "classified_pages": [],
            "pages_by_type": {},
            "processing_errors": [f"PDF extraction failed: {exc}"],
        }

    if not pdf_pages:
        return {
            "pdf_pages": [],
            "classified_pages": [],
            "pages_by_type": {},
            "processing_errors": ["PDF produced no extractable pages."],
        }

    # ── Step 2: Classify each page with LLM ──────────────────────────────
    classified_pages: list[ClassifiedPage] = []
    for page in pdf_pages:
        classification = _classify_single_page(page, errors)
        classified_pages.append(classification)
        logger.info(
            "[Segregator] Page %d → %s (conf=%.2f)",
            page["page_number"],
            classification["document_type"],
            classification["confidence"],
        )

    # ── Step 3: Group pages by document type ─────────────────────────────
    pages_by_type: dict[str, list[PageContent]] = _group_pages(pdf_pages, classified_pages)

    # Summary log
    for dtype, pages in pages_by_type.items():
        logger.info(
            "[Segregator] Type '%s': pages %s",
            dtype,
            [p["page_number"] for p in pages],
        )

    return {
        "pdf_pages": pdf_pages,
        "classified_pages": classified_pages,
        "pages_by_type": pages_by_type,
        "processing_errors": errors,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_single_page(page: PageContent, errors: list[str]) -> ClassifiedPage:
    """
    Call the LLM to classify one page.
    Applies fallback logic on low confidence or invalid type.
    """
    page_text = page["text"]

    # Fast path: empty / very short pages → 'other'
    if page["char_count"] < 30:
        return ClassifiedPage(
            page_number=page["page_number"],
            document_type="other",
            confidence=0.95,
            reasoning="Page contains almost no text; classified as 'other'.",
        )

    result = classify_page_json(page_text, page["page_number"])

    # Handle LLM error response
    if result.get("_error"):
        errors.append(
            f"Page {page['page_number']}: LLM classification failed — {result['_error']}"
        )
        return _fallback_classify(page)

    doc_type = result.get("document_type", "other")
    confidence = float(result.get("confidence", 0.0))
    reasoning = result.get("reasoning", "No reasoning provided.")

    # Validate doc type
    if doc_type not in VALID_DOCUMENT_TYPES:
        logger.warning(
            "[Segregator] LLM returned unknown type '%s' for page %d, using 'other'.",
            doc_type,
            page["page_number"],
        )
        errors.append(
            f"Page {page['page_number']}: Unknown document type '{doc_type}', defaulted to 'other'."
        )
        doc_type = "other"

    # Low confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(
            "[Segregator] Low confidence (%.2f) on page %d, applying keyword fallback.",
            confidence,
            page["page_number"],
        )
        fallback = _fallback_classify(page)
        errors.append(
            f"Page {page['page_number']}: LLM confidence {confidence:.2f} below threshold; "
            f"keyword fallback applied → '{fallback['document_type']}'."
        )
        return fallback

    return ClassifiedPage(
        page_number=page["page_number"],
        document_type=doc_type,
        confidence=confidence,
        reasoning=reasoning,
    )


def _fallback_classify(page: PageContent) -> ClassifiedPage:
    """
    Keyword-based heuristic classifier used when LLM confidence is too low
    or the LLM call fails entirely.
    """
    text_lower = page["text"].lower()

    rules: list[tuple[str, list[str]]] = [
        ("discharge_summary",     ["discharge summary", "date of discharge", "diagnosis at discharge", "attending physician"]),
        ("itemized_bill",         ["total amount", "subtotal", "invoice no", "bill no", "item description", "charges"]),
        ("identity_document",     ["aadhar", "aadhaar", "pan card", "passport no", "date of birth", "voter id"]),
        ("claim_forms",           ["claim form", "policy number", "insured name", "claim number", "tpa"]),
        ("prescription",          ["rx", "prescribed by", "sig:", "tablet", "capsule", "dosage"]),
        ("investigation_report",  ["lab report", "haemoglobin", "wbc", "rbc", "radiology", "pathology", "specimen"]),
        ("cash_receipt",          ["cash receipt", "received with thanks", "amount received", "receipt no"]),
        ("cheque_or_bank_details",["cancelled cheque", "account number", "ifsc", "neft", "bank name", "branch"]),
    ]

    for doc_type, keywords in rules:
        if any(kw in text_lower for kw in keywords):
            matched = [kw for kw in keywords if kw in text_lower]
            return ClassifiedPage(
                page_number=page["page_number"],
                document_type=doc_type,
                confidence=0.55,
                reasoning=f"Keyword fallback matched: {matched[:3]}",
            )

    return ClassifiedPage(
        page_number=page["page_number"],
        document_type="other",
        confidence=0.40,
        reasoning="No keywords matched; defaulted to 'other'.",
    )


def _group_pages(
    pdf_pages: list[PageContent],
    classified_pages: list[ClassifiedPage],
) -> dict[str, list[PageContent]]:
    """
    Build a dict mapping document_type → list of PageContent objects
    so each downstream agent receives only its relevant pages.
    """
    # Index pdf_pages by page number for O(1) lookup
    page_map: dict[int, PageContent] = {p["page_number"]: p for p in pdf_pages}

    groups: dict[str, list[PageContent]] = {}
    for cp in classified_pages:
        dtype = cp["document_type"]
        if dtype not in groups:
            groups[dtype] = []
        page_content = page_map.get(cp["page_number"])
        if page_content:
            groups[dtype].append(page_content)

    return groups