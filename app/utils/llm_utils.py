"""
LLM utilities — thin wrappers around LangChain's ChatGroq.

Provides:
* get_llm()           — singleton-ish LLM handle (model configurable via env)
* extract_json()      — prompt → validated dict, with retry and fallback
* build_system_msg()  — consistent system-prompt builder
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))    # deterministic
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm(model: Optional[str] = None, temperature: Optional[float] = None) -> ChatGroq:
    """
    Return a cached ChatGroq instance.
    API key is read from GROQ_API_KEY env var automatically by LangChain.
    """
    return ChatGroq(
        model=model or DEFAULT_MODEL,
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_retries=2,        # network-level retries by LangChain
        request_timeout=60,
    )


# ---------------------------------------------------------------------------
# Core extraction helper
# ---------------------------------------------------------------------------

def extract_json(
    system_prompt: str,
    user_prompt: str,
    schema_hint: Optional[str] = None,
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """
    Call the LLM with the given prompts and parse the response as JSON.

    * Strips markdown fences before parsing.
    * Retries up to `retries` times on JSON decode errors.
    * Returns an empty dict with an `_error` key on total failure (never raises).

    Args:
        system_prompt: The system instruction.
        user_prompt:   The user content (page text, etc.).
        schema_hint:   Optional JSON schema string appended to system prompt.
        retries:       How many attempts before giving up.

    Returns:
        Parsed dict — always a dict, never None.
    """
    llm = get_llm()
    full_system = system_prompt
    if schema_hint:
        full_system += f"\n\nExpected JSON schema:\n{schema_hint}"

    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_prompt),
    ]

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            return _parse_json_response(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_DELAY)
        except Exception as exc:
            last_error = exc
            logger.error("LLM call failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(RETRY_DELAY)

    logger.error("All %d LLM attempts failed. Last error: %s", retries, last_error)
    return {"_error": str(last_error), "_partial": True}


def classify_page_json(page_text: str, page_number: int) -> dict[str, Any]:
    """
    Classify a single PDF page using the LLM.
    Returns: {document_type, confidence, reasoning}
    """
    system = SEGREGATOR_SYSTEM_PROMPT
    user = (
        f"Classify PAGE {page_number}:\n\n"
        f"{page_text[:3500]}\n\n"   # hard cap to avoid token blow-up
        "Respond ONLY with valid JSON."
    )
    return extract_json(system, user)


# ---------------------------------------------------------------------------
# System prompts — centralised for consistency
# ---------------------------------------------------------------------------

SEGREGATOR_SYSTEM_PROMPT = """You are a medical document classifier specialising in insurance claim packages.

Your job: classify a single PDF page into EXACTLY ONE document type.

Document types (use these exact strings):
- claim_forms           : Insurance claim forms, patient registration, pre-auth requests
- cheque_or_bank_details: Cancelled cheques, NEFT forms, bank account details
- identity_document     : Aadhar card, PAN, passport, driving license, voter ID
- itemized_bill         : Hospital bills with itemised line items, charges, totals
- discharge_summary     : Clinical discharge notes, diagnosis, treatment summary
- prescription          : Doctor's prescription, medication orders
- investigation_report  : Lab reports, radiology, pathology, blood test results
- cash_receipt          : Payment receipts, cash memos
- other                 : Anything that doesn't fit above

Rules:
1. Analyse ALL text on the page before deciding.
2. If the page has mixed content, choose the DOMINANT type.
3. Confidence must reflect actual certainty (0.0-1.0).
4. Reasoning must be one concise sentence explaining key signals.

Respond ONLY with this JSON (no markdown, no extra keys):
{
  "document_type": "<one of the exact strings above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}"""


ID_AGENT_SYSTEM_PROMPT = """You are a medical claim data extractor specialising in PATIENT IDENTITY information.

Extract the following fields from the provided document pages:
- patient_name        : Full name of the patient
- date_of_birth       : DOB in ISO format YYYY-MM-DD (convert from any format)
- id_number           : Government ID number (Aadhar, PAN, Passport, etc.) — include type prefix e.g. "AADHAR: 1234..."
- policy_number       : Insurance policy number
- contact_details     : Object with keys: phone, email, address (null if not found)

Rules:
1. If a field is genuinely absent, set it to null — NEVER fabricate.
2. Normalise dates to YYYY-MM-DD.
3. For each field, provide a confidence_<field> float (0.0-1.0).
4. Extract from ALL provided pages; later pages override earlier if conflicting.

Respond ONLY with this JSON:
{
  "patient_name": "<string or null>",
  "date_of_birth": "<YYYY-MM-DD or null>",
  "id_number": "<string or null>",
  "policy_number": "<string or null>",
  "contact_details": {
    "phone": "<string or null>",
    "email": "<string or null>",
    "address": "<string or null>"
  },
  "confidence_patient_name": <float>,
  "confidence_date_of_birth": <float>,
  "confidence_id_number": <float>,
  "confidence_policy_number": <float>,
  "confidence_contact_details": <float>,
  "overall_confidence": <float>,
  "extraction_notes": "<any caveats or observations>"
}"""


DISCHARGE_AGENT_SYSTEM_PROMPT = """You are a medical claim data extractor specialising in DISCHARGE SUMMARY information.

Extract the following fields from the provided clinical documents:
- diagnosis           : Primary diagnosis (and secondary if present) — preserve medical terminology
- admission_date      : Date of admission in ISO format YYYY-MM-DD
- discharge_date      : Date of discharge in ISO format YYYY-MM-DD
- physician_name      : Treating/consulting physician full name and designation
- hospital_name       : Full name of the hospital/clinic
- procedure_done      : Procedures or surgeries performed (list, empty list if none)
- discharge_condition : Patient condition at discharge (e.g. "Stable", "Improved")

Rules:
1. Null for genuinely missing fields — never guess.
2. Preserve exact medical terminology for diagnosis.
3. For each field, provide a confidence_<field> float (0.0-1.0).

Respond ONLY with this JSON:
{
  "diagnosis": "<string or null>",
  "admission_date": "<YYYY-MM-DD or null>",
  "discharge_date": "<YYYY-MM-DD or null>",
  "physician_name": "<string or null>",
  "hospital_name": "<string or null>",
  "procedure_done": ["<string>", ...],
  "discharge_condition": "<string or null>",
  "confidence_diagnosis": <float>,
  "confidence_admission_date": <float>,
  "confidence_discharge_date": <float>,
  "confidence_physician_name": <float>,
  "confidence_hospital_name": <float>,
  "overall_confidence": <float>,
  "extraction_notes": "<any caveats>"
}"""


BILL_AGENT_SYSTEM_PROMPT = """You are a medical claim data extractor specialising in ITEMIZED BILLING information.

Extract the following from hospital bills / invoices:
- items        : Array of line items, each with: description, quantity (number), rate (number), amount (number), category
- subtotal     : Pre-tax subtotal (number)
- tax          : Tax amount (number, 0 if not applicable)
- discount     : Discount applied (number, 0 if none)
- total_amount : Final payable amount (number)
- currency     : Currency code (e.g. "INR", "USD") — default "INR" if ambiguous
- invoice_number: Invoice/bill number if present

Rules:
1. All monetary values must be numbers (float), never strings.
2. If quantity is missing for a line item, use 1.
3. Categorise items as one of: room_charges, doctor_fees, pharmacy, procedures, diagnostics, miscellaneous.
4. Verify: sum(items[].amount) should approximately equal subtotal. Flag discrepancy in extraction_notes.
5. Null for missing top-level fields, empty array [] for items if none found.

Respond ONLY with this JSON:
{
  "invoice_number": "<string or null>",
  "currency": "<string>",
  "items": [
    {
      "description": "<string>",
      "quantity": <number>,
      "rate": <number>,
      "amount": <number>,
      "category": "<string>"
    }
  ],
  "subtotal": <number or null>,
  "tax": <number>,
  "discount": <number>,
  "total_amount": <number or null>,
  "overall_confidence": <float>,
  "extraction_notes": "<any caveats>"
}"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict[str, Any]:
    """
    Strip markdown fences and parse JSON from an LLM response string.
    Raises json.JSONDecodeError if no valid JSON is found.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block in the string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise json.JSONDecodeError("No JSON object found in LLM response", raw, 0)