"""
Medical Claims Processing API — main FastAPI application.

Endpoints
---------
POST /api/process  — upload a claim PDF, returns structured extraction JSON
GET  /health       — liveness probe
GET  /             — API info
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging setup (before any local imports) ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Local imports ─────────────────────────────────────────────────────────
from app.graph.workflow import claims_graph
from app.graph.state import ClaimState


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Medical Claims API starting up…")
    yield
    logger.info("Medical Claims API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Medical Claims Processor",
    description=(
        "Processes medical claim PDFs using a LangGraph multi-agent pipeline. "
        "Extracts patient identity, discharge summary, and itemized billing data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants & limits
# ---------------------------------------------------------------------------

MAX_PDF_SIZE_MB   = int(os.getenv("MAX_PDF_SIZE_MB", "20"))
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


# ---------------------------------------------------------------------------
# Request validation layer
# ---------------------------------------------------------------------------

def _validate_claim_id(claim_id: str) -> str:
    """Strip and validate claim_id — must be 1-100 printable chars."""
    cid = claim_id.strip()
    if not cid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="claim_id must not be empty.",
        )
    if len(cid) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="claim_id must be ≤ 100 characters.",
        )
    return cid


async def _validate_pdf_file(file: UploadFile) -> bytes:
    """Read and validate the uploaded PDF file."""
    # Content-type check (best-effort — browsers/clients may vary)
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Expected a PDF file, got content-type: {file.content_type}",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDF exceeds maximum allowed size of {MAX_PDF_SIZE_MB} MB.",
        )

    # PDF magic bytes check: %PDF
    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not appear to be a valid PDF (missing %PDF header).",
        )

    return pdf_bytes


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
async def root() -> dict[str, str]:
    return {
        "service": "Medical Claims Processor",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["Meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process", tags=["Claims"])
async def process_claim(
    request: Request,
    claim_id: str = Form(..., description="Unique identifier for this claim"),
    file: UploadFile = File(..., description="The medical claim PDF file"),
) -> JSONResponse:
    """
    Process a medical claim PDF.

    **Workflow**:
    1. Validates input (claim_id format + PDF integrity).
    2. Invokes the LangGraph pipeline:
       - Segregator: classifies each page by document type.
       - ID Agent: extracts patient identity information.
       - Discharge Agent: extracts clinical discharge data.
       - Bill Agent: extracts itemized billing data.
       - Aggregator: merges all results into a single JSON.
    3. Returns structured JSON with extracted data + confidence scores.
    """
    request_id = str(uuid.uuid4())[:8]
    start_ts   = time.perf_counter()

    logger.info(
        "[%s] POST /api/process — claim_id=%r, filename=%r",
        request_id,
        claim_id,
        file.filename,
    )

    # ── Input validation ──────────────────────────────────────────────────
    claim_id  = _validate_claim_id(claim_id)
    pdf_bytes = await _validate_pdf_file(file)

    logger.info("[%s] PDF validated — %d bytes", request_id, len(pdf_bytes))

    # ── Build initial state ───────────────────────────────────────────────
    initial_state: ClaimState = {
        "claim_id":         claim_id,
        "pdf_bytes":        pdf_bytes,
        "pdf_pages":        [],
        "classified_pages": [],
        "pages_by_type":    {},
        "patient_info":     None,
        "discharge_summary": None,
        "billing":          None,
        "final_output":     None,
        "processing_errors": [],
    }

    # ── Invoke LangGraph pipeline ─────────────────────────────────────────
    try:
        final_state: ClaimState = claims_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"{claim_id}-{request_id}"}},
        )
    except Exception as exc:
        logger.exception("[%s] Pipeline raised an unexpected exception: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing pipeline failed: {str(exc)}",
        )

    # ── Extract response ──────────────────────────────────────────────────
    final_output: Any = final_state.get("final_output")
    if final_output is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline completed but produced no output. Check server logs.",
        )

    elapsed_ms = round((time.perf_counter() - start_ts) * 1000)
    final_output["_meta"] = {
        "request_id":    request_id,
        "processing_ms": elapsed_ms,
        "pdf_pages":     len(final_state.get("pdf_pages", [])),
        "filename":      file.filename or "unknown.pdf",
    }

    logger.info(
        "[%s] Completed in %dms. Overall confidence=%.2f, errors=%d",
        request_id,
        elapsed_ms,
        final_output.get("quality", {}).get("overall_confidence", 0),
        final_output.get("quality", {}).get("error_count", 0),
    )

    return JSONResponse(content=final_output, status_code=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s: %s", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )