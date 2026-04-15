"""
PDF utilities — page-wise extraction using PyMuPDF (fitz).

Responsibilities
----------------
* Open a PDF from raw bytes (never touches disk).
* Extract plain text per page with layout preservation.
* Detect rudimentary table presence (tab/pipe character heuristics).
* Return a list of PageContent dicts ready for LLM classification.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import fitz  # PyMuPDF

from app.graph.state import PageContent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pages(pdf_bytes: bytes) -> list[PageContent]:
    """
    Extract text content from every page of a PDF supplied as raw bytes.

    Returns a list of PageContent in ascending page-number order.
    Raises ValueError if the bytes cannot be opened as a PDF.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    pages: list[PageContent] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = _extract_text(page)
        pages.append(
            PageContent(
                page_number=page_index + 1,   # 1-indexed for human readability
                text=text,
                char_count=len(text),
                has_tables=_has_tables(page, text),
            )
        )
        logger.debug(
            "Extracted page %d — %d chars, tables=%s",
            page_index + 1,
            len(text),
            pages[-1]["has_tables"],
        )

    doc.close()
    logger.info("Extracted %d pages from PDF.", len(pages))
    return pages


def get_pages_as_images(pdf_bytes: bytes, page_numbers: list[int], dpi: int = 150) -> dict[int, bytes]:
    """
    Render specified pages as PNG bytes (1-indexed page numbers).
    Useful when text extraction quality is too low (scanned docs).
    Returns {page_number: png_bytes}.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: dict[int, bytes] = {}
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # scale factor relative to 72 dpi base

    for pnum in page_numbers:
        if pnum < 1 or pnum > len(doc):
            logger.warning("Page %d out of range (total=%d), skipping.", pnum, len(doc))
            continue
        page = doc[pnum - 1]
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        result[pnum] = pix.tobytes("png")

    doc.close()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(page: fitz.Page) -> str:
    """
    Extract text using PyMuPDF's 'text' mode which preserves reading order.
    Falls back to raw dict extraction on failure.
    """
    try:
        # 'text' flag 7 = TEXT_PRESERVE_WHITESPACE | TEXT_PRESERVE_LIGATURES | TEXT_MEDIABOX_CLIP
        text = page.get_text("text", flags=7)
        return text.strip()
    except Exception:
        # Last-resort: concatenate all span texts from the raw dict
        try:
            blocks = page.get_text("dict")["blocks"]
            parts: list[str] = []
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        parts.append(span.get("text", ""))
            return " ".join(parts).strip()
        except Exception as exc:
            logger.error("Text extraction failed for page %s: %s", page.number + 1, exc)
            return ""


def _has_tables(page: fitz.Page, text: str) -> bool:
    """
    Heuristic table detector:
    1. Check for pipe characters (markdown/ASCII tables).
    2. Check for PyMuPDF find_tables result count > 0.
    3. Check for tab-separated columns spanning multiple lines.
    """
    if "|" in text and text.count("|") >= 4:
        return True
    if "\t" in text and text.count("\t") >= 3:
        return True
    try:
        tabs = page.find_tables()
        if tabs and len(tabs.tables) > 0:
            return True
    except Exception:
        pass
    return False


def build_page_context_prompt(page: PageContent, max_chars: int = 3000) -> str:
    """
    Format a single PageContent into a prompt-friendly block.
    Truncates very long pages to keep token costs manageable.
    """
    text = page["text"]
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated — original length {len(page['text'])} chars]"

    return (
        f"=== PAGE {page['page_number']} ===\n"
        f"[char_count={page['char_count']}, has_tables={page['has_tables']}]\n\n"
        f"{text}\n"
    )