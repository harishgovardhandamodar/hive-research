from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SECTION_HEADING = re.compile(
    r"^(#{1,3}\s+|\d+\.\d*\s+|[A-Z][A-Z\s]{2,}(?:\n|$))",
    re.MULTILINE,
)

REFERENCE_PATTERN = re.compile(
    r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.DOTALL
)

ARXIV_REF_PATTERN = re.compile(r"(\d{4}\.\d{4,5})")


def extract_text(pdf_path: str | Path) -> str:
    import fitz

    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def extract_metadata(pdf_path: str | Path) -> dict[str, Any]:
    import fitz

    doc = fitz.open(str(pdf_path))
    meta = doc.metadata or {}
    doc.close()
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
    }


def extract_references(text: str) -> list[dict[str, str]]:
    refs = []
    for match in REFERENCE_PATTERN.finditer(text):
        refs.append({"num": match.group(1), "text": match.group(2).strip()})
    return refs


def extract_referenced_arxiv_ids(text: str) -> list[str]:
    return list(set(ARXIV_REF_PATTERN.findall(text)))


def extract_sections(text: str) -> list[dict[str, Any]]:
    lines = text.split("\n")
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] = {"heading": "abstract", "content": []}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_HEADING.match(stripped):
            if current["content"]:
                sections.append(current)
            current = {"heading": stripped, "content": []}
        else:
            current["content"].append(stripped)
    if current["content"]:
        sections.append(current)
    for s in sections:
        s["content"] = "\n".join(s["content"])
    return sections
