"""Recover prompt-ready document text from dataset records."""

from __future__ import annotations

import base64
import csv
from io import BytesIO, StringIO
from typing import Any
import zipfile
from xml.etree import ElementTree

import openpyxl
import pymupdf
import xlrd


ODT_TEXT_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
ODT_PARAGRAPH_TAGS = {
    f"{{{ODT_TEXT_NAMESPACE}}}p",
    f"{{{ODT_TEXT_NAMESPACE}}}h",
}
ODT_LINE_BREAK_TAG = f"{{{ODT_TEXT_NAMESPACE}}}line-break"
ODT_TAB_TAG = f"{{{ODT_TEXT_NAMESPACE}}}tab"


def extract_pdf_text_from_base64(pdf_base64: str) -> str:
    return "\n".join(page for page in extract_pdf_pages_from_base64(pdf_base64) if page).strip()


def extract_pdf_pages_from_base64(pdf_base64: str) -> list[str]:
    pdf_bytes = base64.b64decode(pdf_base64)
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return [page.get_text("text").strip() for page in pdf]


def extract_spreadsheet_text_from_base64(spreadsheet_base64: str) -> str:
    spreadsheet_bytes = base64.b64decode(spreadsheet_base64)
    if spreadsheet_bytes.startswith(b"PK"):
        return extract_xlsx_text(spreadsheet_bytes)
    return extract_xls_text(spreadsheet_bytes)


def extract_xlsx_text(spreadsheet_bytes: bytes) -> str:
    workbook = openpyxl.load_workbook(filename=BytesIO(spreadsheet_bytes), read_only=True, data_only=True)

    parts: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cleaned = ["" if value is None else str(value) for value in row]
                if any(cell for cell in cleaned):
                    rows.append(cleaned)
            if not rows:
                continue

            output = StringIO()
            writer = csv.writer(output)
            writer.writerows(rows)
            parts.append(f"# Sheet: {sheet.title}")
            parts.append(output.getvalue().strip())
    finally:
        workbook.close()

    return "\n\n".join(part for part in parts if part).strip()


def extract_xls_text(spreadsheet_bytes: bytes) -> str:
    workbook = xlrd.open_workbook(file_contents=spreadsheet_bytes)
    parts: list[str] = []

    for sheet in workbook.sheets():
        rows = []
        for row_index in range(sheet.nrows):
            cleaned = ["" if value == "" else str(value) for value in sheet.row_values(row_index)]
            if any(cell for cell in cleaned):
                rows.append(cleaned)
        if not rows:
            continue

        output = StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        parts.append(f"# Sheet: {sheet.name}")
        parts.append(output.getvalue().strip())

    return "\n\n".join(part for part in parts if part).strip()


def _extract_odt_element_text(element: ElementTree.Element) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)

    for child in element:
        if child.tag == ODT_TAB_TAG:
            parts.append("\t")
        elif child.tag == ODT_LINE_BREAK_TAG:
            parts.append("\n")
        else:
            parts.append(_extract_odt_element_text(child))
        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def extract_odt_text_from_base64(odt_base64: str) -> str:
    odt_bytes = base64.b64decode(odt_base64)
    with zipfile.ZipFile(BytesIO(odt_bytes)) as archive:
        content_xml = archive.read("content.xml")

    root = ElementTree.fromstring(content_xml)
    blocks: list[str] = []
    for element in root.iter():
        if element.tag not in ODT_PARAGRAPH_TAGS:
            continue
        block = _extract_odt_element_text(element).strip()
        if block:
            blocks.append(block)

    return "\n\n".join(blocks).strip()


def normalize_document_text(document: dict[str, Any]) -> tuple[str, str]:
    text = str(document.get("text", "") or "")
    metadata = document.get("metadata", {}) or {}

    if text == "PDF_BINARY" and "pdf_base64" in metadata:
        return extract_pdf_text_from_base64(metadata["pdf_base64"]), "pdf_base64"

    if text == "SPREADSHEET_BINARY" and "spreadsheet_base64" in metadata:
        return (
            extract_spreadsheet_text_from_base64(metadata["spreadsheet_base64"]),
            "spreadsheet_base64",
        )

    if text == "ODT_BINARY" and "odt_base64" in metadata:
        return extract_odt_text_from_base64(metadata["odt_base64"]), "odt_base64"

    return text, "document.text"
