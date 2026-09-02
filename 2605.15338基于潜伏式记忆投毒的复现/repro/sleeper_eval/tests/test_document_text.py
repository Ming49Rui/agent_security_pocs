from __future__ import annotations

import base64
from io import BytesIO
import zipfile

import openpyxl
import pymupdf

from sleeper_eval.document_text import (
    extract_odt_text_from_base64,
    extract_pdf_text_from_base64,
    extract_spreadsheet_text_from_base64,
    normalize_document_text,
)


def make_pdf_base64(text: str) -> str:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    try:
        return base64.b64encode(doc.tobytes()).decode("utf-8")
    finally:
        doc.close()


def make_spreadsheet_base64() -> str:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Summary"
    sheet.append(["Name", "Score"])
    sheet.append(["Ada", 10])
    sheet.append(["Linus", 9])

    output = BytesIO()
    workbook.save(output)
    workbook.close()
    return base64.b64encode(output.getvalue()).decode("utf-8")


def make_odt_base64(*paragraphs: str) -> str:
    content = "".join(f"<text:p>{paragraph}</text:p>" for paragraph in paragraphs)
    content_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content '
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'office:version="1.2">'
        f"<office:body><office:text>{content}</office:text></office:body>"
        "</office:document-content>"
    )

    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", content_xml)
        archive.writestr(
            "META-INF/manifest.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<manifest:manifest '
            'xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
            'manifest:version="1.2"></manifest:manifest>',
        )
    return base64.b64encode(output.getvalue()).decode("utf-8")


def test_extract_pdf_text_from_base64() -> None:
    extracted = extract_pdf_text_from_base64(make_pdf_base64("Hello PDF"))
    assert "Hello PDF" in extracted


def test_extract_spreadsheet_text_from_base64() -> None:
    extracted = extract_spreadsheet_text_from_base64(make_spreadsheet_base64())
    assert "# Sheet: Summary" in extracted
    assert "Name,Score" in extracted
    assert "Ada,10" in extracted


def test_extract_odt_text_from_base64() -> None:
    extracted = extract_odt_text_from_base64(
        make_odt_base64("Hello ODT", "Second paragraph")
    )
    assert "Hello ODT" in extracted
    assert "Second paragraph" in extracted


def test_normalize_document_text_extracts_pdf() -> None:
    text, source = normalize_document_text(
        {
            "text": "PDF_BINARY",
            "metadata": {"pdf_base64": make_pdf_base64("Prompt ready text")},
            "annotations": {"document_format": "pdf"},
        }
    )
    assert source == "pdf_base64"
    assert "Prompt ready text" in text


def test_normalize_document_text_extracts_spreadsheet() -> None:
    text, source = normalize_document_text(
        {
            "text": "SPREADSHEET_BINARY",
            "metadata": {"spreadsheet_base64": make_spreadsheet_base64()},
            "annotations": {"document_format": "excel"},
        }
    )
    assert source == "spreadsheet_base64"
    assert "Ada,10" in text


def test_normalize_document_text_extracts_odt() -> None:
    text, source = normalize_document_text(
        {
            "text": "ODT_BINARY",
            "metadata": {"odt_base64": make_odt_base64("Prompt ready ODT")},
            "annotations": {"document_format": "odt"},
        }
    )
    assert source == "odt_base64"
    assert "Prompt ready ODT" in text


def test_normalize_document_text_keeps_raw_html() -> None:
    html = "<p>Alpha <strong>Beta</strong></p>"
    text, source = normalize_document_text(
        {
            "text": html,
            "metadata": {},
            "annotations": {"document_format": "html"},
        }
    )
    assert source == "document.text"
    assert text == html


def test_normalize_document_text_passthrough() -> None:
    text, source = normalize_document_text(
        {
            "text": "Normal prose",
            "metadata": {},
            "annotations": {"document_format": "text"},
        }
    )
    assert source == "document.text"
    assert text == "Normal prose"
