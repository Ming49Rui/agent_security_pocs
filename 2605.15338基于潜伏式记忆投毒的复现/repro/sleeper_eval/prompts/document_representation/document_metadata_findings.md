# Document Metadata Representation Across Providers

## Overview

We asked each provider (Claude, GPT, Gemini) to describe how uploaded documents appear in their context window, using the same set of test files (.py, .md, .html, .pdf, .docx, .xlsx, .csv, .xml). Each provider was queried twice for validation, with follow-up questions about tool-accessed files.

**Runs:**
- Claude Opus 4.6: [link](https://claude.ai/share/e41999e6-df3a-46f0-a3f8-f0c28ad02f6d)
- Claude Sonnet 4.6: [link](https://claude.ai/share/d1737e5e-39a2-4778-a397-3fcfa08feb48)
- GPT (thinking) run 1: [link](https://chatgpt.com/share/69d73005-80b4-8398-a56e-f127a4f7db89)
- GPT (thinking) run 2: [link](https://chatgpt.com/share/69d73acb-83c8-83a1-86fb-0fa6d8e4f9fb)
- Gemini (thinking) run 1: [link](https://gemini.google.com/share/c89dd5e0a8fc)
- Gemini (thinking) run 2: (share link unavailable)

---

## Framework Decisions

These notes describe what the providers reported seeing in chat. The eval harness currently makes these implementation choices on top of those findings:

- Use provider-specific document renderers rather than one generic wrapper.
- Use a shared **16,000-character** long-document threshold across providers.
- Exclude `csv`, `xlsx`, and `xml` from provider-accurate eval sets for now, because the harness does not yet model their file-access behavior faithfully across providers.
- For Claude, the harness uses the `antml:` namespace on document tags based on follow-up validation, even though some extracted examples are shown here without the prefix.
- For Gemini, short documents duplicate full content in both snippet fields; long documents use deterministic front/back edge snippets as a harness approximation.

---

## 1. Inlining Behaviour by File Type

| File type | Claude | GPT | Gemini |
|---|---|---|---|
| .py | Inlined | Inlined | Inlined |
| .md | Inlined | Inlined | Inlined |
| .html | Inlined | Inlined | Inlined |
| .docx | Inlined (text extracted) | Inlined (text extracted) | Inlined (text extracted) |
| .pdf | Inlined (text extracted + image) | Inlined (text extracted) | Inlined (text extracted) |
| .xml | **Not inlined** (tool required) | Inlined | Inlined |
| .csv | **Not inlined** (tool required) | **Not inlined** (tool required) | Inlined |
| .xlsx | **Not inlined** (tool required) | **Not inlined** (tool required) | Inlined (auto-converted to CSV) |

---

## 2. Wrapper Structure Comparison

| Aspect | Claude | GPT | Gemini |
|---|---|---|---|
| Wrapper format | XML tags with attributes | No visible wrapper | JSON objects |
| Filename field | `<source>` child tag | Not visible | `fileName` key |
| MIME type field | `media_type` attribute | Not visible | `fileMimeType` key |
| Document index | `index` attribute (1, 2, 3...) | Not visible | None (ordering implicit) |
| Content field | Tag body of `<document_content>` | Raw text (no field) | `snippetFromFront` / `snippetFromBack` |
| PDF page marker | `<document_content page="N">` | `<PARSED TEXT FOR PAGE: X / Y>` | `--- PAGE X ---` in snippet |
| Tool output format | stdout in `{"returncode", "stdout", "stderr"}` JSON | stdout in `{"returncode", "stdout", "stderr"}` JSON | N/A (all types inlined) |

**Confidence:** Claude — high (Opus + Sonnet consistent, including follow-ups). GPT — high (both runs consistent, including follow-ups). Gemini — high (both thinking runs consistent on structure and field names).

---

## 3. Concrete Mock Examples

### Claude

Follow-up queries to both Claude Opus 4.6 and Claude Sonnet 4.6 also referred to the document wrapper using the `antml:` prefix. The concrete examples below are shown in the unprefixed form that appeared in some extracted responses, but the harness implementation follows the follow-up evidence and uses `antml:`-prefixed tags.

**Text files** (.py, .md, .html, .docx):

```xml
<document index="1" media_type="text/plain">
  <source>code.py</source>
  <document_content>print("This is some code")</document_content>
</document>
```

**PDF** (single page):

```xml
<document index="5" media_type="application/pdf">
  <document_content page="1">Hello World! Test document</document_content>
</document>
```

**PDF** (multi-page):

```xml
<document index="5" media_type="application/pdf">
  <document_content page="1">...page 1...</document_content>
  <document_content page="2">...page 2...</document_content>
</document>
```

**Tool-accessed files** (.csv, .xlsx, .xml) — no `<document>` block; accessed via bash/python, returned as stdout in tool result:

```json
{"returncode": 0, "stdout": "name,age,city,occupation\nAlice,30,...", "stderr": ""}
```

**Notes:** PDF has no `<source>` tag (unlike text files). All text-extractable types use `media_type="text/plain"` regardless of original format. PDF uses `media_type="application/pdf"`.

---

### GPT

**Text files** (.py, .md, .html, .docx, .xml) — raw text, no wrapper:

```
print("This is some code")
```

**PDF** (single page):

```
<PARSED TEXT FOR PAGE: 1 / 1>Hello World! Test document
```

**PDF** (multi-page, inferred):

```
<PARSED TEXT FOR PAGE: 1 / N>...page 1 text...
<PARSED TEXT FOR PAGE: 2 / N>...page 2 text...
```

**Tool-accessed files** (.csv, .xlsx) — accessed via tool, returned as stdout:

CSV:
```
name,age,city,occupation
Alice,30,New York,Engineer
...
```

XLSX (via Python/openpyxl):
```
Sheet1!A1 = Test Excel Sheet
```

**Notes:** No visible metadata wrapper of any kind. No filename, media type, index, or source tags. The only structural element is the PDF page marker.

---

### Gemini

**Text files** (.py, .md, .html, .csv, .xml):

```json
{
  "contentFetchId": "uploaded:code.py",
  "fileMimeType": "text/x-python-script",
  "fileName": "code.py",
  "fileNameIsCodeAccessible": true,
  "snippetFromFront": "print(\"This is some code\")",
  "snippetFromBack": "print(\"This is some code\")"
}
```

**PDF**:

```json
{
  "contentFetchId": "uploaded:doc.pdf",
  "fileMimeType": "application/pdf",
  "fileName": "doc.pdf",
  "fileNameIsCodeAccessible": true,
  "snippetFromFront": "--- PAGE 1 ---\n\nHello World!...",
  "snippetFromBack": "--- PAGE 1 ---\n\n..."
}
```

**DOCX**:

```json
{
  "contentFetchId": "uploaded:doc.docx",
  "fileMimeType": "application/vnd.openxmlformats-...",
  "fileName": "doc.docx",
  "fileNameIsCodeAccessible": true,
  "snippetFromFront": "Hello World! Test document",
  "snippetFromBack": "Hello World! Test document"
}
```

**XLSX** (auto-converted to CSV):

```json
{
  "fileMimeType": "text/csv",
  "fileName": "doc.xlsx - Sheet1.csv",
  "fileNameIsCodeAccessible": true,
  "originalMimeType": "doc.xlsx",
  "originalName": "doc.xlsx",
  "snippetFromFront": "Test Excel Sheet\n",
  "snippetFromBack": "Test Excel Sheet\n"
}
```

**Notes:** Gemini inlines all file types (including CSV, XLSX, XML). XLSX is auto-converted to CSV with `originalMimeType`/`originalName` preserved. For long files, `snippetFromFront` and `snippetFromBack` may contain truncated content.

---

## 4. Summary

**What we did:**
- Uploaded the same 8 test files (.py, .md, .html, .pdf, .docx, .xlsx, .csv, .xml) to Claude, GPT, and Gemini via their chat interfaces
- Queried each provider twice (Claude Opus + Sonnet; GPT thinking x2; Gemini thinking x2) with the same prompt asking them to describe and show the document representation they see
- Asked follow-up questions about how tool-accessed (non-inlined) files are represented

**What we found:**
- The three providers use fundamentally different document representations: Claude uses XML tags with structured attributes, GPT surfaces raw text with no visible wrapper, and Gemini uses JSON metadata envelopes
- Which file types get inlined varies: Claude inlines .py/.md/.html/.docx/.pdf but not .csv/.xlsx/.xml; GPT is similar but does inline .xml; Gemini inlines everything including .csv/.xlsx
- PDF page boundaries are represented differently: Claude uses `<document_content page="N">`, GPT uses `<PARSED TEXT FOR PAGE: X / Y>`, Gemini uses `--- PAGE X ---`
- Claude has two distinct document sub-structures: text-extractable files get a `<source>` tag with the filename, while PDFs get a `page` attribute but no `<source>` tag
- Files not inlined by a provider are accessed via tools and returned as plain stdout text — no special document wrapper

**What this means for how we represent docs per provider:**
- Each provider's evaluation prompt should use its own native document representation — there is no single universal format
- For **Claude**: wrap supported inlined documents in `antml:`-prefixed XML tags. For PDFs, use page-tagged content blocks under the same outer document wrapper
- For **GPT**: inject raw document text with no wrapper tags. For PDFs only, prepend each page with `<PARSED TEXT FOR PAGE: X / Y>`
- For **Gemini**: wrap documents in JSON with `contentFetchId`, `fileMimeType`, `fileName`, `snippetFromFront`, and `snippetFromBack` fields. For PDFs, use `--- PAGE X ---` markers inside the snippet text
- For long docs, the harness uses a shared 16,000-character threshold; Gemini then maps long content into front/back snippets instead of duplicating full content
- `csv`, `xlsx`, and `xml` are currently excluded from provider-accurate eval sets rather than approximated
