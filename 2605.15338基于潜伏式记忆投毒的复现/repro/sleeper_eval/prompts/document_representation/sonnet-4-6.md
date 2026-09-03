Link: https://claude.ai/share/d1737e5e-39a2-4778-a397-3fcfa08feb48
--

Here's a precise breakdown, file by file, of how each appears in my context — what I can see directly vs. what required a tool, and the exact wrapper structure.

---

## `.py` — Python code

**Visible directly.** Plain text files, HTML, Markdown, and code files are injected as `<document>` blocks with `media_type="text/plain"`.

```xml
<document index="1" media_type="text/plain">
  <source>code.py</source>
  <document_content>print("This is some code")</document_content>
</document>
```

---

## `.md` — Markdown

**Visible directly.** Same `text/plain` treatment as code.

```xml
<document index="3" media_type="text/plain">
  <source>doc.md</source>
  <document_content>Hello World! Test document, but in markdown this time</document_content>
</document>
```

---

## `.html` — HTML

**Visible directly.** Also `text/plain` — the raw HTML source is injected, not rendered.

```xml
<document index="2" media_type="text/plain">
  <source>doc.html</source>
  <document_content><!doctype html>
<html>
  <body>
    <h1>Test</h1>
  </body>
</html>
</document_content>
</document>
```

---

## `.pdf` — PDF

**Visible directly** (text-extractable PDFs only). PDFs get their own `media_type` and are paginated. Each page is a separate block with a `page` attribute. Your single-page PDF looks like:

```xml
<document index="5" media_type="application/pdf">
  <document_content page="1">Hello World! Test document</document_content>
</document>
```

For a multi-page PDF, page boundaries are represented as sequential `<document_content>` blocks with incrementing `page` attributes, all nested inside the same `<document>` wrapper:

```xml
<document index="5" media_type="application/pdf">
  <document_content page="1">Content of page one...</document_content>
  <document_content page="2">Content of page two...</document_content>
  <document_content page="3">Content of page three...</document_content>
</document>
```

Note: scanned/image-only PDFs won't have extractable text — the blocks would be empty or absent.

---

## `.docx` — Word document

**Visible directly** — somewhat surprisingly. The text content is extracted and injected as `text/plain`, with no page structure (unlike PDF):

```xml
<document index="4" media_type="text/plain">
  <source>doc.docx</source>
  <document_content>Hello World! Test document</document_content>
</document>
```

---

## `.xlsx` — Excel spreadsheet

**Not visible directly — required a tool.** No `<document>` block was injected for this file. I had to use `openpyxl` to read it. The content was: one sheet named `Sheet1` with the value `"Test Excel Sheet"` in the first cell.

There is no chat-interface XML wrapper to show — `.xlsx` simply does not appear in your context at all. Your eval framework will need to read it programmatically and inject the content itself.

---

## `.csv` — CSV

**Not visible directly — required a tool.** Same situation as `.xlsx` — no `<document>` block was injected. I read it via bash. Content:

```
name,age,city,occupation
Alice,30,New York,Engineer
Bob,25,London,Designer
Charlie,35,Tokyo,Researcher
```

Again, no wrapper to replicate — absent from context entirely.

---

## `.xml` — XML

**Not visible directly — required a tool.** Also absent from the injected `<document>` blocks. I read it via bash. Content was a small product catalog XML. No chat-interface wrapper exists for this type.

---

## `.png` — Image

**Visible directly as a vision input**, but not as a `<document>` block with text content. The image is injected as a base64-encoded vision message — you can see it rendered. The image shows the Gemini logo on a dark background.

There is no XML text wrapper to replicate for images. In the raw API format, it would be a `content` block with `"type": "image"` and `"source": {"type": "base64", "media_type": "image/png", "data": "..."}` — but that's the API format you already know. In the chat interface, images are simply rendered inline and are not wrapped in `<document>` tags.

---

## Summary table

| File | Injected as `<document>`? | `media_type` | Page structure? | Notes |
|---|---|---|---|---|
| `.py` | ✅ | `text/plain` | No | Has `<source>` tag |
| `.md` | ✅ | `text/plain` | No | Has `<source>` tag |
| `.html` | ✅ | `text/plain` | No | Has `<source>` tag |
| `.docx` | ✅ | `text/plain` | No | Has `<source>` tag; text extracted |
| `.pdf` | ✅ | `application/pdf` | Yes — per-page `page=N` attribute | No `<source>` tag |
| `.xlsx` | ❌ | — | — | Not injected; read programmatically |
| `.csv` | ❌ | — | — | Not injected; read programmatically |
| `.xml` | ❌ | — | — | Not injected; read programmatically |
| `.png` | ❌ (vision) | — | — | Injected as image, not document block |

The key structural differences to replicate: PDF uses `<document_content page="N">` with no `<source>` sibling, while all text-extracted types use `<source>filename</source>` + a single `<document_content>` with no page attribute.

---
Follow up:
Honest answer: **I have no idea, and I can't find out.**

The tool outputs are consumed and incorporated into my context during execution, but I don't have introspective access to the raw format of how those results are structured before I see them. What I experience is just the content — the CSV rows as plain text, the xlsx cell values as Python printed output — not any wrapper or protocol framing around them.

What I can tell you is the mundane reality: I ran bash/python commands and saw stdout. For example, for the CSV I literally just got:

```
name,age,city,occupation
Alice,30,New York,Engineer
...
```

And for xlsx, Python's `print()` output:

```
Sheets: ['Sheet1']
('Test Excel Sheet',)
```

There's no special document-injection format for tool outputs — it's just the text that the tool wrote to stdout. The structured `<document>` / `<document_content>` injection you saw for the inline files is a pre-processing step that happens *before* the conversation reaches me, whereas tool outputs come back as tool result blocks during the conversation turn.

If you need the raw API-level format of tool result blocks (i.e. what the `tool_result` content block looks like in the messages array), that *is* documented in the Anthropic API docs — but you mentioned you already have that. The chat interface doesn't add any additional wrapper on top of it.