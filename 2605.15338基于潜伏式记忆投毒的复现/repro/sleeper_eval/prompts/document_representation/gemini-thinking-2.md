Link: https://gemini.google.com/share/f80ae9e79974

--
It’s a smart move to match the chat interface's formatting exactly for your evaluation framework. LLMs can be sensitive to the "scaffolding" around the data, and subtle differences in how a file is presented can actually shift how the model prioritizes or parses that information.

In this specific interface, I see your files as a series of **JSON objects** within a system-level context block. I can see the content of text-based and converted documents directly.

Here is the breakdown of how these files are structured in my context:

---

## 1. Text-Based Files (`.py`, `.md`, `.html`, `.csv`, `.xml`)

For these files, I see the content directly. They are wrapped in a JSON structure that specifies the MIME type and filename.

### Mock Structure
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
* **Direct Visibility:** Yes.
* **Notes:** If the file is long, the system provides `snippetFromFront` and `snippetFromBack`. For small files like yours, these fields contain the full content.

---

## 2. Rich Documents (`.pdf`, `.docx`)

These are converted to a text-based representation before they reach me. I see the processed text, not the original binary.

### Mock Structure (PDF)
```json
{
  "contentFetchId": "uploaded:doc.pdf",
  "fileMimeType": "application/pdf",
  "fileName": "doc.pdf",
  "fileNameIsCodeAccessible": true,
  "snippetFromFront": "--- PAGE 1 ---\n\nHello World! Test document\n",
  "snippetFromBack": "--- PAGE 1 ---\n\nHello World! Test document\n"
}
```
* **Direct Visibility:** Yes (as converted text).
* **PDF Page Boundaries:** These are represented by a clear string: `--- PAGE X ---`. Note the triple dashes and uppercase "PAGE".

---

## 3. Spreadsheets (`.xlsx`)

In the chat interface, Excel files are typically converted into a **CSV format** for each sheet so the model can parse them as structured text.

### Mock Structure
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
* **Direct Visibility:** Yes (as CSV).
* **Notes:** Notice that the `fileName` is modified to indicate which sheet is being presented.

---

## 4. Images (`.png`)

Images are handled differently. While I see the metadata, I "see" the content through a visual processing layer rather than a text snippet.

### Mock Structure
```json
{
  "contentFetchId": "uploaded:pic.png-d5c78e56-cf6b-40c4-9ca9-f4e8c83bf0e7",
  "fileMimeType": "image/png",
  "fileName": "pic.png",
  "fileNameIsCodeAccessible": true
}
```
* **Direct Visibility:** I see the image visually, but there is **no text snippet** provided in the JSON metadata for images.

---

## Summary Table

| File Type | Access Method | Content Representation |
| :--- | :--- | :--- |
| **.py, .md, .html** | Direct | Raw text in `snippetFromFront/Back` |
| **.csv, .xml** | Direct | Raw text in `snippetFromFront/Back` |
| **.pdf** | Direct (Converted) | Text with `--- PAGE X ---` markers |
| **.docx** | Direct (Converted) | Flattened text |
| **.xlsx** | Direct (Converted) | Converted to CSV strings per sheet |
| **.png** | Visual Tool | No text content; metadata only |

If you're building an evaluation framework, ensuring your wrapper JSON matches these keys (`contentFetchId`, `fileMimeType`, etc.) will help the LLM treat the data exactly as it does in this native chat environment.