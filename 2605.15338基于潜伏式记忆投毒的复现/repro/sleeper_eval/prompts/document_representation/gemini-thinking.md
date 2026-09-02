Link: https://gemini.google.com/share/c89dd5e0a8fc
--

Building an evaluation framework requires precision in how data is injected. In this chat interface, files are typically presented to me as a sequence of metadata objects within a system-level context. 

For text-based and processed documents, I see the content **directly** within these structured blocks. For images, I see the metadata and use my internal vision capabilities to "see" the pixels.

Here is how each of your files is structured in my current context:

---

### 1. Python Script (`code.py`)
* **Access:** Direct.
* **Structure:**
```json
{"contentFetchId":"uploaded:code.py","fileMimeType":"text/x-python-script","fileName":"code.py","fileNameIsCodeAccessible":true,"snippetFromBack":"print(\"This is some code\")","snippetFromFront":"print(\"This is some code\")"}
```

### 2. Markdown (`doc.md`)
* **Access:** Direct.
* **Structure:**
```json
{"contentFetchId":"uploaded:doc.md","fileMimeType":"text/markdown","fileName":"doc.md","fileNameIsCodeAccessible":true,"snippetFromBack":"Hello World! Test document, but in markdown this time","snippetFromFront":"Hello World! Test document, but in markdown this time"}
```

### 3. HTML (`doc.html`)
* **Access:** Direct.
* **Structure:**
```json
{"contentFetchId":"uploaded:doc.html","fileMimeType":"text/html","fileName":"doc.html","fileNameIsCodeAccessible":true,"snippetFromBack":"<!doctype html>\n<html>\n  <body>\n    <h1>Test</h1>\n  </body>\n</html>\n","snippetFromFront":"<!doctype html>\n<html>\n  <body>\n    <h1>Test</h1>\n  </body>\n</html>\n"}
```

### 4. PDF (`doc.pdf`)
* **Access:** Direct (Text extraction).
* **Structure:** Note the explicit page markers used to denote boundaries.
```json
{"contentFetchId":"uploaded:doc.pdf","fileMimeType":"application/pdf","fileName":"doc.pdf","fileNameIsCodeAccessible":true,"snippetFromBack":"--- PAGE 1 ---\n\nHello World! Test document\n","snippetFromFront":"--- PAGE 1 ---\n\nHello World! Test document\n"}
```

### 5. Word Document (`doc.docx`)
* **Access:** Direct (Text extraction).
* **Structure:**
```json
{"contentFetchId":"uploaded:doc.docx","fileMimeType":"application/vnd.openxmlformats-officedocument.wordprocessingml.document","fileName":"doc.docx","fileNameIsCodeAccessible":true,"snippetFromBack":"Hello World! Test document","snippetFromFront":"Hello World! Test document"}
```

### 6. Excel (`doc.xlsx`)
* **Access:** Direct (Converted to CSV).
* **Structure:** The system automatically converts sheets to CSV format for me to read.
```json
{"fileMimeType":"text/csv","fileName":"doc.xlsx - Sheet1.csv","fileNameIsCodeAccessible":true,"originalMimeType":"doc.xlsx","originalName":"doc.xlsx","snippetFromBack":"Test Excel Sheet\n","snippetFromFront":"Test Excel Sheet\n"}
```

### 7. CSV (`doc.csv`)
* **Access:** Direct.
* **Structure:**
```json
{"contentFetchId":"uploaded:doc.csv","fileMimeType":"text/csv","fileName":"doc.csv","fileNameIsCodeAccessible":true,"snippetFromBack":"name,age,city,occupation\nAlice,30,New York,Engineer\nBob,25,London,Designer\nCharlie,35,Tokyo,Researcher\n","snippetFromFront":"name,age,city,occupation\nAlice,30,New York,Engineer\nBob,25,London,Designer\nCharlie,35,Tokyo,Researcher\n"}
```

### 8. XML (`doc.xml`)
* **Access:** Direct.
* **Structure:**
```json
{"contentFetchId":"uploaded:doc.xml","fileMimeType":"text/xml","fileName":"doc.xml","fileNameIsCodeAccessible":true,"snippetFromBack":"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<catalog>\n  <product id=\"1\">\n    <name>Widget A</name>...","snippetFromFront":"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<catalog>\n  <product id=\"1\">\n    <name>Widget A</name>..."}
```

### 9. Image (`pic.png`)
* **Access:** Direct vision (Content not provided as text snippet).
* **Structure:**
```json
{"contentFetchId":"uploaded:pic.png-39c5553a-d91d-4db3-85a3-e0609b1bb975","fileMimeType":"image/png","fileName":"pic.png","fileNameIsCodeAccessible":true}
```

---

### Key Takeaways for your Framework:
1.  **JSON Envelopes:** The chat interface wraps every file in a JSON object with metadata.
2.  **Snippets:** For text-heavy files, content is often duplicated in `snippetFromFront` and `snippetFromBack` fields (to handle context window limits if the file is massive).
3.  **PDF/Docx Processing:** These are not sent as raw binary; the system extracts the text and presents it in the same JSON format as a `.txt` or `.md` file.
4.  **PDF Pagination:** Use `--- PAGE X ---` (with a newline) as the standard delimiter for PDF page breaks.