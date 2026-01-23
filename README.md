## 2md

Convert one or more documents/images to Markdown using **Mistral OCR**.

### Use as a library

```python
from pdf2md_cli import ConvertOptions, convert_file, convert_files

# single file
res = convert_file("docs/sample.pdf", api_key="...", outdir="out")
print(res.markdown_path)

# batch (supports globs)
batch = convert_files(["docs/*.pdf", "docs/*.png"], api_key="...", outdir="out", workers=4)
print(len(batch.succeeded), len(batch.failed))
```

### Install (uv)

```bash
uv tool install -e . --reinstall
```

### Usage

Set your API key:

```bash
setx MISTRAL_API_KEY "YOUR_KEY"
```

Run:

```bash
2md path\\to\\file.pdf
2md path\\to\\image.png
2md path\\to\\file.docx
2md path\\to\\slides.pptx
2md docs\\*.pdf --workers 4
2md docs\\*.png --workers 4
2md path\\to\\file.pdf -o out
2md path\\to\\file.pdf --model mistral-ocr-latest
2md path\\to\\file.pdf --keep-remote-file
2md path\\to\\file.pdf                      # default: --table-format html (extract + inline; no tbl-*.html files)
2md path\\to\\file.pdf --table-format markdown
2md path\\to\\file.pdf --extract-table --table-format html
2md path\\to\\file.pdf --extract-table --table-format markdown
2md path\\to\\file.pdf --no-front-matter --no-page-markers
2md docs\\*.pdf --workers 4 --retries 8 --backoff-max-ms 60000
```

Help:

```bash
2md -h
2md help tables
2md help advanced
```

Table handling behavior:

| Flags | Mistral `table_format` | Markdown output | Sidecar files |
| --- | --- | --- | --- |
| *(default)* `--table-format html` | `html` | HTML tables are inlined | none |
| `--table-format markdown` | *(not sent)* | Tables stay inline as markdown | none |
| `--extract-table --table-format html` | `html` | Links to `tbl-*.html` | writes `tbl-*.html` |
| `--extract-table --table-format markdown` | `markdown` | Links to `tbl-*.md` | writes `tbl-*.md` |

Header/footer handling (advanced):

- `--header {inline,discard,extract,comment}` and `--footer {inline,discard,extract,comment}`
  - `comment` (default): extract and add them back as HTML comments in the markdown
  - `inline`: keep headers/footers in the main markdown
  - `discard`: extract headers/footers but drop them
  - `extract`: extract and write `<stem>_headers_footers.md`

Supported document formats:

- PDF (`.pdf`)
- Word (`.docx`)
- PowerPoint (`.pptx`)
- Text (`.txt`)
- EPUB (`.epub`)
- XML / DocBook / JATS XML (`.xml`)
- RTF (`.rtf`)
- OpenDocument Text (`.odt`)
- BibTeX / BibLaTeX (`.bib`)
- FictionBook (`.fb2`)
- Jupyter Notebooks (`.ipynb`)
- LaTeX (`.tex`)
- OPML (`.opml`)
- Troff (`.1`, `.man`)

Supported image formats:

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- AVIF (`.avif`)
- TIFF (`.tif`, `.tiff`)
- GIF (`.gif`)
- HEIC/HEIF (`.heic`, `.heif`)
- BMP (`.bmp`)
- WebP (`.webp`)

Retries and backoff:

- By default the CLI retries transient API failures (e.g. HTTP 429/5xx) with exponential backoff + jitter.
- Tweak with `--retries`, `--backoff-initial-ms`, `--backoff-max-ms`, `--backoff-multiplier`, `--backoff-jitter`.
  (For contributors: set `PDF2MD_ENABLE_MOCK=1` to enable the hidden mock backend for UX testing.)

### Remote file cleanup (after OCR)

For document inputs (e.g. `.pdf`, `.docx`, `.pptx`), this CLI uploads the file to Mistral's Files API with `purpose="ocr"` and **deletes the uploaded file after the OCR call completes** (best-effort).

For image inputs, the CLI sends a `data:` URL to the OCR API (no file upload), so there is no remote file to delete.

If you are calling the API yourself and want to delete an uploaded file explicitly, you can do:

```python
from mistralai import Mistral
import os

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    res = mistral.files.delete(file_id="3b6d45eb-e30b-416f-8019-f47e2e93d930")
    print(res)
```
