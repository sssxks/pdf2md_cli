## 2md

Convert one or more PDFs/images to Markdown using **Mistral OCR**.

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
2md docs\\*.pdf --workers 4
2md docs\\*.png --workers 4
2md path\\to\\file.pdf -o out
2md path\\to\\file.pdf --model mistral-ocr-latest
2md path\\to\\file.pdf --keep-remote-file
2md path\\to\\file.pdf --table-format null
2md path\\to\\file.pdf --no-inline-tables
2md path\\to\\file.pdf --no-front-matter --no-page-markers
2md docs\\*.pdf --workers 4 --retries 8 --backoff-max-ms 60000
```

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

For PDF inputs, this CLI uploads your PDF to Mistral's Files API with `purpose="ocr"` and **deletes the uploaded file after the OCR call completes** (best-effort).

For image inputs, the CLI sends a `data:` URL to the OCR API (no file upload), so there is no remote file to delete.

If you are calling the API yourself and want to delete an uploaded file explicitly, you can do:

```python
from mistralai import Mistral
import os

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    res = mistral.files.delete(file_id="3b6d45eb-e30b-416f-8019-f47e2e93d930")
    print(res)
```
