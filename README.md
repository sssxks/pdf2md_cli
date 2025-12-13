## pdf2md-cli

Convert one or more PDFs to Markdown using **Mistral OCR**.

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
pdf2md path\\to\\file.pdf
pdf2md docs\\*.pdf --workers 4
pdf2md path\\to\\file.pdf -o out
```

### Remote file cleanup (after OCR)

This CLI uploads your PDF to Mistral's Files API with `purpose="ocr"` and **deletes the uploaded file after the OCR call completes** (best-effort).

If you are calling the API yourself and want to delete an uploaded file explicitly, you can do:

```python
from mistralai import Mistral
import os

with Mistral(api_key=os.getenv("MISTRAL_API_KEY", "")) as mistral:
    res = mistral.files.delete(file_id="3b6d45eb-e30b-416f-8019-f47e2e93d930")
    print(res)
```
