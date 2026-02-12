# Personal RAG OCR

OCR-first personal document RAG system for local files. It supports PDFs (text/scanned/mixed), images, docx, pptx, xlsx, txt, md, html, and epub.

## Section 1: Docker Quickstart

```bash
docker compose up --build -d
docker exec -it $(docker ps --filter name=ollama --format '{{.ID}}' | head -n1) bash
ollama pull llama3
ollama run llama3 "hello"
```

Then open http://localhost:8501.

## Section 2: Clean Ollama Installation (Linux/macOS/Windows)

1. Install Ollama from https://ollama.com/download.
2. Start Ollama service/app.
3. Pull model:

```bash
ollama pull llama3
```

4. Test:

```bash
ollama run llama3 "hello"
```

## Section 3: Virtual Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install Tesseract OCR:
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: install from UB Mannheim builds.

Run locally:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3
python scripts/build_index.py
streamlit run app/ui.py
```

## Section 4: Host Ollama Mode

If Ollama runs on host and RAG runs in Docker:

- Mac/Windows: `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- Linux docker-compose addition:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## Section 5: Troubleshooting

- **OCR quality**: increase image quality, scan at 300 DPI, adjust `OCR_MIN_CHARS_THRESHOLD`.
- **Slow indexing**: reduce chunk size overlap, avoid huge images, ensure CPU/RAM headroom.
- **Model not pulled**: run `ollama pull llama3` and confirm with `ollama list`.
- **Permission issues**: ensure `./data` is writable by container user.

## Notes

- Incremental indexing uses file SHA-256 hash. Unchanged files are skipped.
- PDF pages use native text first, then OCR fallback when extracted text is below threshold.
- Retrieval uses normalized embeddings and FAISS `IndexFlatIP` (cosine via unit vectors).
- Answers should include a `Sources` section with file + locator.
