# ResearchPaper AI

**Read research papers with questions, answers, and page-level citations.** ResearchPaper AI is a full-stack document research assistant. Upload a text-based PDF, ask a question, and inspect the passages that support each answer.

This repository is designed as a portfolio project: it includes a working browser app, a typed API, PDF ingestion and retrieval, per-browser data isolation, a no-key sample mode, and deployment notes.

## What it does

- Upload and remove research-paper PDFs, with file type and size validation.
- Extract text by page, OCR image-only pages with Gemini Vision, split into overlapping chunks, and cite source pages in answers.
- Use Gemini embeddings and structured/streaming Gemini answers when a server-side `GEMINI_API_KEY` is configured.
- Run local keyword search and extractive answers without an API key.
- Try the bundled sample paper without configuring credentials.
- Stream answers in the browser and keep conversation history for the current browser session.
- Keep documents, vector indexes, and conversations scoped to a random HTTP-only browser session.
- Search all session documents or limit a question to one selected PDF.
- Use the responsive dark/light interface and OpenAPI documentation.

## Architecture

```text
Browser (HTML/CSS/JavaScript)
        │ same-origin HTTP + HTTP-only session cookie
        ▼
FastAPI routes ── session-scoped services ── SQLite metadata/history
        │                                      ├─ per-session PDF files
        │                                      └─ per-session FAISS or keyword index
        ├─ pypdf extraction + Gemini Vision OCR → overlapping text chunks
        ├─ Gemini embeddings + structured/streaming generation (optional)
        └─ local keyword retrieval + extractive responses (no key)
```

The API key is read by server-side settings only. It is never requested by the website or returned by the health endpoint. For OCR, image-only pages are rendered and sent to Gemini; text extraction and ordinary PDFs stay local until you ask a question. Browser session IDs are high-entropy UUIDs stored in an HTTP-only, SameSite=Lax cookie. API reads and writes are filtered by that ID.

## Run locally

Requirements: Python 3.12 or newer.

```bash
git clone https://github.com/patelpreet404-alt/researchpaper-ai.git
cd researchpaper-ai
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000). Choose **Try the sample paper** and ask a question to verify the no-key path. Uploads are limited to 25 MB by default. Scanned pages can be OCR'd with a Gemini key, up to `MAX_OCR_PAGES` pages per PDF (30 by default).

To enable generated answers, semantic retrieval, and OCR, add a Gemini API key to your private `.env` file:

```env
GEMINI_API_KEY=your-key
```

The key stays on the server. Never commit `.env` or paste credentials into chat. The existing `.gitignore` excludes `.env` and user-generated upload/index data.

## Configuration

See [.env.example](.env.example) for the full list. Important settings:

| Setting | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Enables embeddings, generated answers, and OCR; leave blank for local mode | empty |
| `GEMINI_CHAT_MODEL` | Gemini generation and OCR model | `gemini-3.8-flash` |
| `GEMINI_EMBEDDING_MODEL` | Embedding model | `gemini-embedding-001` |
| `MAX_OCR_PAGES` | Maximum image-only pages sent for OCR per PDF | `30` |
| `MAX_UPLOAD_MB` | Maximum PDF size in local/container deployments | `25` |
| `DATABASE_URL` | SQLAlchemy database URL | local SQLite |
| `UPLOAD_DIR` | Uploaded PDF storage root | `./data/uploads` |
| `VECTOR_STORE_DIR` | FAISS/keyword index root | `./data/vector_store` |

## API

- `GET /api/health` — readiness and whether server-side AI credentials are configured.
- `POST /api/documents` — upload and index one PDF.
- `POST /api/documents/sample` — add the bundled sample PDF to the current browser session.
- `GET /api/documents` / `DELETE /api/documents/{id}` — list or remove session documents.
- `POST /api/chat/stream` — stream a grounded answer with source metadata.
- `GET /api/conversations` / `GET /api/conversations/{id}` — session conversation history.
- `GET /api/docs` — interactive Swagger UI.

## Checks

```bash
pytest tests -v
ruff check app tests
mypy app
```

The integration tests avoid paid model calls. For a browser check, start the app and confirm that two independent browser profiles have separate libraries and histories, then add the sample paper and ask a question in one profile.

## Vercel preview setup

The root `index.py` exposes the FastAPI website to Vercel. Connect the GitHub repository to a Vercel **Preview** project and set `APP_ENV=production` so cookies use HTTPS. Add `GEMINI_API_KEY` in Vercel Project Settings to enable AI answers and OCR; the local sample mode does not need it.

Vercel Functions have a read-only filesystem except for temporary `/tmp` scratch space, and cap request and response bodies at 4.5 MB. The app therefore uses `/tmp` and a 3 MB PDF limit on Vercel. That storage is temporary and can differ between function instances: use Vercel deployment for reviewing the interface and short demos only. Durable multi-user use needs managed persistence for the database, uploaded files, and vector index before enabling production traffic. See Vercel's [Python runtime](https://vercel.com/docs/functions/runtimes/python), [FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi), and [function limits](https://vercel.com/docs/functions/limitations).

No production deployment has been made from this repository.

## Portfolio summary

Possible resume description, adjusted to match the work you personally completed:

> Built a full-stack research-paper assistant with FastAPI and a responsive JavaScript interface. Implemented PDF parsing and OCR, page-aware chunking, session-isolated document storage, FAISS semantic retrieval, streamed Gemini answers, and source citations, with a local no-key demo path.

Suggested interview topics: session ownership and data isolation, retrieval quality, chunk size and overlap tradeoffs, model credential handling, PDF extraction limits, and the difference between ephemeral function storage and durable persistence.

## Project structure

```text
app/
├── api/             FastAPI routes and request dependencies
├── core/            shared settings and exceptions
├── domain/          typed business objects and API schemas
├── infrastructure/  SQLAlchemy models, database, and repositories
├── services/        PDF extraction, chunking, retrieval, and chat
└── static/          responsive website assets
data/sample_docs/    bundled demo PDF
tests/               API and service tests
```

## License

MIT. See [LICENSE](LICENSE).
