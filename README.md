# ResearchPaper AI

**Chat with your PDF documents using Retrieval-Augmented Generation (RAG).**

PDF-ChatGPT is a production-ready, enterprise-style RAG application built with FastAPI, LangChain,
FAISS, and the OpenAI **Responses API**. Upload one or more PDFs, and ask natural-language questions
that are answered using only the content of your documents - with citations back to the exact page.

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

> New to Python, VS Code, or RAG entirely? See **[INSTRUCTION.md](INSTRUCTION.md)** for a complete
> beginner's walkthrough that assumes zero prior experience.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Key Concepts](#key-concepts)
  - [Embeddings](#embeddings)
  - [Chunking](#chunking)
  - [Semantic Search](#semantic-search)
  - [Vector Databases](#vector-databases)
  - [FAISS](#faiss)
- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Visual Studio Code Setup](#visual-studio-code-setup)
- [OpenAI API Setup](#openai-api-setup)
- [Running the Application](#running-the-application)
- [API Reference (Swagger)](#api-reference-swagger)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that grounds a large language model's answers
in an external knowledge source instead of relying purely on what the model memorized during training.

Without RAG, an LLM can only answer from its training data - it has no idea what's inside *your* PDF,
and it may confidently make things up ("hallucinate"). RAG fixes this by:

1. Breaking your documents into small, searchable **chunks**.
2. Converting each chunk into a numeric vector (an **embedding**) that captures its meaning.
3. Storing those vectors in a **vector database** (FAISS, in this project).
4. At question time, embedding the *question* and finding the most similar chunks (**semantic search**).
5. Feeding those chunks to the LLM as context, and asking it to answer **using only that context**.

The result: answers that are accurate, up to date, grounded in your own data, and traceable back to a
specific document and page.

```
PDF Upload -> Text Extraction -> Chunking -> Embeddings -> Vector Database
                                                              │
                                                              ▼
User Question ──────────────────────────────────────► Retriever (semantic search)
                                                              │
                                                              ▼
                                                        Relevant Chunks
                                                              │
                                                              ▼
                                                            LLM  ──►  Answer + Citations
```

## Key Concepts

### Embeddings

An embedding is a list of floating-point numbers (a vector) that represents the *meaning* of a piece of
text in high-dimensional space. Texts with similar meaning end up with vectors that are close together,
even if they don't share the same words. PDF-ChatGPT uses OpenAI's `text-embedding-3-small` model
(configurable) via `langchain-openai`'s `OpenAIEmbeddings` to embed both document chunks and user
questions.

### Chunking

LLMs have a limited context window, and embedding an entire 100-page PDF as one vector would lose all
fine-grained detail. **Chunking** splits documents into smaller, overlapping pieces (by default ~1000
characters with a 150-character overlap) so each chunk is small enough to embed precisely and large
enough to remain coherent. This project uses LangChain's `RecursiveCharacterTextSplitter`, which tries
paragraph breaks first, then sentences, then words - avoiding awkward mid-sentence cuts.

### Semantic Search

Semantic search retrieves text based on *meaning*, not exact keyword matches. A question like "How much
does it cost?" can retrieve a chunk that says "the subscription is priced at $49/month" even though no
words overlap. This works because both are embedded into vectors that land near each other in vector
space, and a similarity metric (cosine similarity) measures the distance between them.

### Vector Databases

A vector database is a data store optimized for similarity search over embeddings - instead of exact
matches like a SQL `WHERE` clause, it answers "which vectors are closest to this one?" efficiently, even
across millions of entries.

### FAISS

**FAISS** (Facebook AI Similarity Search) is an open-source library for efficient similarity search and
clustering of dense vectors. PDF-ChatGPT uses `faiss-cpu` through LangChain's `FAISS` vector store
wrapper. The index is persisted to disk under `data/vector_store/`, so it survives application restarts
without needing a separate database server - ideal for a self-hosted or small-team deployment.

---

## Features

| Feature                     | Description                                                                 |
|------------------------------|-------------------------------------------------------------------------------|
| Upload PDF                  | Drag-and-drop or click-to-upload a single PDF                                |
| Multiple PDFs                | Upload and index any number of documents                                     |
| Document Library              | Sidebar view of all uploaded documents with status, page & chunk counts       |
| Automatic Text Extraction    | `pypdf`-based per-page extraction, including encrypted PDFs with blank passwords |
| Chunking                    | Overlapping, paragraph-aware chunking via LangChain                          |
| Embedding Generation          | OpenAI `text-embedding-3-small` embeddings                                    |
| Vector Database                | Persistent FAISS index on disk                                               |
| Semantic Search                | Top-k retrieval with a relevance threshold and optional per-document scoping |
| Conversation History           | Every conversation and message persisted in SQLite                            |
| Conversation Memory            | Prior turns are replayed into the model for follow-up questions               |
| Streaming Responses            | Token-by-token answers over Server-Sent Events (SSE)                          |
| Markdown Rendering              | Answers rendered with `marked.js`, including tables and lists                |
| Syntax Highlighting             | Code blocks highlighted with `highlight.js`                                  |
| Dark Mode                       | Toggleable dark/light theme, persisted in `localStorage`                     |
| Responsive Design                | Collapsible sidebar and adaptive layout for mobile and tablet                |

## Architecture

PDF-ChatGPT follows **Clean Architecture** principles: dependencies point inward, and business logic
never depends on frameworks or delivery mechanisms.

```
┌─────────────────────────────────────────────────────────────────┐
│  Delivery layer:  app/api/        (FastAPI routers, DI, HTTP)    │
├─────────────────────────────────────────────────────────────────┤
│  Use cases:       app/services/   (RAG pipeline, chat, ingestion)│
├─────────────────────────────────────────────────────────────────┤
│  Domain:          app/domain/     (entities, schemas - no deps)  │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure:  app/infrastructure/ (SQLAlchemy, repositories) │
└─────────────────────────────────────────────────────────────────┘
```

- **`app/api`** - FastAPI route handlers. Thin: they validate input, call a service, and shape the
  HTTP response. No business logic lives here.
- **`app/services`** - the RAG pipeline and chat orchestration. Each service has one responsibility
  (PDF extraction, chunking, embeddings, vector storage, retrieval, chat generation, memory,
  document ingestion), following the **modular services** pattern.
- **`app/domain`** - framework-independent entities (`dataclasses`) and the Pydantic v2 schemas that
  define the public API contract.
- **`app/infrastructure`** - SQLAlchemy engine/session management, ORM models, and the repository
  pattern that translates between ORM rows and domain entities.
- **`app/core`** - cross-cutting concerns: a typed exception hierarchy and shared constants/prompts.
- **`app/static`** - the vanilla HTML/CSS/JS single-page chat UI (no build step required).

### RAG pipeline in this codebase

| Step                | Module                                       |
|----------------------|-----------------------------------------------|
| PDF upload            | `app/api/routes_documents.py`                 |
| Text extraction       | `app/services/pdf_service.py`                 |
| Chunking             | `app/services/chunking_service.py`            |
| Embeddings            | `app/services/embedding_service.py`           |
| Vector database        | `app/services/vector_store_service.py` (FAISS) |
| Retriever              | `app/services/retrieval_service.py`           |
| LLM (Responses API)    | `app/services/chat_service.py`                |
| Answer + citations     | `app/api/routes_chat.py`                      |

### Why the OpenAI Responses API?

This project uses `client.responses.stream(...)` for token-by-token answers and
`client.responses.parse(...)` with a Pydantic `text_format` schema for **structured output** - the
current generation OpenAI interface that supersedes Chat Completions, unifying text, tool use, and
structured generation behind one API.

## Folder Structure

```
pdf-chatgpt/
├── app/
│   ├── main.py # FastAPI app factory, lifespan, exception handlers
│   ├── config.py # Pydantic Settings (env-driven configuration)
│   ├── logging_config.py # Rotating file + console logging setup
│   ├── api/
│   │   ├── deps.py # Dependency-injected service singletons
│   │   ├── routes_health.py # GET /api/health
│   │   ├── routes_documents.py # Upload / list / get / delete documents
│   │   └── routes_chat.py # Chat, streaming chat, conversations
│   ├── core/
│   │   ├── constants.py # Prompts, file-type allow-lists
│   │   └── exceptions.py # Typed domain exception hierarchy
│   ├── domain/
│   │   ├── models.py # Framework-free dataclass entities
│   │   └── schemas.py # Pydantic v2 request/response schemas
│   ├── infrastructure/
│   │   ├── db.py # SQLAlchemy engine/session
│   │   ├── models_db.py # ORM models (documents, conversations, messages)
│   │   └── repository.py # Repository pattern over the ORM
│   ├── services/
│   │   ├── pdf_service.py # PDF -> per-page text (pypdf)
│   │   ├── chunking_service.py # Text -> overlapping chunks (LangChain)
│   │   ├── embedding_service.py # Text -> vectors (OpenAI embeddings)
│   │   ├── vector_store_service.py # FAISS persistence + similarity search
│   │   ├── retrieval_service.py # Query orchestration + relevance filtering
│   │   ├── chat_service.py # Responses API streaming + structured output
│   │   ├── memory_service.py # Conversation history read/write
│   │   └── document_service.py # End-to-end ingestion pipeline
│   └── static/
│       ├── index.html
│       ├── favicon.svg
│       ├── css/style.css
│       └── js/app.js
├── data/
│   ├── uploads/ # Uploaded PDFs (git-ignored, kept via .gitkeep)
│   ├── vector_store/ # Persisted FAISS index (git-ignored)
│   └── sample_docs/ # Small sample PDF for first-run testing
├── logs/ # Rotating application logs (git-ignored)
├── tests/ # Pytest suite (unit + integration)
├── .env.example # Configuration template
├── .gitignore
├── requirements.txt # Production dependencies
├── requirements-dev.txt # + pytest, ruff, mypy
├── pyproject.toml # Ruff / mypy / pytest configuration
├── Start App.bat # Windows one-click launcher
├── Start App (Mac).command # macOS one-click launcher
├── INSTRUCTION.md # Zero-to-running beginner's guide
└── README.md # You are here
```

## Installation

### 1. Python Installation

Install **Python 3.12 or newer** from [python.org/downloads](https://www.python.org/downloads/).
On Windows, check **"Add python.exe to PATH"** during setup. Verify with:

```bash
python --version
```

### 2. Clone or download the repository

```bash
git clone https://github.com/your-org/pdf-chatgpt.git
cd pdf-chatgpt
```

### 3. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat
```

### 4. Install dependencies

```bash
pip install -r requirements.txt

# For development (tests, linting, type-checking):
pip install -r requirements-dev.txt
```

### 5. Configure environment variables

```bash
cp .env.example .env # macOS / Linux
copy .env.example .env # Windows
```

Then open `.env` and set `OPENAI_API_KEY` (see [OpenAI API Setup](#openai-api-setup)).

## Visual Studio Code Setup

1. Install [Visual Studio Code](https://code.visualstudio.com/).
2. Open the project folder: `File -> Open Folder... -> pdf-chatgpt`.
3. Install the recommended extensions when prompted (or open the Extensions panel and install):
   - **Python** (ms-python.python)
   - **Pylance** (ms-python.vscode-pylance)
   - **Ruff** (charliermarsh.ruff)
4. Select the interpreter: `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose
   `./venv/bin/python` (or `.\venv\Scripts\python.exe` on Windows).
5. Use the pre-configured launch profiles in `.vscode/launch.json`:
   - **FastAPI: PDF-ChatGPT (uvicorn --reload)** - runs the server with the debugger attached.
   - **Python: Run tests** - runs the full pytest suite.

## OpenAI API Setup

1. Create an account at [platform.openai.com](https://platform.openai.com/).
2. Go to **API Keys** -> **Create new secret key**.
3. Copy the key (starts with `sk-...`) - it is only shown once.
4. Paste it into `.env`:

   ```env
   OPENAI_API_KEY=sk-your-real-key-here
   ```

5. Ensure your account has billing enabled; embeddings and chat completions are metered per token.

## Running the Application

```bash
# with the virtual environment activated
uvicorn app.main:app --reload
```

Or simply double-click **`Start App.bat`** (Windows) or **`Start App (Mac).command`** (macOS) - both
scripts create the virtual environment, install dependencies, verify `.env`, and launch the server
automatically.

Then open **http://localhost:8000** in your browser.

## API Reference (Swagger)

Interactive API documentation is auto-generated by FastAPI and available at:

- **Swagger UI:** http://localhost:8000/api/docs
- **ReDoc:** http://localhost:8000/api/redoc
- **OpenAPI JSON schema:** http://localhost:8000/api/openapi.json

## Screenshots

> Screenshots are not committed to keep the repository lightweight. Run the app locally and capture
> your own - recommended shots:
>
> - `docs/screenshots/chat-dark-mode.png` - main chat view (dark mode)
> - `docs/screenshots/chat-light-mode.png` - main chat view (light mode)
> - `docs/screenshots/document-library.png` - sidebar with multiple indexed PDFs
> - `docs/screenshots/swagger-ui.png` - the `/api/docs` Swagger page
>
> Create a `docs/screenshots/` folder locally and reference images with standard Markdown:
> `![Chat view](docs/screenshots/chat-dark-mode.png)`

## Testing

```bash
pytest tests -v
```

Tests that require real OpenAI embeddings (network calls) are automatically skipped unless a real
`OPENAI_API_KEY` is set, so the suite runs fully offline in CI. Lint and type-check with:

```bash
ruff check app tests
mypy app
```

## Deployment

### Vercel portfolio demo

`demo/vercel.json` publishes a static site from `demo/`. It uses the included three-page sample PDF and prepared, cited answers. Visitors can try sample questions, inspect citations, and switch themes. The demo makes no API calls, accepts no uploads, and needs no environment variables. Deploy the repository as a Vercel project with the Framework Preset set to **Other** and `demo` as the Root Directory.

The FastAPI application remains available for local, single-user use. Do not expose its document and conversation endpoints as a public multi-user service without adding user isolation, durable storage, and usage limits. Local SQLite, uploaded PDFs, and FAISS files are not suitable as persistent storage on Vercel Functions.

### Docker (recommended for production)

This repository intentionally ships without a Dockerfile to keep it minimal, but PDF-ChatGPT
containerizes cleanly. A typical `Dockerfile` would:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY data/sample_docs ./data/sample_docs
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Mount `data/` and `logs/` as volumes so uploads, the FAISS index, and the SQLite database persist
across container restarts.

### Platform-as-a-Service (Render, Railway, Fly.io, Azure App Service)

1. Set the start command to `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
2. Configure `OPENAI_API_KEY` and any other `.env` values as platform environment variables/secrets.
3. Attach a persistent disk/volume for `data/` if you need uploads and the vector index to survive
   redeploys (otherwise the app still works, but the document library resets on each deploy).

### Reverse proxy (Nginx)

Run behind Nginx or Caddy for TLS termination; forward `/` to `127.0.0.1:8000` and enable
`proxy_buffering off;` for the SSE streaming endpoint to work correctly.

## Troubleshooting

| Symptom                                            | Likely Cause & Fix                                                                 |
|------------------------------------------------------|----------------------------------------------------------------------------------|
| `OPENAI_API_KEY is not configured` (HTTP 412)          | `.env` is missing or still has the placeholder key. Set a real key and restart.  |
| `409 Conflict` on `/api/chat`                          | No documents indexed yet. Upload a PDF first.                                    |
| `No extractable text was found`                         | The PDF is scanned images with no OCR text layer. Use a text-based PDF, or add an OCR step (see Future Improvements). |
| Upload fails with `413`/file-too-large errors             | Increase `MAX_UPLOAD_MB` in `.env`, or reduce the file size.                      |
| `ModuleNotFoundError` when running `uvicorn`               | The virtual environment isn't activated, or `pip install -r requirements.txt` wasn't run. |
| Port 8000 already in use                                   | Stop the other process, or run with `--port 8001` and update your browser URL.   |
| Streaming responses stop mid-answer                         | Check your OpenAI account's rate limits/quota; see the server logs in `logs/app.log`. |
| `sqlite3.OperationalError: no such table`                    | The database wasn't initialized - this only happens if you import `app.main` without running the app's lifespan (e.g., in ad-hoc scripts). Restart via `uvicorn`. |

## Future Improvements

- OCR support for scanned PDFs (e.g., via `pytesseract`) so image-only documents become searchable.
- Multi-tenant auth (API keys or OAuth) for shared deployments.
- Swap FAISS for a managed vector database (pgvector, Pinecone, Qdrant) for horizontal scaling.
- Hybrid search (BM25 + embeddings) for better keyword-sensitive retrieval.
- Re-ranking retrieved chunks with a cross-encoder before generation.
- Support for additional file types (DOCX, TXT, Markdown, HTML).
- WebSocket-based chat as an alternative to SSE for bidirectional streaming.

## License

Released under the [MIT License](LICENSE).
