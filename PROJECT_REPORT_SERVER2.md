# Project Report: `server2.py`

## 1. Purpose of this service

`server2.py` is a Flask-based Retrieval-Augmented Generation (RAG) API for the TRIX chatbot. It answers user questions using:

- A local knowledge file (`knowledge.txt`) as the source of truth
- Google embedding model to convert text chunks into vectors
- FAISS in-memory vector store for semantic retrieval
- Groq-hosted LLaMA model for final response generation

The design goal is: answer only from known context, avoid hallucinations, and expose simple API endpoints for frontend or automation use.

---

## 2. High-level architecture

The runtime flow is:

1. Load environment variables and validate required keys.
2. Read and chunk the knowledge document.
3. Generate embeddings for all chunks and build FAISS index.
4. Create a `RetrievalQA` chain using:
   - Groq LLM (`ChatGroq`)
   - FAISS retriever (`k=4`)
   - Custom Trix prompt
5. Serve HTTP endpoints (`/`, `/health`, `/initialize`, `/ask`).
6. For each user question:
   - Retrieve relevant chunks
   - Inject chunks + question into prompt
   - Generate final answer
   - Return JSON response

---

## 3. Technology stack

- Web framework: Flask (`flask`)
- CORS: `flask-cors`
- Env loading: `python-dotenv`
- RAG framework: LangChain
- Document loader: `TextLoader`
- Chunking: `RecursiveCharacterTextSplitter`
- Vector DB: FAISS (`faiss-cpu`)
- Embeddings: `GoogleGenerativeAIEmbeddings`
- LLM: `ChatGroq` (default model `llama-3.3-70b-versatile`)

---

## 4. Code structure walkthrough

### 4.1 Startup checks and configuration

Main configuration and key validation happen at import time:

- Loads `.env` via `load_dotenv()`
- Requires:
  - `GROQ_API_KEY` (for generation)
  - `GOOGLE_API_KEY` (for embeddings)
- Reads:
  - `DOCUMENT_PATH`
  - `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
  - `EMBEDDING_MODEL` (default: `models/gemini-embedding-001`)
- Static retrieval settings:
  - `TOP_K_CHUNKS = 4`
  - `CHUNK_SIZE = 800`
  - `CHUNK_OVERLAP = 150`

If required keys are missing, process exits immediately (`sys.exit(1)`), which prevents partially broken runtime.

### 4.2 Prompt design

`TRIX_PROMPT_TEMPLATE` enforces behavior:

- Use only provided context
- If answer missing, return a fixed fallback sentence
- Keep answers short and simple
- Do not guess

This is the main anti-hallucination control layer.

### 4.3 Document loading and chunking

`load_and_process_document(file_path)` performs:

1. Empty path check
2. File existence check
3. File reading through `TextLoader`
4. Basic non-empty guard (`total_chars >= 10`)
5. Chunk splitting with configured size/overlap
6. Zero-chunk guard

Returned output is a list of `Document` chunks used for embeddings.

### 4.4 Vector store creation

`create_vector_store(chunks)`:

- Instantiates `GoogleGenerativeAIEmbeddings`
- Converts chunks into vectors
- Builds FAISS index in memory using `FAISS.from_documents(...)`

It also normalizes common failure causes:

- Authentication/key failures
- Unsupported embedding model failures
- Generic vector-store creation failures

### 4.5 QA chain assembly

`create_qa_chain(vector_store)`:

- Creates `ChatGroq` with `temperature=0` (deterministic style)
- Builds LangChain `PromptTemplate`
- Creates `RetrievalQA.from_chain_type(...)` with:
  - `chain_type="stuff"`
  - retriever from FAISS (`k=4`)
  - custom prompt injected

This final object is assigned to global `qa_system`.

### 4.6 Global state model

Service lifecycle is controlled by global variables:

- `qa_system`: QA chain or `None`
- `is_initializing`: boolean startup state
- `initialization_error`: string or `None`
- `_init_lock`: threading lock to avoid duplicate initialization races

### 4.7 Initialization workflow

`run_initialization()` executes 3 phases:

1. Load and split knowledge document
2. Build FAISS vector store
3. Assemble QA chain

On success:

- `qa_system` populated
- `initialization_error = None`
- `is_initializing = False`

On failure:

- `initialization_error` set
- `qa_system = None`
- `is_initializing = False`

This function is run in a daemon thread.

---

## 5. HTTP API behavior

### 5.1 `GET /`

Returns service metadata:

- service name
- selected LLM and embedding model
- endpoint definitions

Useful as a self-documenting landing endpoint.

### 5.2 `GET /health`

Reports current state:

- `status`: one of `error | initializing | ready | not_started`
- `is_ready`
- `is_initializing`
- `llm_model`
- `error` (if any)

Use this for readiness/liveness checks.

### 5.3 `POST /initialize`

Behavior:

- If already ready: returns `{"initialized": true, "status": "ready"}`
- If currently initializing: returns status message
- Else:
  - acquires lock
  - sets init flags
  - starts background init thread
  - returns `status: starting`

This endpoint is safe against concurrent duplicate initialization attempts.

### 5.4 `POST /ask`

Expected JSON body:

```json
{ "question": "What time does registration start?" }
```

Flow:

1. If initializing: returns 503
2. If not initialized: returns 500 with init error
3. Validates JSON and non-empty question
4. Calls `qa_system.invoke({"query": question})`
5. Returns `{"answer": "<generated text>"}`

Validation and error responses:

- Missing/invalid JSON or no `question`: `400`
- Empty question after trim: `400`
- Internal chain/runtime failure: `500`

---

## 6. Startup behavior (`__main__`)

When run via `python server2.py`:

1. Sets `is_initializing = True`
2. Starts background initialization thread immediately
3. Reads server settings:
   - `HOST` (default `0.0.0.0`)
   - `PORT` (default `5000`)
   - `DEBUG` (`true/false`)
4. Runs Flask with `use_reloader=False`

Meaning: service starts listening quickly while model stack warms up asynchronously.

---

## 7. Concurrency and thread-safety analysis

What is handled well:

- `_init_lock` prevents duplicate initialization races.
- `is_initializing` gate protects `/ask` during warmup.
- daemon thread keeps initialization non-blocking.

What to note:

- Global mutable state is process-local. In multi-worker deployments, each worker will initialize independently.
- FAISS index is in memory and rebuilt on process start; no persistent store is used.

---

## 8. Data and retrieval behavior

- Source of truth is only the file at `DOCUMENT_PATH`.
- Retrieval depth is fixed at top 4 chunks.
- Prompt explicitly says to refuse unknown answers.
- `chain_type="stuff"` injects retrieved chunks directly into one prompt.

Practical implication:

- Accuracy depends on chunk quality and source file content.
- Large source files can increase initialization latency and memory usage.

---

## 9. Operational notes

- CORS is open to all origins (`origins: "*"`) with credentials disabled.
- Detailed logs are printed to stdout with INFO level.
- No authentication/rate limiting on endpoints.
- No persistent caching of embeddings/index; every restart recomputes vectors.

---

## 10. Known gaps and risks

1. No API auth: any client can call `/ask` and consume tokens.
2. Open CORS policy: suitable for dev/public API only with caution.
3. No request rate limit: potential abuse/cost risk.
4. Initialization cost on every restart: no saved FAISS index on disk.
5. Prompt/event naming is aligned to "Trikon 3.0" for 2026 in the current chatbot configuration.
6. Secrets appear present in `.env`; keys should not be committed or exposed.

---

## 11. Recommended improvements

1. Add API key/JWT middleware for `/ask` and `/initialize`.
2. Add request throttling (Flask-Limiter or reverse proxy limits).
3. Persist FAISS index to disk and reload on startup to reduce warmup time.
4. Add structured observability (request IDs, latency metrics, token usage).
5. Add automated tests for:
   - init success/failure paths
   - endpoint status transitions
   - prompt fallback behavior when context is missing
6. Align prompt identity text with current event year/details.
7. Add optional source citation return mode for debugging answer provenance.

---

## 12. Endpoint quick test commands

```bash
# 1) check status
curl -s http://localhost:5000/health

# 2) trigger initialization (if not already running)
curl -s -X POST http://localhost:5000/initialize

# 3) ask a question
curl -s -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the registration close time?"}'
```

---

## 13. Summary

`server2.py` is a clean, practical RAG API with asynchronous startup and clear status endpoints. The core implementation is sound for hackathon/demo production use. Main hardening needs are security controls (auth/rate limits), persistent indexing, and minor prompt/data alignment updates.
