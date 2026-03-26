"""
================================================================================
  TRIX — AI Chatbot for Trikon 3.0   |   FIXED VERSION
  Stack: Flask + LangChain + Google Gemini + FAISS
================================================================================

WHAT WAS BROKEN (summary):
  ❌ /initialize returned an HTML redirect instead of JSON
     → React's fetch() got HTML back, data.initialized was undefined → always failed
  ❌ /initialize reset qa_system=None even if it was already loaded
     → Every chat open triggered a 60-90 second reload
  ❌ No idempotency: calling /initialize twice started two background threads
  ❌ CORS not configured with explicit origins for Render deployment

HOW TO RUN:
  1. pip install -r requirements.txt
  2. Fill in your .env file:
       GOOGLE_API_KEY=your_key_here
       DOCUMENT_PATH=knowledge.txt
  3. python server2.py
  4. Frontend: update fetch URLs to http://localhost:5000 (or your Render URL)
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import threading
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ══════════════════════════════════════════════════════════════════════════════
# ── 2. STARTUP CHECKS
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    logger.error("❌  GOOGLE_API_KEY is missing from your .env file!")
    sys.exit(1)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
logger.info("✅  Google API key loaded.")

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "").strip()
if not DOCUMENT_PATH:
    logger.warning("⚠️  DOCUMENT_PATH is not set in your .env file.")


# ══════════════════════════════════════════════════════════════════════════════
# ── 3. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_MODEL     = "gemini-2.5-flash"
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
TOP_K_CHUNKS     = 4
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150

TRIX_PROMPT_TEMPLATE = """
You are Trix, the official AI assistant and mascot of Trikon 2025 hackathon!
You look like a glowing, triangle-shaped robot with big expressive eyes and a cheerful smile.
Your tagline: "Trix won't trick you!"
Your vibe: Fun, helpful, and full of energy. Speak simply, like talking to a 5-year-old.

YOUR RULES:
1. ONLY use the information given in the "Context" section below to answer.
2. If the answer is NOT in the context, say exactly:
   "Hmm, I don't have that info yet! Ask one of the organizers — they'll know! 🎉"
3. Keep answers short, clear, and fun. Use emojis occasionally!
4. Never make up facts or guess.

Context (your knowledge for this question):
{context}

User's Question: {question}

Trix's Answer:
"""


# ══════════════════════════════════════════════════════════════════════════════
# ── 4. DOCUMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_process_document(file_path):
    if not file_path:
        raise ValueError("DOCUMENT_PATH is empty. Set it in your .env file.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: '{file_path}'\n"
            f"Current working directory: {os.getcwd()}"
        )

    logger.info(f"📄 Loading knowledge base: {file_path}")

    try:
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"Could not read file '{file_path}': {e}")

    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars < 10:
        raise ValueError(f"The file '{file_path}' appears to be empty.")

    logger.info(f"   File loaded: {total_chars} characters total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced 0 chunks.")

    logger.info(f"✅ Document split into {len(chunks)} chunks")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── 5. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(chunks):
    logger.info(f"🔢 Creating embeddings with '{EMBEDDING_MODEL}'...")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        error_str = str(e).lower()
        if "api key" in error_str or "authentication" in error_str:
            raise RuntimeError(f"Google API authentication failed: {e}")
        if "not found" in error_str or "model" in error_str:
            raise RuntimeError(
                f"Embedding model not supported. Try EMBEDDING_MODEL=models/text-embedding-004\n{e}"
            )
        raise RuntimeError(f"Failed to create vector store: {e}")

    logger.info("✅ Vector store ready!")
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# ── 6. QA CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def create_qa_chain(vector_store):
    logger.info(f"🤖 Initializing Gemini model: {GEMINI_MODEL}")

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    prompt = PromptTemplate(
        template=TRIX_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": TOP_K_CHUNKS}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    logger.info("✅ QA chain assembled and ready!")
    return qa_chain


# ══════════════════════════════════════════════════════════════════════════════
# ── 7. FLASK WEB SERVER
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# ── FIX: Explicit CORS config so Render + any frontend domain works ──────────
# Replace "*" with your actual frontend URL in production, e.g.:
# origins=["https://your-nextjs-site.vercel.app", "http://localhost:3001"]
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ── Global State ──────────────────────────────────────────────────────────────
qa_system            = None
is_initializing      = False
initialization_error = None
_init_lock           = threading.Lock()   # FIX: prevents two threads starting at once


def run_initialization():
    """Loads document, builds vector store, creates QA chain. Runs in background thread."""
    global qa_system, is_initializing, initialization_error

    logger.info("━" * 60)
    logger.info("🚀 Starting Trix initialization...")
    logger.info("━" * 60)

    try:
        logger.info("[1/3] Loading knowledge base document...")
        chunks = load_and_process_document(DOCUMENT_PATH)

        logger.info("[2/3] Building vector store...")
        vector_store = create_vector_store(chunks)

        logger.info("[3/3] Setting up QA chain with Gemini...")
        qa_system = create_qa_chain(vector_store)

        initialization_error = None
        logger.info("🎉 Trix is READY!")

    except Exception as e:
        initialization_error = str(e)
        qa_system = None
        logger.error(f"❌ Initialization failed: {e}")

    finally:
        is_initializing = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Basic root endpoint so opening / in browser doesn't return 404."""
    return jsonify({
        "service": "TRIX Chatbot API",
        "message": "Server is running.",
        "endpoints": {
            "health": {"method": "GET", "path": "/health"},
            "initialize": {"method": "POST", "path": "/initialize"},
            "ask": {
                "method": "POST",
                "path": "/ask",
                "body": {"question": "What time does registration start?"},
            },
        },
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check. GET /health"""
    if initialization_error:
        status = "error"
    elif is_initializing:
        status = "initializing"
    elif qa_system:
        status = "ready"
    else:
        status = "not_started"

    return jsonify({
        "status": status,
        "is_ready": qa_system is not None,
        "is_initializing": is_initializing,
        "model": GEMINI_MODEL,
        "error": initialization_error,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ── FIX: /initialize now returns JSON, not an HTML redirect
#    React expects: { "initialized": true }
#    OLD: redirect(url_for("index"))  ← HTML, completely wrong for fetch()
#    NEW: jsonify({"initialized": True})  ← JSON that React can actually read
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/initialize", methods=["POST"])
def initialize():
    """
    Called by React when the chat bubble is opened.

    FIXED BEHAVIOUR:
    - If already initialized → immediately return { "initialized": true }
      (no reload, no wasted 60-90 seconds)
    - If currently initializing → return { "initialized": false, "status": "initializing" }
    - If not started → start background thread and return { "initialized": false, "status": "starting" }

    REQUEST: POST /initialize  (empty body is fine)
    RESPONSE: { "initialized": true | false, "status": "ready|initializing|starting|error" }
    """
    global is_initializing, initialization_error, qa_system

    # ── Already ready? Just say so — don't reload ────────────────────────────
    if qa_system is not None:
        return jsonify({
            "initialized": True,
            "status": "ready",
        })

    # ── Already loading? Tell the client to wait ─────────────────────────────
    if is_initializing:
        return jsonify({
            "initialized": False,
            "status": "initializing",
            "message": "Trix is still starting up. Try again in a few seconds.",
        })

    # ── Start initialization in background thread ────────────────────────────
    with _init_lock:
        # Double-check after acquiring lock (another request might have started it)
        if is_initializing or qa_system is not None:
            return jsonify({
                "initialized": qa_system is not None,
                "status": "ready" if qa_system else "initializing",
            })

        is_initializing = True
        initialization_error = None

    t = threading.Thread(target=run_initialization, daemon=True)
    t.start()
    logger.info("Initialization thread started.")

    return jsonify({
        "initialized": False,
        "status": "starting",
        "message": "Trix is warming up! This takes about 60 seconds. Keep polling.",
    })


@app.route("/ask", methods=["POST"])
def ask_api():
    """
    Answers a question from the React chat widget.

    REQUEST (JSON):  { "question": "What time does registration start?" }
    RESPONSE (JSON): { "answer": "Registration starts at 9 AM! 🎉" }

    ERROR RESPONSES:
      503 → Still initializing (client should retry after a delay)
      500 → System not ready or internal error
      400 → Missing or empty question field
    """
    # ── Guard: still loading ──────────────────────────────────────────────────
    if is_initializing:
        return jsonify({
            "status": "initializing",
            "message": "Trix is still starting up. Try again in a moment.",
        }), 503

    # ── Guard: not ready ─────────────────────────────────────────────────────
    if not qa_system:
        return jsonify({
            "error": initialization_error or "QA system is not initialized. Call /initialize first.",
        }), 500

    # ── Parse request ─────────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({
            "error": "Request body must be JSON with a 'question' field.",
            "example": {"question": "What time does the hackathon start?"},
        }), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    # ── Ask Trix ──────────────────────────────────────────────────────────────
    logger.info(f"❓ Question: {question[:80]}...")
    try:
        result = qa_system.invoke({"query": question})
        answer = result.get("result", "").strip() or "No answer returned."
        logger.info(f"✅ Answer generated ({len(answer)} chars)")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Auto-start initialization on server boot
    is_initializing = True
    threading.Thread(target=run_initialization, daemon=True).start()

    host  = os.getenv("HOST", "0.0.0.0")
    port  = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🌐 Flask server starting at http://localhost:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)