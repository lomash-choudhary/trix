"""
================================================================================
  TRIX — AI Chatbot for Trikon 3.0
  Stack: Flask + LangChain + Google Gemini + FAISS
================================================================================

FOR JUNIORS — HOW TO READ THIS FILE:
  This file is organized into 7 clearly labelled sections.
  Search for "── 1.", "── 2." etc. to jump between them.

  1. IMPORTS          → All external libraries this code needs
  2. STARTUP CHECKS   → Catches common mistakes BEFORE the server starts
  3. CONFIG           → All the settings you are allowed to change
  4. DOCUMENT LOADING → Reads your .txt knowledge base and chunks it up
  5. VECTOR STORE     → Turns text into searchable number-lists (embeddings)
  6. QA CHAIN         → Connects everything to Gemini to answer questions
  7. FLASK WEB SERVER → Routes, HTML page, and the main server loop

COMMON ERRORS AND FIXES:
  ❌ "GOOGLE_API_KEY not found"
     → Create a .env file and add: GOOGLE_API_KEY=your_key_here
     → Get a key at: https://aistudio.google.com/app/apikey

  ❌ "Document not found"
     → Add to .env: DOCUMENT_PATH=knowledge.txt
     → Make sure the .txt file actually exists in the same folder

  ❌ "ModuleNotFoundError"
     → Run: pip install -r requirements.txt

  ❌ Answers are wrong or empty
     → Make sure your .txt file has actual content
     → Try increasing TOP_K_CHUNKS in the CONFIG section below

HOW TO RUN:
  1. pip install -r requirements.txt
  2. Fill in your .env file (copy from .env.example)
  3. python app.py
  4. Visit http://localhost:5000 in your browser
================================================================================
"""


# ══════════════════════════════════════════════════════════════════════════════
# ── 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
# If any of these fail with "ModuleNotFoundError", run: pip install -r requirements.txt

import os
import sys
import threading
import logging

from dotenv import load_dotenv                                      # Reads the .env file
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_cors import CORS                                         # Allows other apps to call our API

from langchain_community.document_loaders import TextLoader         # Reads .txt files
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits text into chunks
from langchain_community.vectorstores import FAISS                  # Fast in-memory search database
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA                            # The QA pipeline
from langchain.prompts import PromptTemplate                        # Formats questions for the AI


# ══════════════════════════════════════════════════════════════════════════════
# ── 2. STARTUP CHECKS  (runs immediately when you start the server)
# ══════════════════════════════════════════════════════════════════════════════

# Set up logging — this prints timestamped messages to your terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]   # Print to terminal
)
logger = logging.getLogger(__name__)

# Load the .env file so os.getenv() can read it
# The .env file must be in the SAME folder as this app.py file
load_dotenv()

# ── Check 1: Google API Key ───────────────────────────────────────────────────
# Without this, NOTHING works. We check it immediately so you get a clear error.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if not GOOGLE_API_KEY:
    logger.error("=" * 60)
    logger.error("❌  GOOGLE_API_KEY is missing from your .env file!")
    logger.error("    1. Go to: https://aistudio.google.com/app/apikey")
    logger.error("    2. Create a free API key")
    logger.error("    3. Open your .env file and add:")
    logger.error("       GOOGLE_API_KEY=paste_your_key_here")
    logger.error("=" * 60)
    sys.exit(1)   # Stop the server — no point continuing without a key

# Also expose it as an environment variable (some LangChain internals need this)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
logger.info("✅  Google API key loaded.")

# ── Check 2: Document Path ───────────────────────────────────────────────────
# This is the path to your .txt knowledge base file.
# We store it here but only validate it later when initialization runs.
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "").strip()

if not DOCUMENT_PATH:
    logger.warning("⚠️  DOCUMENT_PATH is not set in your .env file.")
    logger.warning("    The system will fail when it tries to load the document.")
    logger.warning("    Add this to your .env file: DOCUMENT_PATH=knowledge.txt")


# ══════════════════════════════════════════════════════════════════════════════
# ── 3. CONFIG  — THINGS JUNIORS CAN SAFELY EDIT
# ══════════════════════════════════════════════════════════════════════════════

# Which Gemini model to use for generating answers
# "gemini-1.5-flash" → faster and free tier friendly  ✅ recommended
# "gemini-1.5-pro"   → smarter but slower and may cost more
GEMINI_MODEL = "gemini-2.5-flash"

# Which Google model to use for creating embeddings
# NOTE: `models/embedding-001` is deprecated on many newer API endpoints.
# Use text-embedding-004 by default, but allow override from .env if needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip() or "models/gemini-embedding-001"

# How many relevant text chunks to send to Gemini per question
# Higher = more context = better answers, but slower and more expensive
TOP_K_CHUNKS = 4

# Max characters per chunk when splitting the document
# Lower = more chunks (more precise), Higher = fewer chunks (more context per chunk)
CHUNK_SIZE = 800

# How many characters to overlap between chunks
# Overlap prevents losing information at chunk boundaries
CHUNK_OVERLAP = 150

# ── Trix's Personality Prompt ────────────────────────────────────────────────
# Edit this to change how Trix talks, what rules it follows, etc.
# {context} → replaced with the relevant text chunks found in the document
# {question} → replaced with the user's actual question
# DO NOT remove {context} or {question} — LangChain needs them!

TRIX_PROMPT_TEMPLATE = """
You are Trix, the official AI assistant and mascot of Trikon 3.0 hackathon!
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
    """
    Reads your .txt knowledge base file and splits it into chunks.

    WHY CHUNKS?
    Gemini has a limit on how much text it can process at once.
    Instead of sending the whole document, we:
      1. Split the document into small chunks
      2. Find the chunks most relevant to the question
      3. Send only those chunks to Gemini

    Args:
        file_path (str): Path to the .txt file

    Returns:
        list[Document]: A list of LangChain Document objects (text chunks)

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty
    """
    # ── Validate the file path ────────────────────────────────────────────────
    if not file_path:
        raise ValueError(
            "DOCUMENT_PATH is empty. Set it in your .env file.\n"
            "Example: DOCUMENT_PATH=knowledge.txt"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: '{file_path}'\n"
            f"Current working directory: {os.getcwd()}\n"
            "Check that:\n"
            "  1. The file actually exists\n"
            "  2. The path in DOCUMENT_PATH is correct (can be relative or absolute)"
        )

    logger.info(f"📄 Loading knowledge base: {file_path}")

    # ── Load the file ─────────────────────────────────────────────────────────
    # encoding="utf-8" handles special characters (Hindi, emojis, etc.)
    # errors="replace" means broken characters become ? instead of crashing
    try:
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(
            f"Could not read file '{file_path}': {e}\n"
            "Make sure the file is a plain text (.txt) file."
        )

    # ── Check the file isn't empty ────────────────────────────────────────────
    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars < 10:
        raise ValueError(
            f"The file '{file_path}' appears to be empty or has very little content.\n"
            "Add your knowledge base content to this file."
        )

    logger.info(f"   File loaded: {total_chars} characters total")

    # ── Split into chunks ─────────────────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — to keep chunks as meaningful as possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these split points in order
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting produced 0 chunks. Check your document content.")

    logger.info(f"✅ Document split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── 5. VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════

def create_vector_store(chunks):
    """
    Converts text chunks into embeddings and stores them in FAISS.

    WHAT IS AN EMBEDDING?
    An embedding is a list of ~768 numbers that represents the "meaning" of text.
    Text with similar meanings gets similar numbers.
    This lets us search by meaning ("match time") not just keywords ("schedule").

    WHAT IS FAISS?
    FAISS (Facebook AI Similarity Search) is a fast in-memory database
    that finds the most similar embeddings to a query embedding.

    Args:
        chunks (list[Document]): Output from load_and_process_document()

    Returns:
        FAISS: A searchable vector store loaded in memory
    """
    logger.info("🔢 Creating embeddings — this may take 30–90 seconds...")
    logger.info(f"   Processing {len(chunks)} chunks using '{EMBEDDING_MODEL}'")

    try:
        # Initialize Google's embedding model
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

        # Build the FAISS index — this calls the Google API once per chunk
        # If this fails with an auth error, your API key is wrong or expired
        vector_store = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        error_str = str(e).lower()
        if "api key" in error_str or "authentication" in error_str or "permission" in error_str:
            raise RuntimeError(
                "Google API authentication failed!\n"
                "Your GOOGLE_API_KEY might be wrong or expired.\n"
                "Get a new key at: https://aistudio.google.com/app/apikey\n"
                f"Original error: {e}"
            )
        if "not found" in error_str or "embedcontent" in error_str or "model" in error_str:
            raise RuntimeError(
                "Embedding model is not supported by your current Gemini API endpoint.\n"
                "Try setting EMBEDDING_MODEL=models/text-embedding-004 in your .env file.\n"
                f"Current EMBEDDING_MODEL: {EMBEDDING_MODEL}\n"
                f"Original error: {e}"
            )
        raise RuntimeError(f"Failed to create vector store: {e}")

    logger.info("✅ Vector store ready!")
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# ── 6. QA CHAIN
# ══════════════════════════════════════════════════════════════════════════════

def create_qa_chain(vector_store):
    """
    Assembles the full question-answering pipeline.

    HOW IT WORKS STEP BY STEP:
      User types a question
         ↓
      Question is turned into an embedding
         ↓
      FAISS finds the TOP_K_CHUNKS most similar text chunks
         ↓
      Chunks + question are inserted into TRIX_PROMPT_TEMPLATE
         ↓
      Filled prompt is sent to Gemini
         ↓
      Gemini returns an answer
         ↓
      Answer is shown to the user

    Args:
        vector_store (FAISS): Output from create_vector_store()

    Returns:
        RetrievalQA: A LangChain chain ready to answer questions
    """
    logger.info(f"🤖 Initializing Gemini model: {GEMINI_MODEL}")

    # ChatGoogleGenerativeAI = the Gemini language model
    # temperature=0 → always gives the same answer for the same question (deterministic)
    # Set temperature=0.7 if you want more creative/varied answers
    # NOTE: We do NOT ping the model here — it wastes free-tier API quota on every startup.
    #       The first real connectivity test happens when a user submits a question.
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )
    logger.info("   Gemini model client initialised ✅")

    # Build the prompt using our TRIX_PROMPT_TEMPLATE from the CONFIG section
    prompt = PromptTemplate(
        template=TRIX_PROMPT_TEMPLATE,
        input_variables=["context", "question"],   # Must match {context} and {question} in template
    )

    # RetrievalQA is the full pipeline:
    #   - chain_type="stuff" → stuff all chunks into one prompt (simplest approach)
    #   - retriever          → searches FAISS for relevant chunks
    #   - prompt             → formats the final question sent to Gemini
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": TOP_K_CHUNKS}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,  # Set True if you want to see which chunks were used
    )

    logger.info("✅ QA chain assembled and ready!")
    return qa_chain


# ══════════════════════════════════════════════════════════════════════════════
# ── 7. FLASK WEB SERVER
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # Allows the API to be called from a different domain/app (e.g. a React frontend)

# ── Global State ──────────────────────────────────────────────────────────────
# These variables are shared across all web requests.
# They track whether the AI system is ready.

qa_system           = None   # Will hold the QA chain once initialized
is_initializing     = False  # True while the system is loading in the background
initialization_error = None  # Holds the error message string if setup failed


def run_initialization():
    """
    Loads the document, builds the vector store, and creates the QA chain.

    This runs in a background THREAD so Flask can start serving requests
    immediately. Users see a "loading" page instead of a timeout.

    The `global` keyword is needed because we're MODIFYING the global variables,
    not just reading them.
    """
    global qa_system, is_initializing, initialization_error

    logger.info("━" * 60)
    logger.info("🚀 Starting Trix initialization...")
    logger.info("━" * 60)

    try:
        # Step 1 — Load and chunk the .txt document
        logger.info("[1/3] Loading knowledge base document...")
        chunks = load_and_process_document(DOCUMENT_PATH)

        # Step 2 — Build FAISS vector store from chunks
        logger.info("[2/3] Building vector store...")
        vector_store = create_vector_store(chunks)

        # Step 3 — Connect vector store to Gemini
        logger.info("[3/3] Setting up QA chain with Gemini...")
        qa_system = create_qa_chain(vector_store)

        logger.info("━" * 60)
        logger.info("🎉 Trix is READY! Visit http://localhost:5000")
        logger.info("━" * 60)

    except Exception as e:
        # Save error so the web page can show it to the user
        initialization_error = str(e)
        qa_system = None
        logger.error("━" * 60)
        logger.error(f"❌ Initialization failed: {e}")
        logger.error("   Fix the issue above, then click 'Retry' in the browser.")
        logger.error("━" * 60)

    finally:
        # ALWAYS flip this flag back when done — success or failure
        is_initializing = False


# ── HTML Template ─────────────────────────────────────────────────────────────
# This is the full HTML page served at http://localhost:5000
# Juniors: feel free to edit the CSS and HTML text below.
# The {% if %} blocks are Jinja2 template syntax — they control what shows up.

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trix — Trikon 3.0 Assistant</title>
    <style>
        /* ── Reset & Base ── */
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea, #764ba2);
            min-height: 100vh;
            padding: 30px 20px;
        }

        /* ── Page Title ── */
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 24px;
            text-shadow: 1px 2px 6px rgba(0,0,0,0.3);
            font-size: 1.8rem;
        }

        /* ── Main Card ── */
        .card {
            background: rgba(255, 255, 255, 0.97);
            border-radius: 16px;
            padding: 32px;
            max-width: 740px;
            margin: 0 auto;
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }

        /* ── Question textarea ── */
        textarea {
            width: 100%;
            padding: 14px;
            border: 2px solid #ddd;
            border-radius: 10px;
            resize: vertical;
            min-height: 110px;
            margin-bottom: 14px;
            font-family: Arial, sans-serif;
            font-size: 15px;
            line-height: 1.5;
            transition: border-color 0.2s;
        }
        textarea:focus { outline: none; border-color: #667eea; }

        /* ── Buttons ── */
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 12px 22px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
            font-weight: bold;
            transition: opacity 0.2s, transform 0.1s;
        }
        .btn:hover { opacity: 0.88; transform: translateY(-1px); }
        .btn-blue { background: linear-gradient(135deg, #2196F3, #1565C0); }

        /* ── Trix's Answer Box ── */
        .answer-box {
            background: #f0faf0;
            border-left: 5px solid #43a047;
            border-radius: 10px;
            padding: 20px 22px;
            margin-top: 24px;
            line-height: 1.7;
            font-size: 15px;
            white-space: pre-wrap;       /* Keeps line breaks from Gemini */
            word-wrap: break-word;
        }

        /* ── Status Bar ── */
        .status-bar {
            margin-top: 28px;
            padding: 14px 18px;
            background: #f5f7ff;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
            font-size: 14px;
            color: #555;
        }

        /* ── Error Box ── */
        .error-box {
            background: #fff0f0;
            border-left: 5px solid #e53935;
            border-radius: 10px;
            padding: 18px 22px;
            margin-bottom: 20px;
            color: #b71c1c;
        }
        .error-box pre {
            margin-top: 10px;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-word;
            background: #ffeaea;
            padding: 10px;
            border-radius: 6px;
        }

        /* ── Loading Box ── */
        .loading-box {
            background: #fffde7;
            border-left: 5px solid #f9a825;
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 20px;
            text-align: center;
        }
        .spinner {
            display: inline-block;
            width: 22px; height: 22px;
            border: 3px solid #ddd;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        h3 { margin-bottom: 14px; color: #333; }
    </style>
</head>
<body>
<h1>🔺 Trix — Trikon 3.0 AI Assistant</h1>

<div class="card">

    <!-- ── ERROR STATE: Show if initialization failed ── -->
    {% if initialization_error %}
    <div class="error-box">
        <strong>⚠️ Setup Failed</strong>
        <pre>{{ initialization_error }}</pre>
        <p style="margin-top:12px;">Fix the issue above, then retry:</p>
        <br>
        <form action="/initialize" method="post">
            <button class="btn btn-blue" type="submit">🔄 Retry Initialization</button>
        </form>
    </div>
    {% endif %}

    <!-- ── LOADING STATE: Show while system is starting ── -->
    {% if is_initializing %}
    <div class="loading-box">
        <span class="spinner"></span>
        <strong>Trix is warming up...</strong>
        <p style="margin-top:10px; color:#888; font-size:14px;">
            Loading the knowledge base and connecting to Gemini.<br>
            This usually takes 1–2 minutes. Refresh the page in a moment!
        </p>
    </div>

    <!-- ── READY STATE: Show the chat form when system is up ── -->
    {% elif qa_system %}
    <form action="/web-ask" method="post">
        <h3>💬 Ask Trix anything about Trikon 3.0:</h3>
        <textarea
            name="question"
            placeholder="e.g. What time does the hackathon start? Where is the venue?"
            required
        >{{ question }}</textarea>
        <button class="btn" type="submit">🚀 Ask Trix</button>
    </form>

    <!-- Show Trix's answer if one exists -->
    {% if answer %}
    <div class="answer-box">
        <strong>🔺 Trix says:</strong><br><br>{{ answer }}
    </div>
    {% endif %}

    <!-- ── NOT STARTED STATE: Show manual start button ── -->
    {% elif not initialization_error %}
    <div style="text-align:center; padding: 20px 0;">
        <p style="margin-bottom:16px; color:#555;">Trix hasn't started yet.</p>
        <form action="/initialize" method="post">
            <button class="btn" type="submit">▶️ Start Trix</button>
        </form>
    </div>
    {% endif %}

    <!-- ── Status Bar (always visible at bottom) ── -->
    <div class="status-bar">
        System Status:
        {% if is_initializing %}
            <span style="color:#f9a825;">⏳ Initializing...</span>
        {% elif qa_system %}
            <span style="color:#43a047;">✅ Ready &nbsp;</span>
        {% elif initialization_error %}
            <span style="color:#e53935;">❌ Error — see above</span>
        {% else %}
            <span style="color:#9e9e9e;">Not started</span>
        {% endif %}
    </div>

</div>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """
    Home page — renders the main chat UI.
    GET http://localhost:5000/
    """
    return render_template_string(
        HTML_TEMPLATE,
        qa_system=qa_system is not None,
        is_initializing=is_initializing,
        initialization_error=initialization_error,
        question="",
        answer=None,
        model=GEMINI_MODEL,
    )


@app.route("/initialize", methods=["POST"])
def initialize():
    """
    Starts (or restarts) the initialization process in a background thread.
    Called when the user clicks "Start Trix" or "Retry" on the page.
    POST http://localhost:5000/initialize
    """
    global is_initializing, initialization_error, qa_system

    # Don't start a second thread if one is already running
    if is_initializing:
        logger.info("Initialize called but already in progress — ignoring.")
        return redirect(url_for("index"))

    # Reset everything before retrying
    is_initializing = True
    initialization_error = None
    qa_system = None

    # daemon=True → thread is killed automatically if the main program exits
    t = threading.Thread(target=run_initialization, daemon=True)
    t.start()

    logger.info("Initialization thread started.")
    return redirect(url_for("index"))


@app.route("/web-ask", methods=["POST"])
def web_ask():
    """
    Handles a question submitted from the HTML form.
    Returns the full HTML page with Trix's answer inserted.
    POST http://localhost:5000/web-ask
    """
    question = request.form.get("question", "").strip()

    # ── Guard: system still loading ──
    if is_initializing:
        return render_template_string(
            HTML_TEMPLATE,
            qa_system=False, is_initializing=True,
            initialization_error=None,
            question=question, answer=None, model=GEMINI_MODEL,
        )

    # ── Guard: system not ready ──
    if not qa_system:
        return render_template_string(
            HTML_TEMPLATE,
            qa_system=False, is_initializing=False,
            initialization_error=initialization_error or "System is not initialized. Click Start.",
            question=question, answer=None, model=GEMINI_MODEL,
        )

    # ── Guard: empty question ──
    if not question:
        return redirect(url_for("index"))

    # ── Ask Trix ──
    logger.info(f"❓ Question received: {question[:80]}...")
    try:
        # .invoke() is the modern replacement for calling the chain directly
        # It sends question → FAISS → Gemini → returns {"result": "..."}
        result = qa_system.invoke({"query": question})
        answer = result.get("result", "").strip()

        if not answer:
            answer = "I got an empty response. Please try rephrasing your question!"

        logger.info(f"✅ Answer generated ({len(answer)} chars)")

    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        answer = (
            f"Oops! I hit a problem while thinking. 😅\n\n"
            f"Error: {e}\n\n"
            "Try asking again, or check the terminal for more details."
        )

    return render_template_string(
        HTML_TEMPLATE,
        qa_system=True, is_initializing=False,
        initialization_error=None,
        question=question, answer=answer, model=GEMINI_MODEL,
    )


@app.route("/ask", methods=["POST"])
def ask_api():
    """
    JSON API endpoint for programmatic access (Postman, mobile apps, etc.)

    REQUEST (JSON body):
        { "question": "What time does registration start?" }

    RESPONSE (JSON):
        { "answer": "Registration starts at 9 AM! 🎉" }

    ERROR RESPONSES:
        503 → System still initializing
        500 → System not ready or crashed
        400 → Bad request (missing question field)
    """
    if is_initializing:
        return jsonify({
            "status": "initializing",
            "message": "Trix is still starting up. Try again in a minute."
        }), 503

    if not qa_system:
        return jsonify({
            "error": initialization_error or "QA system is not initialized."
        }), 500

    # Parse and validate the JSON body
    data = request.get_json(silent=True)  # silent=True → returns None instead of 400 on bad JSON
    if not data or "question" not in data:
        return jsonify({
            "error": "Request body must be JSON with a 'question' field.",
            "example": {"question": "What time does the hackathon start?"}
        }), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        result = qa_system.invoke({"query": question})
        answer = result.get("result", "").strip() or "No answer returned."
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint — for deployment monitoring or debugging.
    GET http://localhost:5000/health

    Returns JSON with the current system status.
    """
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
        "document_path": DOCUMENT_PATH,
        "error": initialization_error,
    })


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — runs when you do: python app.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start initialization immediately in the background
    # The web server starts right away so the browser doesn't time out
    is_initializing = True
    threading.Thread(target=run_initialization, daemon=True).start()

    # Read server settings from .env (or use defaults)
    host = os.getenv("HOST", "0.0.0.0")          # 0.0.0.0 = accessible on your local network
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🌐 Flask server starting at http://localhost:{port}")
    logger.info(f"   (debug={debug}, host={host})")

    # NOTE: debug=True auto-restarts the server on code changes, but it
    # also starts initialization TWICE — keep debug=False in production.
    app.run(host=host, port=port, debug=debug, use_reloader=False)
    #                                           ↑ use_reloader=False prevents
    #                                             the background thread from
    #                                             running twice