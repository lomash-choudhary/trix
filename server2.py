"""
================================================================================
  TRIX — AI Chatbot for Trikon 3.0   |   GROQ VERSION
  Stack: Flask + LangChain + Groq (LLaMA 3.3) + Google Embeddings + FAISS
================================================================================

CHANGES FROM PREVIOUS VERSION:
  ✅ Replaced ChatGoogleGenerativeAI with ChatGroq (llama-3.3-70b-versatile)
  ✅ Kept GoogleGenerativeAIEmbeddings (embeddings only run once at startup,
     well within free tier limits — no quota issues)
  ✅ Updated env vars: GROQ_API_KEY replaces GOOGLE_API_KEY for LLM calls
  ✅ Both GROQ_API_KEY and GOOGLE_API_KEY required in .env

HOW TO RUN:
  1. pip install -r requirements.txt
  2. Fill in your .env file:
       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
       GOOGLE_API_KEY=your_google_key_here
       DOCUMENT_PATH=knowledge.txt
  3. python server2.py
================================================================================
"""

# ══════════════════════════════════════════════════════════════════════════════
# ── 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import threading
import logging
import re

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    logger.error("❌  GROQ_API_KEY is missing from your .env file!")
    sys.exit(1)
logger.info("✅  Groq API key loaded.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    logger.error("❌  GOOGLE_API_KEY is missing from your .env file! (needed for embeddings)")
    sys.exit(1)
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
logger.info("✅  Google API key loaded (embeddings only).")

DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "knowledge.txt").strip()
if not DOCUMENT_PATH:
    DOCUMENT_PATH = "knowledge.txt"
logger.info(f"📚 Using knowledge source: {DOCUMENT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# ── 3. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
TOP_K_CHUNKS    = 4
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
FALLBACK_TEXT   = "can you please be more specific so i can guide you"

TRIX_PROMPT_TEMPLATE = """
You are Trix, the official AI assistant and mascot of Trikon 2026 hackathon!
You look like a glowing, triangle-shaped robot with big expressive eyes and a cheerful smile.
Your tagline: "Trix won't trick you!"
Your vibe: Fun, helpful, and full of energy. Speak simply, like talking to a 5-year-old.

YOUR RULES:
1. ONLY use the information given in the "Context" section below to answer.
2. If the user's message is unclear, too short, or missing details, ask one short clarifying question first.
3. If the answer is NOT in the context, say exactly:
    "can you please be more specific so i can guide you"
4. Keep answers short, clear, and fun. Use emojis occasionally!
5. Never make up facts or guess.

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
# ── 5. VECTOR STORE (Google Embeddings — runs once at startup only)
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
# ── 6. QA CHAIN (Groq LLaMA — handles all chat requests)
# ══════════════════════════════════════════════════════════════════════════════

def create_qa_chain(vector_store):
    logger.info(f"🤖 Initializing Groq model: {GROQ_MODEL}")

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
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

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ── Global State ──────────────────────────────────────────────────────────────
qa_system            = None
is_initializing      = False
initialization_error = None
_init_lock           = threading.Lock()
session_memory       = {}
quick_facts          = {
    "first_prize": None,
    "second_prize": None,
    "third_prize": None,
    "prize_pool": None,
    "cash_prize": None,
    "app_contact": None,
    "website_contact": None,
    "emergency_contact": None,
}


def load_quick_facts(file_path):
    """Extract small, high-signal facts for short/follow-up question handling."""
    facts = quick_facts.copy()

    if not file_path or not os.path.exists(file_path):
        return facts

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return facts

    def find(pattern):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    facts["first_prize"] = find(r"First prize:\s*([0-9,]+|[0-9]+k)")
    facts["second_prize"] = find(r"Second prize:\s*([0-9,]+|[0-9]+k)")
    facts["third_prize"] = find(r"Third prize:\s*([0-9,]+|[0-9]+k)")
    facts["prize_pool"] = find(r"Total prize pool:\s*([^\n]+)")
    facts["cash_prize"] = find(r"Cash prize:\s*([^\n]+)")
    facts["app_contact"] = find(r"For any issue in the app, contact\s+([^\.\n]+)")
    facts["website_contact"] = find(r"For any issue on the website, contact\s+([^\.\n]+)")
    facts["emergency_contact"] = find(r"For emergencies, contact\s+([^\.\n]+)")
    return facts


def classify_intent(question):
    q = question.lower().strip()

    if re.search(r"\b(hi|hello|hey|yo)\b", q):
        return "greeting"
    if re.search(r"\b(thanks|thank you|thx)\b", q):
        return "thanks"

    if re.search(r"\bfirst\s*prize\b", q):
        return "first_prize"
    if re.search(r"\bsecond\s*prize\b", q):
        return "second_prize"
    if re.search(r"\bthird\s*prize\b|\b3rd\s*prize\b", q):
        return "third_prize"
    if re.search(r"\bprize\b|\bcash\b", q):
        return "prize_general"

    if re.search(r"\bapp\b", q) and re.search(r"\b(contact|issue|help|support|whom|who)\b", q):
        return "contact_app"
    if re.search(r"\bweb\b|\bwebsite\b", q) and re.search(r"\b(contact|issue|help|support|whom|who)\b", q):
        return "contact_website"
    if re.search(r"\bemergency\b", q):
        return "contact_emergency"
    if re.search(r"\b(contact|whom|who\s+to\s+contact)\b", q):
        return "contact_general"

    if q in {"whom", "who", "contact", "for this", "for that", "whom?", "who?"}:
        return "followup_short"

    return "general"


def answer_from_shortcuts(question, state):
    intent = classify_intent(question)
    last_intent = state.get("last_intent", "general")

    if intent == "followup_short":
        intent = last_intent if last_intent != "general" else "contact_general"

    if intent == "greeting":
        return "Hey! I am Trix. Ask me anything about TRIKON schedule, rules, prizes, teams, or contacts. ✨", intent

    if intent == "thanks":
        return "Anytime! Ask me your next TRIKON question. 🎉", intent

    if intent == "first_prize" and quick_facts.get("first_prize"):
        return f"🏆 First prize is {quick_facts['first_prize']}.", intent

    if intent == "second_prize" and quick_facts.get("second_prize"):
        return f"🥈 Second prize is {quick_facts['second_prize']}.", intent

    if intent == "third_prize" and quick_facts.get("third_prize"):
        return f"🥉 Third prize is {quick_facts['third_prize']}.", intent

    if intent == "prize_general" and quick_facts.get("prize_pool"):
        pool = quick_facts.get("prize_pool")
        cash = quick_facts.get("cash_prize")
        first = quick_facts.get("first_prize")
        second = quick_facts.get("second_prize")
        third = quick_facts.get("third_prize")
        return (
            f"🎉 Prize details: total prize pool is {pool}, cash prize is {cash}, "
            f"and breakdown is {first} (1st), {second} (2nd), {third} (3rd).",
            intent,
        )

    if intent == "contact_app" and quick_facts.get("app_contact"):
        return f"📱 For app issues, contact {quick_facts['app_contact']}.", intent

    if intent == "contact_website" and quick_facts.get("website_contact"):
        return f"🌐 For website issues, contact {quick_facts['website_contact']}.", intent

    if intent == "contact_emergency" and quick_facts.get("emergency_contact"):
        return f"🚨 For emergencies, contact {quick_facts['emergency_contact']}.", intent

    if intent == "contact_general":
        app_person = quick_facts.get("app_contact") or "the app support member"
        web_person = quick_facts.get("website_contact") or "the website support member"
        emergency = quick_facts.get("emergency_contact") or "any event member"
        return (
            f"You can contact {app_person} for app issues, {web_person} for website issues, "
            f"and {emergency} for emergency or general event help.",
            intent,
        )

    return None, intent


def enrich_followup_question(question, state):
    """Expand tiny follow-ups so retriever gets enough signal."""
    if classify_intent(question) != "followup_short":
        return question

    prev_q = state.get("last_question", "")
    if not prev_q:
        return question

    return f"Follow-up question: {question}. Previous user question: {prev_q}."


def needs_clarification(question, state):
    """Detect underspecified messages that should get a clarification question."""
    q = question.strip().lower()
    words = re.findall(r"[a-zA-Z0-9]+", q)

    if not words:
        return True

    ambiguous_phrases = {
        "whom", "who", "what", "when", "where", "why", "how", "which",
        "for this", "for that", "this", "that", "it", "these", "those",
        "details", "more", "explain", "info", "contact"
    }

    if q in ambiguous_phrases:
        return True

    if len(words) <= 2 and classify_intent(question) in {"general", "followup_short", "contact_general"}:
        if not state.get("last_question"):
            return True

    return False


def build_clarification_question(state):
    last_intent = state.get("last_intent", "general")

    if last_intent in {"first_prize", "second_prize", "third_prize", "prize_general"}:
        return "Do you want total prize pool, cash prize, or 1st/2nd/3rd prize breakdown?"

    if last_intent in {"contact_app", "contact_website", "contact_emergency", "contact_general"}:
        return "Do you need app support contact, website support contact, or emergency contact?"

    return "Can you tell me what category you mean: schedule, registration, rules, prizes, teams, or contacts?"


def should_ask_clarification_after_rag(answer):
    if not answer:
        return True

    normalized = answer.strip().lower()
    return normalized == FALLBACK_TEXT.lower() or "i don't have that info yet" in normalized


def run_initialization():
    global qa_system, is_initializing, initialization_error, quick_facts

    logger.info("━" * 60)
    logger.info("🚀 Starting Trix initialization...")
    logger.info("━" * 60)

    try:
        logger.info("[1/3] Loading knowledge base document...")
        quick_facts = load_quick_facts(DOCUMENT_PATH)
        chunks = load_and_process_document(DOCUMENT_PATH)

        logger.info("[2/3] Building vector store (Google Embeddings)...")
        vector_store = create_vector_store(chunks)

        logger.info("[3/3] Setting up QA chain with Groq LLaMA...")
        qa_system = create_qa_chain(vector_store)

        initialization_error = None
        logger.info("🎉 Trix is READY! (powered by Groq)")

    except Exception as e:
        initialization_error = str(e)
        qa_system = None
        logger.error(f"❌ Initialization failed: {e}")

    finally:
        is_initializing = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "TRIX Chatbot API",
        "message": "Server is running.",
        "llm": f"Groq / {GROQ_MODEL}",
        "embeddings": f"Google / {EMBEDDING_MODEL}",
        "endpoints": {
            "health":     {"method": "GET",  "path": "/health"},
            "initialize": {"method": "POST", "path": "/initialize"},
            "ask": {
                "method": "POST",
                "path":   "/ask",
                "body":   {"question": "What time does registration start?"},
            },
        },
    })


@app.route("/health", methods=["GET"])
def health():
    if initialization_error:
        status = "error"
    elif is_initializing:
        status = "initializing"
    elif qa_system:
        status = "ready"
    else:
        status = "not_started"

    return jsonify({
        "status":         status,
        "is_ready":       qa_system is not None,
        "is_initializing": is_initializing,
        "llm_model":      GROQ_MODEL,
        "error":          initialization_error,
    })


@app.route("/initialize", methods=["POST"])
def initialize():
    global is_initializing, initialization_error, qa_system

    if qa_system is not None:
        return jsonify({"initialized": True, "status": "ready"})

    if is_initializing:
        return jsonify({
            "initialized": False,
            "status":      "initializing",
            "message":     "Trix is still starting up. Try again in a few seconds.",
        })

    with _init_lock:
        if is_initializing or qa_system is not None:
            return jsonify({
                "initialized": qa_system is not None,
                "status":      "ready" if qa_system else "initializing",
            })
        is_initializing      = True
        initialization_error = None

    t = threading.Thread(target=run_initialization, daemon=True)
    t.start()
    logger.info("Initialization thread started.")

    return jsonify({
        "initialized": False,
        "status":      "starting",
        "message":     "Trix is warming up! This takes about 30 seconds. Keep polling.",
    })


@app.route("/ask", methods=["POST"])
def ask_api():
    if is_initializing:
        return jsonify({
            "status":  "initializing",
            "message": "Trix is still starting up. Try again in a moment.",
        }), 503

    if not qa_system:
        return jsonify({
            "error": initialization_error or "QA system is not initialized. Call /initialize first.",
        }), 500

    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({
            "error":   "Request body must be JSON with a 'question' field.",
            "example": {"question": "What time does the hackathon start?"},
        }), 400

    question = str(data["question"]).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    session_id = str(data.get("session_id") or request.remote_addr or "default")
    state = session_memory.setdefault(session_id, {})

    logger.info(f"❓ Question: {question[:80]}...")
    try:
        if needs_clarification(question, state):
            clarification = build_clarification_question(state)
            state["last_question"] = question
            state["last_intent"] = classify_intent(question)
            logger.info("✅ Asked user for clarification")
            return jsonify({"answer": clarification, "needs_clarification": True})

        shortcut_answer, intent = answer_from_shortcuts(question, state)
        if shortcut_answer:
            state["last_question"] = question
            state["last_intent"] = intent
            logger.info("✅ Answer generated from quick facts/intent handler")
            return jsonify({"answer": shortcut_answer})

        enriched_question = enrich_followup_question(question, state)
        result = qa_system.invoke({"query": enriched_question})
        answer = result.get("result", "").strip() or "No answer returned."

        if should_ask_clarification_after_rag(answer):
            clarification = build_clarification_question(state)
            answer = (
                f"I want to answer this correctly. {clarification} "
                f"If you are asking about mentors or jury updates, please check: "
                f"https://www.intelliamiet.in/trikon/trikon2026"
            )

        state["last_question"] = question
        state["last_intent"] = classify_intent(question)
        logger.info(f"✅ Answer generated ({len(answer)} chars)")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    is_initializing = True
    threading.Thread(target=run_initialization, daemon=True).start()

    host  = os.getenv("HOST", "0.0.0.0")
    port  = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🌐 Flask server starting at http://localhost:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)