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
TOP_K_CHUNKS    = 6
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
FALLBACK_TEXT   = "can you please be more specific so i can guide you"

TRIX_PROMPT_TEMPLATE = """
You are Trix, the official AI assistant and mascot of Trikon 2026 hackathon!
You look like a glowing, triangle-shaped robot with big expressive eyes and a cheerful smile.
Your tagline: "Trix won't trick you!"
Your vibe: Friendly, confident, and practical.

COMMUNICATION STYLE:
- Give useful, context-rich answers in 3 to 6 lines.
- Include concrete details whenever available (dates, times, venue, names, roles, phone/email/links).
- If a user asks about a person, include role plus available contact details.
- For list-like answers (timeline, rules, prizes), use short bullet points.
- Keep language natural and easy to understand.

YOUR RULES:
1. ONLY use the information given in the "Context" section below to answer.
2. If the user's message is unclear, too short, or missing details, ask one short clarifying question first.
3. If the answer is NOT in the context, say exactly:
    "can you please be more specific so i can guide you"
4. Never output placeholder text like "app support member" or "website support member". If exact details are missing, clearly say they are not listed.
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
    "organized_by": None,
    "app_contact": None,
    "website_contact": None,
    "emergency_contact": None,
}
person_profiles      = {}

CONTACT_DETAIL_PATTERN = re.compile(
    r"\b(contact|details?|detail|deatils|detais|phone|email|instagram|linkedin|github|reach|connect|number)\b",
    re.IGNORECASE,
)
PERSON_INFO_PATTERN = re.compile(
    r"\b(who\s+is|about|role|profile|member|coordinator|president|lead|team)\b",
    re.IGNORECASE,
)
ANY_MEMBER_PATTERN = re.compile(
    r"\b(any|anyone|someone|somebody)\b.*\b(member|contact|person)\b|\bany\s+member\b",
    re.IGNORECASE,
)


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


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
    # Debug note: keep organizer in quick_facts so "by whom / who organized" can be answered deterministically.
    facts["organized_by"] = find(r"Organized by:\s*([^\n]+)")
    facts["app_contact"] = (
        find(r"For any technical issue[^\n]*contact\s+([^\.\n]+)")
        or find(r"For any issue in the app,\s*contact\s+([^\.\n]+)")
        or find(r"For any app issue[^\n]*contact\s+([^\.\n]+)")
        or find(r"For any issue.*?contact\s+([^\n]+)")
    )
    facts["website_contact"] = (
        find(r"For any issue on the website,\s*contact\s+([^\.\n]+)")
        or find(r"For website issues,\s*contact\s+([^\.\n]+)")
    )
    facts["emergency_contact"] = (
        find(r"For any non-technical issue[^\n]*contact\s+([^\.\n]+)")
        or find(r"For emergencies,\s*contact\s+([^\.\n]+)")
        or find(r"For emergency[^\n]*contact\s+([^\.\n]+)")
    )
    return facts


def load_people_profiles(file_path):
    profiles = {}
    if not file_path or not os.path.exists(file_path):
        return profiles

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return profiles

    current = None

    def save_profile(profile):
        if not profile or not profile.get("name"):
            return
        key = normalize_text(profile["name"])
        profiles[key] = profile

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("="):
            continue

        if line.startswith("Name:"):
            save_profile(current)
            current = {"name": line.split(":", 1)[1].strip()}
            continue

        if not current:
            continue

        if line.startswith("Role:"):
            current["role"] = line.split(":", 1)[1].strip()
        elif line.startswith("Branch:"):
            current["branch"] = line.split(":", 1)[1].strip()
        elif line.startswith("Area:"):
            current["area"] = line.split(":", 1)[1].strip()
        elif line.startswith("Responsibilities:"):
            current["responsibilities"] = line.split(":", 1)[1].strip()
        elif line.startswith("Contribution:"):
            current["contribution"] = line.split(":", 1)[1].strip()
        elif line.startswith("Email:"):
            current["email"] = line.split(":", 1)[1].strip()
        elif line.startswith("Instagram:"):
            current["instagram"] = line.split(":", 1)[1].strip()
        elif line.startswith("LinkedIn:"):
            current["linkedin"] = line.split(":", 1)[1].strip()
        elif line.startswith("GitHub:"):
            current["github"] = line.split(":", 1)[1].strip()
        elif re.match(r"^Phone\s*(?:no)?\s*:", line, flags=re.IGNORECASE):
            current["phone"] = line.split(":", 1)[1].strip()

    save_profile(current)
    return profiles


def detect_person_reference(question):
    if not question or not person_profiles:
        return None

    q_norm = normalize_text(question)
    q_tokens = set(q_norm.split())

    best_key = None
    best_score = 0

    for key, profile in person_profiles.items():
        name = profile.get("name", "")
        name_tokens = [t for t in normalize_text(name).split() if len(t) >= 3]
        if not name_tokens:
            continue

        full_name = " ".join(name_tokens)
        score = 0

        if full_name and full_name in q_norm:
            score = 10 + len(name_tokens)
        else:
            matched = [t for t in name_tokens if t in q_tokens]
            if len(matched) == len(name_tokens) and len(name_tokens) > 1:
                score = 8 + len(name_tokens)
            elif len(matched) == 1 and matched[0] in {name_tokens[0], name_tokens[-1]}:
                score = 3

        if score > best_score:
            best_key = key
            best_score = score

    return best_key if best_score >= 3 else None


def format_person_response(profile, include_contact_details):
    name = profile.get("name", "This person")
    role = profile.get("role")
    branch = profile.get("branch")
    responsibilities = profile.get("responsibilities") or profile.get("contribution") or profile.get("area")

    intro = name
    if role:
        intro += f" is {role}"
    if branch:
        intro += f" ({branch})"
    intro += "."

    lines = [intro]
    if responsibilities:
        lines.append(f"Work focus: {responsibilities}")

    if include_contact_details:
        direct_contacts = []
        if profile.get("phone"):
            direct_contacts.append(f"Phone: {profile['phone']}")
        if profile.get("email"):
            direct_contacts.append(f"Email: {profile['email']}")
        if direct_contacts:
            lines.append("Direct contact: " + " | ".join(direct_contacts))

        social_links = []
        if profile.get("instagram"):
            social_links.append(f"Instagram: {profile['instagram']}")
        if profile.get("linkedin"):
            social_links.append(f"LinkedIn: {profile['linkedin']}")
        if profile.get("github"):
            social_links.append(f"GitHub: {profile['github']}")
        if social_links:
            lines.append("Profiles: " + " | ".join(social_links))

        if not direct_contacts and not social_links:
            lines.append("No direct contact details are listed in the current knowledge base.")

    return "\n".join(lines)


def split_contact_names(raw_contacts):
    if not raw_contacts:
        return []
    cleaned = re.sub(r"\s+", " ", raw_contacts).strip().strip(".")
    parts = re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", cleaned, flags=re.IGNORECASE)
    seen = set()
    names = []
    for part in parts:
        p = part.strip()
        if not p:
            continue
        key = normalize_text(p)
        if key in seen:
            continue
        seen.add(key)
        names.append(p)
    return names


def find_profile_by_name(name):
    if not name:
        return None

    target = normalize_text(name)
    if target in person_profiles:
        return person_profiles[target]

    target_tokens = set(target.split())
    best_profile = None
    best_score = 0

    for profile in person_profiles.values():
        profile_name = normalize_text(profile.get("name", ""))
        profile_tokens = set(profile_name.split())
        if not profile_tokens:
            continue

        score = len(target_tokens & profile_tokens)
        if score > best_score:
            best_profile = profile
            best_score = score

    return best_profile if best_score >= 1 else None


def build_contact_detail_cards(contact_names):
    cards = []
    for name in contact_names:
        profile = find_profile_by_name(name)
        if profile:
            cards.append(format_person_response(profile, include_contact_details=True))
        else:
            cards.append(f"{name}: Contact details are not listed in the current knowledge base.")
    return cards


def classify_intent(question):
    q = question.lower().strip()

    if re.search(r"\b(hi|hello|hey|yo)\b", q):
        return "greeting"
    if re.search(r"\b(thanks|thank you|thx)\b", q):
        return "thanks"

    person_key = detect_person_reference(q)
    if person_key and CONTACT_DETAIL_PATTERN.search(q):
        return "person_contact"
    if person_key and PERSON_INFO_PATTERN.search(q):
        return "person_info"
    if person_key:
        return "person_info"

    if re.search(r"\bfirst\s*prize\b", q):
        return "first_prize"
    if re.search(r"\bsecond\s*prize\b", q):
        return "second_prize"
    if re.search(r"\bthird\s*prize\b|\b3rd\s*prize\b", q):
        return "third_prize"
    if re.search(r"\bprize\b|\bcash\b", q):
        return "prize_general"

    # Debug note: detect contact-routing intents first.
    if re.search(r"\b(non[-\s]?technical|other|general)\b", q) and re.search(r"\b(contact|issue|help|support|whom|who)\b", q):
        return "contact_emergency"
    if re.search(r"\b(app|technical|tech)\b", q) and re.search(r"\b(contact|issue|help|support|whom|who)\b", q):
        return "contact_app"
    if re.search(r"\bweb\b|\bwebsite\b", q) and re.search(r"\b(contact|issue|help|support|whom|who)\b", q):
        return "contact_website"
    if re.search(r"\bemergency\b", q):
        return "contact_emergency"
    if re.search(r"\b(contact|who\s+to\s+contact|whom\s+to\s+contact)\b", q):
        return "contact_general"

    # Debug note: organizer intent is separate from support-contact intent.
    if re.search(r"\b(by whom|organized by|organised by|who organized|who organised|organizer|organiser)\b", q):
        return "organizer_query"

    # Debug note: very short follow-ups should use previous question context in RAG.
    if q in {"whom", "who", "by whom", "contact", "for this", "for that", "whom?", "who?", "by whom?"}:
        return "followup_short"

    return "general"


def answer_from_shortcuts(question, state):
    intent = classify_intent(question)
    last_intent = state.get("last_intent", "general")
    wants_contact_details = bool(CONTACT_DETAIL_PATTERN.search(question))
    asks_for_any_member = bool(ANY_MEMBER_PATTERN.search(question))

    if intent == "followup_short":
        # Debug note: do NOT default to contact_general, it causes wrong answers for follow-ups like "by whom".
        intent = last_intent if last_intent != "general" else "general"

    if intent == "greeting":
        return "Hey! I am Trix. Ask me anything about TRIKON schedule, rules, prizes, teams, or contacts. ✨", intent

    if intent == "thanks":
        return "Anytime! Ask me your next TRIKON question. 🎉", intent

    if intent == "organizer_query" and quick_facts.get("organized_by"):
        return f"TRIKON is organized by {quick_facts['organized_by']}.", intent

    person_key = detect_person_reference(question)
    if person_key:
        profile = person_profiles.get(person_key, {})
        wants_contacts = intent == "person_contact" or bool(CONTACT_DETAIL_PATTERN.search(question))
        reply = format_person_response(profile, include_contact_details=wants_contacts)
        return reply, "person_contact" if wants_contacts else "person_info"

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
        if wants_contact_details:
            names = split_contact_names(quick_facts["app_contact"])
            cards = build_contact_detail_cards(names)
            if cards:
                return (
                    "🛠️ For technical issues, contact:\n\n" + "\n\n".join(cards),
                    intent,
                )
        return f"🛠️ For technical issues, contact {quick_facts['app_contact']}.", intent

    if intent == "contact_website":
        if quick_facts.get("website_contact"):
            if wants_contact_details:
                names = split_contact_names(quick_facts["website_contact"])
                cards = build_contact_detail_cards(names)
                if cards:
                    return (
                        "🌐 For website issues, contact:\n\n" + "\n\n".join(cards),
                        intent,
                    )
            return f"🌐 For website issues, contact {quick_facts['website_contact']}.", intent

        # If dedicated website contact is missing, route website issues to technical support.
        if quick_facts.get("app_contact"):
            if wants_contact_details:
                names = split_contact_names(quick_facts["app_contact"])
                cards = build_contact_detail_cards(names)
                if cards:
                    return (
                        "🌐 Website issues are handled by technical support. Contact:\n\n"
                        + "\n\n".join(cards),
                        intent,
                    )
            return (
                f"🌐 Website issues are handled by technical support. "
                f"Contact {quick_facts['app_contact']}.",
                intent,
            )

    if intent == "contact_emergency" and quick_facts.get("emergency_contact"):
        if wants_contact_details:
            names = split_contact_names(quick_facts["emergency_contact"])
            cards = build_contact_detail_cards(names)
            if cards:
                return (
                    "🚨 For non-technical/general issues or emergencies, contact:\n\n"
                    + "\n\n".join(cards),
                    intent,
                )
        return (
            f"🚨 For non-technical or general issues (including emergencies), "
            f"contact {quick_facts['emergency_contact']}.",
            intent,
        )

    if intent == "contact_general":
        app_person = quick_facts.get("app_contact")
        web_person = quick_facts.get("website_contact")
        emergency = quick_facts.get("emergency_contact")

        if wants_contact_details or asks_for_any_member:
            response_blocks = []
            if app_person:
                tech_cards = build_contact_detail_cards(split_contact_names(app_person))
                if tech_cards:
                    response_blocks.append(
                        "🛠️ Technical support contacts:\n\n" + "\n\n".join(tech_cards)
                    )
            if emergency:
                nontech_cards = build_contact_detail_cards(split_contact_names(emergency))
                if nontech_cards:
                    response_blocks.append(
                        "🚨 Non-technical/general support contact:\n\n" + "\n\n".join(nontech_cards)
                    )
            if response_blocks:
                return "\n\n".join(response_blocks), intent

        parts = []
        if app_person:
            parts.append(f"For technical issues, contact {app_person}.")
        if web_person:
            parts.append(f"For website issues, contact {web_person}.")
        if emergency:
            parts.append(f"For non-technical/general issues or emergencies, contact {emergency}.")
        if parts:
            return " ".join(parts), intent

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

    if last_intent in {"person_info", "person_contact"}:
        return "Do you want role details, direct contact info, or social profile links for that person?"

    return "Can you tell me what category you mean: schedule, registration, rules, prizes, teams, or contacts?"


def should_ask_clarification_after_rag(answer):
    if not answer:
        return True

    normalized = answer.strip().lower()
    return normalized == FALLBACK_TEXT.lower() or "i don't have that info yet" in normalized


def run_initialization():
    global qa_system, is_initializing, initialization_error, quick_facts, person_profiles

    logger.info("━" * 60)
    logger.info("🚀 Starting Trix initialization...")
    logger.info("━" * 60)

    try:
        logger.info("[1/3] Loading knowledge base document...")
        quick_facts = load_quick_facts(DOCUMENT_PATH)
        person_profiles = load_people_profiles(DOCUMENT_PATH)
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
