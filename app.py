# app.py
import os
import io
import threading
import logging
from typing import List, Tuple
import requests
import re
import itertools

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

import numpy as np
import faiss
import fitz  # PyMuPDF
from PIL import Image

# Optional OCR
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

# LLM (Gemini) for generation + OCR fallback (if available)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Translation + TTS
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO

# Local embedding fallback (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBED_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    LOCAL_EMBED_AVAILABLE = False

# ---------------- Config & Logging ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("legal-assistant")

app = Flask(__name__, template_folder="templates", static_folder=None)
CORS(app)

# Max upload size (set to 30 MB; adjust if needed)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

# Gemini config (if you have an API key)
GENAI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
USE_GEMINI_FOR_EMBEDS = os.environ.get("USE_GEMINI_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
if GENAI_AVAILABLE:
    if GENAI_API_KEY:
        try:
            genai.configure(api_key=GENAI_API_KEY)
            log.info("Gemini client configured.")
        except Exception as e:
            log.warning("Failed initial genai.configure: %s", e)
    else:
        log.info("Gemini library available but no API key found. Generation/OCR with Gemini will fail unless key provided.")
else:
    log.info("google.generativeai not installed / not available. Gemini features disabled.")

# Models
EMBED_MODEL = os.environ.get("EMBED_MODEL", "models/embedding-001")
CHAT_MODEL_NAME = os.environ.get("CHAT_MODEL", "gemini-1.5-flash")  # used only if GENAI available

# Try to create CHAT_MODEL wrapper only if genai available
CHAT_MODEL = None
if GENAI_AVAILABLE and GENAI_API_KEY:
    try:
        CHAT_MODEL = genai.GenerativeModel(CHAT_MODEL_NAME)
    except Exception as e:
        log.warning("Could not initialize CHAT_MODEL: %s", e)
        CHAT_MODEL = None

# Local embedder
LOCAL_EMBED_MODEL_NAME = os.environ.get("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")
local_embedder = None
if LOCAL_EMBED_AVAILABLE:
    try:
        local_embedder = SentenceTransformer(LOCAL_EMBED_MODEL_NAME)
        log.info("Local embedding model loaded: %s", LOCAL_EMBED_MODEL_NAME)
    except Exception as e:
        log.warning("Failed to load local embedding model: %s", e)
        local_embedder = None
        LOCAL_EMBED_AVAILABLE = False

# If local embed is available, prefer local embeddings (no quota problems).
# You can force using Gemini embeddings by setting USE_GEMINI_FOR_EMBEDS=True (and providing API key).
if LOCAL_EMBED_AVAILABLE:
    PREFER_LOCAL_EMBEDS = True
else:
    PREFER_LOCAL_EMBEDS = False

if USE_GEMINI_FOR_EMBEDS:
    log.info("Environment requested using Gemini for embeddings.")
    PREFER_LOCAL_EMBEDS = False

# ---------------- Gemini Key Rotation ----------------
# Accept comma-separated keys in GEMINI_API_KEYS env var.
GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
key_cycle = itertools.cycle(GEMINI_KEYS) if GEMINI_KEYS else None
_current_key = None
if key_cycle:
    _current_key = next(key_cycle)
    if GENAI_AVAILABLE and _current_key:
        try:
            genai.configure(api_key=_current_key)
            log.info("Configured Gemini with first key from GEMINI_API_KEYS.")
        except Exception as e:
            log.warning("Initial gemini.configure with GEMINI_API_KEYS failed: %s", e)

def configure_gemini(key: str):
    """Set gemini client key for the genai library (if available)."""
    if not GENAI_AVAILABLE:
        return
    try:
        genai.configure(api_key=key)
        log.info("Gemini configured to new key (rotated).")
    except Exception as e:
        log.warning("configure_gemini failed: %s", e)

def call_gemini_generate(inputs, model_name=CHAT_MODEL_NAME, tries=None):
    """
    Wrapper for model.generate_content with key rotation on error.
    inputs: prompt string or list of inputs compatible with genai.GenerativeModel.generate_content
    """
    global _current_key
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini client not available.")
    if GEMINI_KEYS:
        tries = tries or len(GEMINI_KEYS)
    else:
        tries = tries or 1

    last_exc = None
    for _ in range(tries):
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(inputs)
            return resp
        except Exception as e:
            last_exc = e
            log.warning("Gemini generate_content error: %s", e)
            if key_cycle:
                try:
                    _current_key = next(key_cycle)
                    configure_gemini(_current_key)
                    log.info("Rotated Gemini key and retrying generate_content.")
                except Exception as e2:
                    log.warning("Failed to rotate key during generate_content retry: %s", e2)
            else:
                # no rotation possible, break
                break
    raise RuntimeError(f"Gemini generate_content failed after {tries} tries. Last error: {last_exc}")

def call_gemini_embed(model: str, content, task_type: str = "retrieval_document", tries=None):
    """
    Wrapper for genai.embed_content with key rotation on error.
    model: embedding model string
    content: single string or list of strings
    """
    global _current_key
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini client not available.")
    if GEMINI_KEYS:
        tries = tries or len(GEMINI_KEYS)
    else:
        tries = tries or 1

    last_exc = None
    for _ in range(tries):
        try:
            res = genai.embed_content(model=model, content=content, task_type=task_type)
            return res
        except Exception as e:
            last_exc = e
            log.warning("Gemini embed_content error: %s", e)
            if key_cycle:
                try:
                    _current_key = next(key_cycle)
                    configure_gemini(_current_key)
                    log.info("Rotated Gemini key and retrying embed_content.")
                except Exception as e2:
                    log.warning("Failed to rotate key during embed_content retry: %s", e2)
            else:
                break
    raise RuntimeError(f"Gemini embed_content failed after {tries} tries. Last error: {last_exc}")

# ---------------- In-memory state & thread-safety ----------------
index_lock = threading.Lock()
index = None      # faiss index; will init dynamically when we know DIM
DIM = None
chunks: List[str] = []
uploaded_pdfs: List[dict] = []  # {"filename":..., "chunks":[ ... ]}

# ---------------- Helpers ----------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Simple word-based chunking. Returns list of chunk strings."""
    words = text.split()
    out = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        out.append(" ".join(words[start:end]))
        start += max(1, chunk_size - overlap)
    return out

def init_faiss(dim: int):
    """Initialize global FAISS index with the provided dimension."""
    global index, DIM
    with index_lock:
        if index is None:
            DIM = dim
            index = faiss.IndexFlatL2(DIM)
            log.info("Initialized FAISS index with dim=%d", DIM)
        else:
            if DIM != dim:
                # Recreate index if dim changed (shouldn't usually happen)
                index = faiss.IndexFlatL2(dim)
                DIM = dim
                log.info("Recreated FAISS index with new dim=%d", DIM)

def add_embeddings_to_index(embeddings: np.ndarray, texts: List[str]):
    """Add embeddings (n x dim) and append text chunks to store in thread-safe manner."""
    global index, chunks
    if embeddings is None or len(texts) == 0:
        return
    n, d = embeddings.shape
    if index is None:
        init_faiss(d)
    if d != DIM:
        # Re-init for safety
        init_faiss(d)

    with index_lock:
        index.add(embeddings.astype("float32"))
        chunks.extend(texts)
        log.info("Added %d vectors to index (total now %d).", n, index.ntotal)

# ---- Embedding: batch-aware, Gemini fallback --> local ----
def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of strings and return (n, dim) numpy float32 array.
    Strategy:
      1) If PREFER_LOCAL_EMBEDS and local model available -> use it.
      2) Else try Gemini batching (if available).
      3) On any failure or ResourceExhausted, fallback to local embedder if available.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    # Ensure we always return a 2D numpy array
    # 1) Local embeddings (fast, no quota)
    if PREFER_LOCAL_EMBEDS and local_embedder is not None:
        try:
            embs = local_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            arr = np.asarray(embs, dtype="float32")
            log.info("Embedded %d texts using local model (dim=%d).", len(texts), arr.shape[1])
            return arr
        except Exception as e:
            log.warning("Local embedding failed: %s", e)
            # Fall through to try GEMINI if available

    # 2) Try GEMINI (batched)
    if GENAI_AVAILABLE and (GENEI_API_KEY := (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or _current_key)):
        all_embs = []
        try:
            # batch and call genai.embed_content with list of texts
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                try:
                    # Use rotation-aware wrapper
                    res = call_gemini_embed(model=EMBED_MODEL, content=batch, task_type="retrieval_document")
                    # Attempt to parse response: handle multiple response shapes
                    batch_embs = []
                    if isinstance(res, dict):
                        # dict form: maybe 'embeddings' key
                        if "embeddings" in res:
                            for e in res["embeddings"]:
                                # e may be dict with 'embedding' or 'values'
                                if isinstance(e, dict) and "embedding" in e:
                                    batch_embs.append(np.array(e["embedding"], dtype="float32"))
                                elif isinstance(e, dict) and "values" in e:
                                    batch_embs.append(np.array(e["values"], dtype="float32"))
                                else:
                                    # fallback: whole object
                                    batch_embs.append(np.array(e, dtype="float32"))
                        elif "embedding" in res:
                            batch_embs.append(np.array(res["embedding"], dtype="float32"))
                        else:
                            # try to coerce numeric arrays
                            batch_embs.append(np.array(res, dtype="float32"))
                    else:
                        # object with attribute .embeddings
                        emb_list = getattr(res, "embeddings", None)
                        if emb_list is not None:
                            for e in emb_list:
                                vals = getattr(e, "values", None) or getattr(e, "embedding", None)
                                if vals is not None:
                                    batch_embs.append(np.array(vals, dtype="float32"))
                                else:
                                    batch_embs.append(np.array(e, dtype="float32"))
                        else:
                            # try to coerce
                            batch_embs.append(np.array(res, dtype="float32"))

                    # extend
                    for be in batch_embs:
                        all_embs.append(be)
                except Exception as e:
                    # If any batch fails (e.g., quota), raise and fall back
                    log.exception("Gemini embedding batch failed: %s", e)
                    raise e

            if len(all_embs) > 0:
                arr = np.vstack([np.asarray(a, dtype="float32") for a in all_embs])
                log.info("Embedded %d texts using Gemini (dim=%d).", arr.shape[0], arr.shape[1])
                return arr
        except Exception as e:
            log.warning("Gemini embeddings failed or quota hit; falling back to local if available. (%s)", e)

    # 3) Final fallback: local embedder if available
    if local_embedder is not None:
        try:
            embs = local_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            arr = np.asarray(embs, dtype="float32")
            log.info("Fallback: embedded %d texts using local model (dim=%d).", len(texts), arr.shape[1])
            return arr
        except Exception as e:
            log.exception("Local fallback embedding also failed: %s", e)

    # If we reach here, no embedding available -> raise error
    raise RuntimeError("No embedding method available. Install sentence-transformers or provide Gemini API key.")

# ---------- Translation ----------
SUPPORTED_LANGS = {"en", "hi", "te", "ta", "mr"}  # extend anytime
def safe_translate(text: str, target_lang: str) -> str:
    target = (target_lang or "en").lower()
    if not text:
        return text
    if target == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target).translate(text) or text
    except Exception:
        return text

# ---------- OCR fallback via Gemini ----------
def ocr_with_gemini(pil_image: Image.Image) -> str:
    """Use Gemini (image -> text) only if CHAT_MODEL is available."""
    if CHAT_MODEL is None and not GEMINI_KEYS:
        return ""
    try:
        # Use the rotation-safe call for generation
        prompt = "Extract all textual content from this image. Return only the plain text in reading order. No commentary."
        resp = call_gemini_generate([prompt, pil_image])
        return (resp.text or "").strip()
    except Exception as e:
        log.warning("Gemini OCR failed: %s", e)
        return ""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF pages; if page text is short, rasterize and OCR."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text_parts = []
    for page in doc:
        txt = (page.get_text("text") or "").strip()
        if len(txt) >= 40:
            all_text_parts.append(txt)
            continue

        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            png_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_bytes))
        except Exception:
            all_text_parts.append(txt)
            continue

        ocr_text = ""
        if HAVE_TESS:
            try:
                # If you install tesseract language packs, set the LANGS accordingly
                ocr_text = pytesseract.image_to_string(img, lang="eng")
            except Exception:
                ocr_text = ""

        if not ocr_text.strip():
            ocr_text = ocr_with_gemini(img)

        all_text_parts.append((ocr_text or "").strip())

    doc.close()
    return "\n\n".join([p for p in all_text_parts if p])

def clean_and_format(text: str) -> str:
    """Remove Gemini markdown and format into readable bullet points."""
    if not text:
        return ""

    # Remove bold/italics/markdown markers
    text = re.sub(r'\*{1,3}', '', text)   # remove *, **, ***
    text = re.sub(r'[_`]', '', text)      # remove _, `
    text = re.sub(r'#+', '', text)        # remove ###

    # Normalize spacing
    text = text.replace("\r\n", "\n")
    # Convert triple newlines to single
    text = re.sub(r'\n{2,}', '\n', text).strip()

    # Replace common list markers with bullets
    text = re.sub(r'^\s*-\s*', '• ', text, flags=re.M)
    text = re.sub(r'^\s*\*\s*', '• ', text, flags=re.M)

    return text.strip()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global uploaded_pdfs

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = file.filename
    log.info("Received upload: %s", filename)

    try:
        file_bytes = file.read()
        text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        log.exception("Failed to read PDF")
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    if not text.strip():
        return jsonify({"error": f"No text found in '{filename}'. (OCR may have failed)"}), 400

    # chunk
    new_chunks = chunk_text(text)
    log.info("PDF split into %d chunks", len(new_chunks))

    # Batch embed and add to index in safe batches
    BATCH_SIZE = 32  # adjust (bigger reduces API calls, but also more memory)
    try:
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[i:i+BATCH_SIZE]
            emb = embed_texts(batch, batch_size=BATCH_SIZE)  # returns (n, dim)
            add_embeddings_to_index(emb, batch)
    except Exception as e:
        log.exception("Embedding failed for upload")
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    uploaded_pdfs.append({"filename": filename, "chunks": new_chunks})
    return jsonify({
        "message": f"PDF '{filename}' processed, You can start asking questions.",
        "filename": filename
    })

@app.route("/remove", methods=["POST"])
def remove_pdf():
    global uploaded_pdfs
    data = request.get_json(force=True)
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    uploaded_pdfs = [p for p in uploaded_pdfs if p["filename"] != filename]
    # rebuild index
    with index_lock:
        # clear and rebuild
        log.info("Rebuilding index after removing %s", filename)
        # clear global state
        global index, chunks, DIM
        index = None
        DIM = None
        chunks = []
        # rebuild from uploaded_pdfs
        for pdf in uploaded_pdfs:
            # embed in batches
            for i in range(0, len(pdf["chunks"]), 32):
                batch = pdf["chunks"][i:i+32]
                emb = embed_texts(batch, batch_size=32)
                add_embeddings_to_index(emb, batch)

    return jsonify({"message": f"PDF '{filename}' removed successfully."})

@app.route("/files", methods=["GET"])
def list_files():
    return jsonify({"files": [p["filename"] for p in uploaded_pdfs]})

@app.route("/summary", methods=["GET"])
def summary():
    lang = request.args.get("lang", "en")
    if not chunks:
        msg = "No documents uploaded yet."
        return jsonify({"summary": safe_translate(msg, lang)})

    sample = "\n\n".join(chunks[:8])
    prompt = (
        "You are a helpful legal assistant. Summarize the following document excerpts "
        "in clear, concise bullet points (5-8 bullets). Focus on key obligations, dates, "
        "amounts, parties, and risks. Keep it brief and readable.\n\n"
        f"Context:\n{sample}"
    )
    try:
        if CHAT_MODEL is None and not GEMINI_KEYS:
            # fallback: simple local summary (very basic): return the sample truncated
            return jsonify({"summary": safe_translate(sample[:1500] + ("..." if len(sample) > 1500 else ""), lang)})
        # Use rotation-aware call for generation
        resp = call_gemini_generate(prompt)
        summ = (resp.text or "").strip()
        cleaned = clean_and_format(summ)
        return jsonify({"summary": safe_translate(cleaned, lang)})
    except Exception as e:
        log.exception("Summary generation failed")
        return jsonify({"summary": safe_translate(f"⚠️ Could not generate summary: {str(e)}", lang)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("message", "")
    lang = data.get("lang", "en")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    smalltalk = {"hello", "hey", "hi", "how are you", "what's up", "good morning", "good evening"}
    if question.strip().lower() in smalltalk:
        prompt = f"You are a friendly assistant. The user said: '{question}'. Reply casually and warmly."
    else:
        retrieved = []
        if index is not None and index.ntotal > 0:
            try:
                q_emb = embed_texts([question], batch_size=1)  # (1, dim)
                q_emb = q_emb.reshape(1, -1).astype("float32")
                k = min(5, index.ntotal)
                with index_lock:
                    D, I = index.search(q_emb, k)
                retrieved = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
            except Exception as e:
                log.exception("RAG retrieval failed")
                retrieved = []

        context = "\n\n".join(retrieved)
        prompt = (
            "Use the following context to answer the user's question. "
            "If the answer is not present in the context, provide a general answer that is still relevant, "
            "in simple language.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    try:
        if CHAT_MODEL is None and not GEMINI_KEYS:
            # Simple fallback: return combined context or a short canned reply
            reply = (context[:1500] + ("\n\n(Answer generation unavailable — Gemini key missing.)")) if context else "Answer generation currently unavailable (no Gemini model configured)."
            return jsonify({"reply": safe_translate(reply, lang), "lang": lang})
        # rotation-aware generation call
        response = call_gemini_generate(prompt)
        reply = (response.text or "").strip()
        reply = clean_and_format(reply)
        reply = safe_translate(reply, lang)
        return jsonify({"reply": reply, "lang": lang})
    except Exception as e:
        log.exception("Chat generation failed")
        err = safe_translate(f"Server error: {str(e)}", lang)
        return jsonify({"error": err}), 500

@app.route("/tts", methods=["POST"])
def tts_api():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        lang = data.get("lang", "en").lower()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Optional safety limit
        if len(text) > 4000:
            text = text[:4000] + " ..."
        text = clean_and_format(text)

        # Generate TTS into memory
        tts_obj = gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts_obj.write_to_fp(buf)   # ✅ fixed
        buf.seek(0)

        return send_file(
            buf,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3"
        )

    except Exception as e:
        app.logger.exception("TTS failed")
        return jsonify({"error": f"TTS failed: {str(e)}"}), 500

@app.route("/export/pdf")
def export_pdf():
    body = "No documents uploaded yet."
    if chunks:
        sample = "\n\n".join(chunks[:6])
        prompt = f"Summarize succinctly in bullet points:\n\n{sample}"
        try:
            if CHAT_MODEL or GEMINI_KEYS:
                # use rotation-aware generation if possible
                resp = call_gemini_generate(prompt)
                body = (resp.text or "Summary not available.").strip()
            else:
                body = sample[:1500]
        except Exception:
            body = "Summary not available."

    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 550, 792 - 50)
    page.insert_textbox(rect, body, fontsize=11, lineheight=1.2)
    pdf_bytes = doc.tobytes()
    doc.close()
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True, download_name="summary.pdf")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    log.info("Starting Legal Document Assistant backend.")
    log.info("Local embeddings available: %s", bool(local_embedder))
    log.info("Gemini generation available: %s", CHAT_MODEL is not None or bool(GEMINI_KEYS))
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
