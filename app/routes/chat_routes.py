"""
app/routes/chat_routes.py  —  Chat, summary, and risk analysis endpoints.
"""
import logging

from flask import Blueprint, jsonify, request, current_app

from .session_routes import get_or_create_session_id
from app.utils.text_utils import clean_and_format, safe_translate

log = logging.getLogger("jurisai.routes.chat")

chat_bp = Blueprint("chat", __name__)

_SMALLTALK = {"hello", "hey", "hi", "how are you", "what's up", "good morning", "good evening"}


@chat_bp.route("/summary", methods=["GET"])
def summary():
    sid = get_or_create_session_id()
    lang = request.args.get("lang", "en")
    gemini = current_app.gemini_service

    chunks, _ = current_app.session_store.get_document(sid)
    if not chunks:
        return jsonify({"summary": safe_translate("No documents uploaded yet.", lang)})

    sample = "\n\n".join(chunks[:8])
    prompt = (
        "You are a helpful legal assistant. Summarize the following document excerpts "
        "in clear, concise bullet points (5-8 bullets). Focus on key obligations, dates, "
        "amounts, parties, and risks. Keep it brief and readable.\n\n"
        f"Context:\n{sample}"
    )
    try:
        if not gemini.available:
            return jsonify({"summary": safe_translate(sample[:1500], lang)})
        resp = gemini.generate(prompt)
        cleaned = clean_and_format((resp.text or "").strip())
        return jsonify({"summary": safe_translate(cleaned, lang)})
    except Exception as exc:
        log.exception("Summary failed")
        return jsonify({"summary": safe_translate(f"⚠️ {exc}", lang)}), 500


@chat_bp.route("/chat", methods=["POST"])
def chat():
    sid = get_or_create_session_id()
    data = request.get_json(force=True)
    question = data.get("message", "").strip()
    lang = data.get("lang", "en")
    gemini = current_app.gemini_service

    if not question:
        return jsonify({"error": "No question provided"}), 400

    if question.lower() in _SMALLTALK:
        prompt = f"You are a friendly legal assistant. The user said: '{question}'. Reply warmly and briefly."
        context = ""
    else:
        retrieved = current_app.rag_service.retrieve(sid, question, top_k=5)
        context = "\n\n".join(retrieved)
        prompt = (
            "Use the following context from the user's uploaded document to answer the question. "
            "If the answer is not present, provide a helpful general legal answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    try:
        if not gemini.available:
            reply = (
                context[:1500] + "\n\n(Gemini unavailable — showing raw context)"
                if context
                else "Answer generation unavailable — no Gemini key configured."
            )
            return jsonify({"reply": safe_translate(reply, lang), "lang": lang})

        resp = gemini.generate(prompt)
        reply = clean_and_format((resp.text or "").strip())
        return jsonify({"reply": safe_translate(reply, lang), "lang": lang})
    except Exception as exc:
        log.exception("Chat failed")
        return jsonify({"error": safe_translate(f"Server error: {exc}", lang)}), 500


@chat_bp.route("/risk", methods=["GET"])
def risk_analysis():
    sid = get_or_create_session_id()
    lang = request.args.get("lang", "en")

    chunks, _ = current_app.session_store.get_document(sid)
    if not chunks:
        return jsonify({"error": safe_translate("No documents uploaded yet.", lang)}), 400

    try:
        result = current_app.risk_service.analyze(chunks)
        if lang != "en":
            result["summary"] = safe_translate(result["summary"], lang)
            for f in result["findings"]:
                f["explanation"] = safe_translate(f["explanation"], lang)
        return jsonify(result)
    except Exception as exc:
        log.exception("Risk analysis failed")
        return jsonify({"error": safe_translate(f"Risk analysis failed: {exc}", lang)}), 500
