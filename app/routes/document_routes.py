"""
app/routes/document_routes.py  —  Document upload, removal, and listing endpoints.
"""
import os
import logging

from flask import Blueprint, jsonify, request, current_app

from .session_routes import get_or_create_session_id, session_guard
from app.services.ocr_service import IMAGE_EXTS, DOCX_EXTS

log = logging.getLogger("jurisai.routes.document")

document_bp = Blueprint("document", __name__)


@document_bp.route("/upload", methods=["POST"])
def upload_pdf():
    """Accept a PDF, extract text, chunk+embed, store in session."""
    sid = get_or_create_session_id()
    err = session_guard(sid)
    if err:
        return err

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files accepted on this endpoint."}), 400

    filename = file.filename
    log.info("[%s] PDF upload: %s", sid[:8], filename)

    try:
        file_bytes = file.read()
        text = current_app.ocr_service.extract_from_pdf(file_bytes)
    except Exception as exc:
        log.exception("PDF extraction failed")
        return jsonify({"error": f"Failed to read PDF: {exc}"}), 500

    if not text.strip():
        return jsonify({"error": f"No text found in '{filename}'."}), 400

    try:
        stats = current_app.rag_service.ingest(sid, filename, text)
    except Exception as exc:
        log.exception("Embedding/store failed")
        return jsonify({"error": f"Embedding failed: {exc}"}), 500

    return jsonify({
        "message":     f"PDF '{filename}' processed. You can start asking questions.",
        "filename":    filename,
        "chunks":      stats["chunks"],
        "session_id":  sid,
        "session_ttl": current_app.session_store.ttl_seconds(sid),
    })


@document_bp.route("/upload_file", methods=["POST"])
def upload_file():
    """Accept an image or DOCX, extract text, chunk+embed, store in session."""
    sid = get_or_create_session_id()
    err = session_guard(sid)
    if err:
        return err

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in IMAGE_EXTS and ext not in DOCX_EXTS:
        return jsonify({"error": f"Unsupported type '{ext}'. Use /upload for PDFs."}), 400

    log.info("[%s] File upload: %s (ext=%s)", sid[:8], filename, ext)

    try:
        file_bytes = file.read()
        if ext in IMAGE_EXTS:
            text = current_app.ocr_service.extract_from_image(file_bytes, filename)
            file_type = "image (OCR)"
        else:
            text = current_app.ocr_service.extract_from_docx(file_bytes)
            file_type = "Word document"
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        log.exception("Extraction failed for '%s'", filename)
        return jsonify({"error": f"Text extraction failed: {exc}"}), 500

    if not text.strip():
        return jsonify({"error": f"No text found in '{filename}'."}), 400

    try:
        stats = current_app.rag_service.ingest(sid, filename, text)
    except Exception as exc:
        log.exception("Embedding/store failed")
        return jsonify({"error": f"Embedding failed: {exc}"}), 500

    return jsonify({
        "message":     f"'{filename}' processed as {file_type}. You can start asking questions.",
        "filename":    filename,
        "file_type":   file_type,
        "chunks":      stats["chunks"],
        "session_id":  sid,
        "session_ttl": current_app.session_store.ttl_seconds(sid),
    })


@document_bp.route("/remove", methods=["POST"])
def remove_file():
    sid = get_or_create_session_id()
    data = request.get_json(force=True)
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    store = current_app.session_store
    chunks, embs = store.get_document(sid)
    if chunks is None:
        return jsonify({"message": "Nothing to remove."}), 200

    files = store.get_files(sid)
    if len(files) <= 1:
        store.delete_session(sid)
        store.create_session(sid)
        from flask import session as flask_session
        flask_session["session_id"] = sid
    else:
        store.delete_session(sid)
        store.create_session(sid)
        from flask import session as flask_session
        flask_session["session_id"] = sid

    log.info("[%s] Removed file '%s'", sid[:8], filename)
    return jsonify({"message": f"'{filename}' removed. Session reset."})


@document_bp.route("/files", methods=["GET"])
def list_files():
    sid = get_or_create_session_id()
    files = current_app.session_store.get_files(sid)
    return jsonify({"files": files})
