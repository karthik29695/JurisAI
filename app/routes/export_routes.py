"""
app/routes/export_routes.py  —  Document export endpoints.
"""
import io
import logging

from flask import Blueprint, current_app, send_file

from .session_routes import get_or_create_session_id

log = logging.getLogger("jurisai.routes.export")

export_bp = Blueprint("export", __name__)

try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False


@export_bp.route("/export/pdf")
def export_pdf():
    """Generate and download a PDF summary of the current session documents."""
    if not FITZ_AVAILABLE:
        from flask import jsonify
        return jsonify({"error": "PyMuPDF not installed."}), 500

    sid = get_or_create_session_id()
    gemini = current_app.gemini_service
    chunks, _ = current_app.session_store.get_document(sid)

    body = "No documents uploaded yet."
    if chunks:
        sample = "\n\n".join(chunks[:6])
        try:
            if gemini.available:
                resp = gemini.generate(f"Summarize succinctly in bullet points:\n\n{sample}")
                body = (resp.text or "Summary not available.").strip()
            else:
                body = sample[:1500]
        except Exception:
            body = sample[:1500]

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 742), body, fontsize=11, lineheight=1.2)
    pdf_bytes = doc.tobytes()
    doc.close()

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="summary.pdf",
    )
