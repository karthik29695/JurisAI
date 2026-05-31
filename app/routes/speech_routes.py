"""
app/routes/speech_routes.py  —  Text-to-Speech endpoint.
"""
import logging
from io import BytesIO

from flask import Blueprint, jsonify, request, send_file

from app.utils.text_utils import clean_and_format

log = logging.getLogger("jurisai.routes.speech")

speech_bp = Blueprint("speech", __name__)

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False


@speech_bp.route("/tts", methods=["POST"])
def tts_api():
    """Convert text to MP3 audio using gTTS."""
    if not GTTS_AVAILABLE:
        return jsonify({"error": "gTTS not installed. Run: pip install gtts"}), 500

    data = request.get_json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en").lower()

    if not text:
        return jsonify({"error": "No text"}), 400

    text = clean_and_format(text[:4000])
    try:
        tts = gTTS(text=text, lang=lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3",
        )
    except Exception as exc:
        return jsonify({"error": f"TTS failed: {exc}"}), 500
