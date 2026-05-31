"""
app/routes/session_routes.py  —  Session lifecycle endpoints.
"""
import uuid
import logging

from flask import Blueprint, jsonify, request, session, current_app

log = logging.getLogger("jurisai.routes.session")

session_bp = Blueprint("session", __name__)


def get_or_create_session_id() -> str:
    """
    Session ID resolution order:
    1. X-Session-ID request header  (cross-origin / Vercel → ngrok)
    2. Flask signed cookie           (same-origin, local dev)
    3. Create new session            (first visit)
    """
    store = current_app.session_store

    sid = request.headers.get("X-Session-ID")
    if sid and store.session_exists(sid):
        return sid

    sid = session.get("session_id")
    if sid and store.session_exists(sid):
        return sid

    sid = str(uuid.uuid4())
    session["session_id"] = sid
    session.permanent = False
    store.create_session(sid)
    log.info("New session created: %s", sid)
    return sid


def session_guard(sid: str):
    """Return an error response tuple if the session is invalid, else None."""
    if not sid:
        return jsonify({"error": "No active session. Please refresh."}), 401
    if not current_app.session_store.session_exists(sid):
        return jsonify({"error": "Session expired. Please refresh."}), 401
    return None


@session_bp.route("/session/info", methods=["GET"])
def session_info():
    sid = get_or_create_session_id()
    store = current_app.session_store
    ttl = store.ttl_seconds(sid)
    files = store.get_files(sid)
    return jsonify({
        "session_id":       sid,
        "session_id_short": sid[:8] + "…",
        "ttl_seconds":      ttl,
        "files":            files,
        "store_type":       type(store).__name__,
    })


@session_bp.route("/session/end", methods=["POST"])
def end_session():
    sid = session.get("session_id")
    if sid:
        current_app.session_store.delete_session(sid)
        log.info("Session explicitly ended: %s", sid[:8])
    session.clear()
    return jsonify({"message": "Session ended. All document data has been deleted."})
