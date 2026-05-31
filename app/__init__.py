"""
app/__init__.py  —  Flask application factory.

Usage
─────
    from app import create_app
    app = create_app()           # uses config/config.py defaults
    app = create_app("production")
"""
import logging

from flask import Flask, render_template
from flask_cors import CORS

from config import config_map
from app.services import (
    GeminiService,
    EmbeddingService,
    create_session_store,
    OCRService,
    RAGService,
    RiskService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("jurisai")


def create_app(env: str = "default") -> Flask:
    """Construct and configure the Flask application."""
    cfg = config_map.get(env, config_map["default"])

    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(cfg)

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS(
        app,
        origins=cfg.ALLOWED_ORIGINS,
        supports_credentials=True,
        allow_headers=["Content-Type", "ngrok-skip-browser-warning", "X-Session-ID"],
        expose_headers=["Content-Type"],
        methods=["GET", "POST", "OPTIONS"],
    )

    # ── Services (attached to app so routes can access via current_app) ───────
    gemini = GeminiService(
        primary_key=cfg.GOOGLE_API_KEY,
        key_pool=cfg.GEMINI_API_KEYS,
        chat_model_name=cfg.CHAT_MODEL_NAME,
        embed_model=cfg.EMBED_MODEL,
    )
    app.gemini_service = gemini

    embedder = EmbeddingService(
        local_model_name=cfg.LOCAL_EMBED_MODEL,
        prefer_local=not cfg.USE_GEMINI_EMBEDDINGS,
        gemini_service=gemini,
    )
    app.embedding_service = embedder

    session_store = create_session_store(cfg.REDIS_URL, cfg.SESSION_TTL_SECONDS)
    app.session_store = session_store

    app.ocr_service = OCRService(gemini_service=gemini)

    rag = RAGService(embedding_service=embedder, session_store=session_store)
    app.rag_service = rag

    risk = RiskService(embedding_service=embedder)
    app.risk_service = risk

    # ── Blueprints ────────────────────────────────────────────────────────────
    from app.routes import session_bp, document_bp, chat_bp, speech_bp, export_bp

    app.register_blueprint(session_bp)
    app.register_blueprint(document_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(speech_bp)
    app.register_blueprint(export_bp)

    # ── Home route ────────────────────────────────────────────────────────────
    @app.route("/")
    def home():
        from app.routes.session_routes import get_or_create_session_id
        sid = get_or_create_session_id()
        log.info("Home — session: %s", sid)
        return render_template("index.html")

    # ── Startup log ───────────────────────────────────────────────────────────
    log.info("JurisAI starting (env=%s).", env)
    log.info("Session store    : %s", type(session_store).__name__)
    log.info("Session TTL      : %ds", cfg.SESSION_TTL_SECONDS)
    log.info("Gemini available : %s", gemini.available)

    return app
