"""
config/config.py  —  JurisAI centralised configuration.
All environment-specific values are read here; no secrets in source code.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Flask ──────────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
    MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 MB
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "None"
    SESSION_COOKIE_SECURE = True           # required with SameSite=None

    # ── CORS ──────────────────────────────────────────────────────────────────
    _raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
    ALLOWED_ORIGINS: list = (
        [o.strip() for o in _raw_origins.split(",") if o.strip()]
        or ["*"]
    )

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 1800))

    # ── Gemini ────────────────────────────────────────────────────────────────
    GOOGLE_API_KEY = (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    GEMINI_API_KEYS: list = [
        k.strip()
        for k in os.getenv("GEMINI_API_KEYS", "").split(",")
        if k.strip()
    ]
    CHAT_MODEL_NAME = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")
    EMBED_MODEL = os.environ.get("EMBED_MODEL", "models/embedding-001")
    USE_GEMINI_EMBEDDINGS = os.environ.get(
        "USE_GEMINI_EMBEDDINGS", "false"
    ).lower() in ("1", "true", "yes")

    # ── Local embeddings ──────────────────────────────────────────────────────
    LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "all-MiniLM-L6-v2")

    # ── Server ────────────────────────────────────────────────────────────────
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() in ("1", "true", "yes")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
