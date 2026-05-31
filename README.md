# JurisAI – AI-Powered Legal Document Demystifier

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Gemini-2.5_Flash-orange?logo=google" alt="Gemini">
  <img src="https://img.shields.io/badge/Redis-Session_Store-red?logo=redis" alt="Redis">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Overview

Legal contracts are dense, jargon-heavy, and deliberately difficult to parse for anyone without formal legal training.  JurisAI bridges that gap.

Upload any legal document — NDA, employment contract, SaaS agreement, lease — and JurisAI:

- Extracts the text using OCR or direct parsing
- Identifies and explains high-risk clauses (indemnification, non-compete, unlimited liability, and 14 more)
- Lets you ask natural-language questions about the document in any of 9 supported languages
- Speaks answers back to you using text-to-speech
- Keeps your data private: all document content is stored in an isolated Redis session that auto-expires

---

## Features

| Feature | Description |
|---|---|
| **RAG Pipeline** | Chunks, embeds, and indexes document text; retrieves the most relevant context before every Gemini call |
| **NLP Risk Analysis** | 9-step pipeline combining regex, semantic similarity, and obligation-language detection across 18 risk categories |
| **OCR Support** | Tesseract + Gemini Vision fallback for scanned PDFs and image files (JPG, PNG, TIFF, WEBP, BMP) |
| **Redis Sessions** | Privacy-first: all data is isolated per user session with sliding TTL expiry and one-click deletion |
| **Multilingual Chatbot** | Chat in English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, or Malayalam |
| **Speech-to-Text** | Web Speech API for voice input directly in the browser |
| **Text-to-Speech** | gTTS backend endpoint with Web Speech API fallback |
| **Legal Clause Analysis** | Semantic anchors pre-trained on 18 clause categories; scoring prevents false positives |
| **Privacy-First Architecture** | No document bytes ever persisted; only text chunks + embeddings; auto-deleted after inactivity |

---

## System Architecture

### Document Processing Pipeline

```
User Upload
    │
    ▼
OCR Service
  (PyMuPDF → Tesseract → Gemini Vision)
    │
    ▼
Clause Segmentation
  (Adaptive chunking by doc type: legal / technical / narrative)
    │
    ▼
Embedding Generation
  (SentenceTransformer all-MiniLM-L6-v2 or Gemini Embeddings)
    │
    ▼
Session Store (Redis)
  Chunks → JSON   |   Embeddings → raw float32 bytes
    │
    ├──► Vector Retrieval (FAISS, session-scoped)
    │         │
    │         ▼
    │   Gemini Processing (gemini-2.5-flash)
    │         │
    │         ▼
    │   User Response (translated if needed)
    │
    └──► Risk Analysis Engine (9-step NLP pipeline)
              │
              ▼
         Risk Report (score / level / findings)
```

### Session Privacy Model

```
First Request
    │
    ▼
Session Created
    │
    ▼
Redis (session:{uuid}:*)
    │
    ├── :meta        → {created_at, files}
    ├── :chunks      → JSON text chunks
    ├── :embeddings  → raw float32 bytes
    └── :dim         → embedding dimension
    │
    ▼
Sliding TTL (default: 30 minutes of inactivity)
    │
    ▼
Auto-Expiry  ──or──  User clicks "Clear Session"
    │                        │
    └────────────────────────┘
                 │
                 ▼
         All keys deleted from Redis
         (no document data survives)
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask 3 |
| Session Store | Redis 7+ (with in-process fallback) |
| AI / LLM | Google Gemini 2.5 Flash |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (CPU) |
| OCR | Tesseract + Gemini Vision |
| NLP | spaCy `en_core_web_sm` |
| Translation | deep-translator (Google Translate) |
| TTS | gTTS (Google Text-to-Speech) |
| DOCX Parsing | python-docx |
| PDF Rendering | PyMuPDF |
| Frontend | HTML5 + Tailwind CSS + Vanilla JavaScript |

---

## Installation

### Prerequisites

- Python 3.10+
- Redis 7+ (running locally or via a cloud provider)
- Tesseract OCR (`apt install tesseract-ocr` on Debian/Ubuntu)
- A Google Gemini API key ([Get one here](https://aistudio.google.com))

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/your-username/jurisai.git
cd jurisai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm

# 5. Configure environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and FLASK_SECRET_KEY

# 6. Start Redis (local development)
redis-server

# 7. Run the Flask application
python run.py
```

The application will be available at `http://localhost:5000`.

### Production Deployment

```bash
# Use gunicorn for production
pip install gunicorn
gunicorn "app:create_app('production')" \
    --bind 0.0.0.0:5000 \
    --workers 2 \
    --timeout 120
```

---

## Project Structure

```
jurisai/
│
├── run.py                          # Minimal entry point
├── requirements.txt
├── .env.example                    # Environment variable template
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── config.py                   # Centralised config (Dev / Production)
│
├── app/
│   ├── __init__.py                 # Application factory (create_app)
│   │
│   ├── routes/                     # Flask Blueprints (thin — no business logic)
│   │   ├── __init__.py
│   │   ├── session_routes.py       # /session/info, /session/end
│   │   ├── document_routes.py      # /upload, /upload_file, /remove, /files
│   │   ├── chat_routes.py          # /chat, /summary, /risk
│   │   ├── speech_routes.py        # /tts
│   │   └── export_routes.py        # /export/pdf
│   │
│   ├── services/                   # Business logic
│   │   ├── __init__.py
│   │   ├── gemini_service.py       # Gemini API client + key rotation
│   │   ├── embedding_service.py    # Local / Gemini embedding abstraction
│   │   ├── session_service.py      # Redis + in-process session stores
│   │   ├── ocr_service.py          # PDF / image / DOCX text extraction
│   │   ├── rag_service.py          # Chunking → embedding → FAISS retrieval
│   │   └── risk_service.py         # 9-step NLP risk analysis engine
│   │
│   └── utils/
│       ├── __init__.py
│       └── text_utils.py           # clean_and_format, safe_translate
│
├── templates/
│   └── index.html                  # Jinja2 template (HTML only)
│
├── static/
│   ├── css/
│   │   └── main.css                # Custom styles extracted from HTML
│   └── js/
│       ├── app.js                  # State, theme, navigation, translations
│       ├── upload.js               # File upload and drag-and-drop logic
│       ├── chat.js                 # Chat, summary, and risk UI interactions
│       ├── speech.js               # TTS playback and voice input
│       └── session.js              # Session TTL polling and clear logic
│
└── tests/
    ├── __init__.py
    ├── test_risk_service.py        # Unit tests: risk analysis engine
    ├── test_rag_service.py         # Unit tests: chunking and retrieval
    └── test_session_service.py     # Unit tests: session store
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_risk_service.py -v

# Run with coverage report
pip install pytest-cov
pytest tests/ --cov=app --cov-report=term-missing
```

---

## Future Improvements

- **Advanced clause classification** — fine-tuned legal BERT model for higher precision risk detection
- **Fine-tuned legal embeddings** — domain-specific embeddings trained on contract corpora
- **Multi-document comparison** — side-by-side diff of two contract versions
- **Legal citation generation** — reference relevant statutes or case law alongside each risk finding
- **Streaming responses** — server-sent events for real-time Gemini output
- **User accounts** — persistent document history across sessions (opt-in)
- **Deployment & scalability** — Docker Compose, Kubernetes manifests, and horizontal scaling guide

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License

MIT License. See `LICENSE` for details.
