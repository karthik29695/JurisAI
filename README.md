# JurisAI
Multilingual AI legal assistant for simplifying documents with voice and text.

Problem:
Legal documents in India are lengthy, complex, and often available only in English. Farmers, small business owners, and regional citizens struggle to understand policies, contracts, and legal terms due to language barriers and lack of expert guidance.

💡 Our Solution – JurisAI

JurisAI is a multilingual GenAI-powered assistant that helps citizens understand legal documents in simple, regional languages.

✅ Upload any policy / contract (PDF)
✅ Get concise, bullet-point summaries
✅ Ask questions in natural language
✅ Listen to answers with Text-to-Speech (TTS)
✅ Voice-based queries using Speech-to-Text (STT)
✅ Beautiful & simple web interface (Bootstrap + Flask)

🖼️ Demo Flow

Splash Screen 
Beautiful landing screen with interactive Get Started.

Upload PDF 
Drag & drop or select a legal document.

Summarize 
AI converts lengthy text → easy-to-read bullet points.

Ask Questions 
Chatbot answers in user’s language.

Voice + Speech Output 

Ask via microphone.

Listen via TTS.

🛠️ Tech Stack

Frontend → HTML, CSS, Bootstrap, JavaScript
Backend → Flask (Python)
AI Models → Google Gemini API (RAG + Summarization)
Speech → gTTS (Text-to-Speech), SpeechRecognition
PDF Parsing → PyMuPDF
Deployment Ready → GitHub + Flask

How to Run

Clone the repo:

git clone https://github.com/yourusername/JurisAI.git
cd JurisAI

Install dependencies:

pip install -r requirements.txt

Add your API key in .env:

GEMINI_API_KEY=your_api_key_here

Run the app:

flask run

Open in browser:
http://127.0.0.1:5000/
