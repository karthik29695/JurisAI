"""
run.py  —  JurisAI entry point.

Start the development server:
    python run.py

For production, use gunicorn:
    gunicorn "app:create_app('production')" --bind 0.0.0.0:5000 --workers 2
"""
import os
from app import create_app

env = os.environ.get("FLASK_ENV", "development")
app = create_app(env)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=app.config["DEBUG"], host="0.0.0.0", port=port)
