import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# load .env if present, but DO NOT overwrite existing env vars (production won't be overwritten)
load_dotenv(BASE_DIR / '.env', override=False)

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-super-secret-key'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')