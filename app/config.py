import os
from dotenv import load_dotenv
from pathlib import Path
import datetime # (جديد) لاستخدامه في مدة الجلسة

BASE_DIR = Path(__file__).resolve().parent.parent
# load .env if present, but DO NOT overwrite existing env vars (production won't be overwritten)
load_dotenv(BASE_DIR / '.env', override=False)

class Config:
    # --- المفتاح السري (كما هو) ---
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-super-secret-key-CHANGE-ME'
    
    # --- مجلدات وإعدادات البيئة (كما هي) ---
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
    FLASK_ENV = os.getenv('FLASK_ENV', 'production')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # --- (!!!) (هذا هو التعديل الأهم لحل مشكلة 401) (!!!) ---
    
    # 1. جعل الجلسات دائمة (حتى يتمكن المتصفح من حفظها)
    SESSION_PERMANENT = True
    
    # 2. تحديد مدة الجلسة (مثلاً 14 يوم)
    PERMANENT_SESSION_LIFETIME = datetime.timedelta(days=14)

    # 3. منع JavaScript من قراءة "الكوكي"
    SESSION_COOKIE_HTTPONLY = True
    
    # 4. تفعيل الكوكي الآمنة في الإنتاج (Render يستخدم HTTPS)
    SESSION_COOKIE_SECURE = os.getenv('FLASK_ENV', 'production') == 'production'
    
    # 5. السماح بإرسال الكوكي عبر الدومينات المختلفة (cross-site) في الإنتاج ليعمل مع Vercel
    SESSION_COOKIE_SAMESITE = 'None' if os.getenv('FLASK_ENV', 'production') == 'production' else 'Lax'
    
    # 6. إلغاء حصر الدومين بـ localhost في الإنتاج
    SESSION_COOKIE_DOMAIN = None if os.getenv('FLASK_ENV', 'production') == 'production' else 'localhost'

    # --- (!!!) (نهاية التعديل) (!!!) ---