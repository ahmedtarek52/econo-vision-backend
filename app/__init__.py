from flask import Flask
import firebase_admin
from firebase_admin import credentials, initialize_app
import os
from flask_cors import CORS

# (تم حذف استيراد Redis بناءً على طلبك السابق والإبقاء عليه كتعليق إذا أردت استرجاعه)
# from .extensions import redis_client 

def create_app():
    app = Flask(__name__)
    
    # --- إعدادات الأمان ---
    app.secret_key = os.environ.get('SECRET_KEY', 'default_secret_key_please_change_me')
    
    # --- (1) تفعيل CORS بشكل عام للتطبيق ---
    # يسمح للفرونت إند (localhost:5173) بالتحدث مع الباك إند
    # supports_credentials=True ضروري لإرسال الـ cookies/auth headers
    CORS(app, 
         supports_credentials=True, 
         origins=["http://localhost:5173"],
         allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept"],
         methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"]
    )

    # --- إعدادات Firebase ---
    try:
        # تحديد مسار ملف المفاتيح (Service Account)
        # يفترض أن الملف موجود في المجلد الرئيسي (root) بجانب مجلد app
        cred_path = os.path.join(os.path.dirname(app.instance_path), 'serviceAccountKey.json')
        
        # في حال كنت تشغل التطبيق ومسار الـ instance مختلف، قد تحتاج لتعديل المسار أعلاه
        # ليكون os.path.join(os.getcwd(), 'serviceAccountKey.json')
        if not os.path.exists(cred_path):
             # محاولة بديلة للمسار المباشر
             cred_path = 'serviceAccountKey.json'

        # التحقق مرة أخرى بعد المحاولة البديلة
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            
            # التحقق مما إذا كان Firebase قد تم تهيئته مسبقاً لتجنب الخطأ عند إعادة التشغيل
            if not firebase_admin._apps:
                initialize_app(cred)
                print(f"✅ Firebase App Initialized Successfully using {cred_path}.")
            else:
                print("ℹ️ Firebase App already initialized.")
        else:
            print("❌ CRITICAL: 'serviceAccountKey.json' NOT FOUND.")
            
    except Exception as e:
        print(f"CRITICAL: Failed to initialize Firebase Admin: {e}")
        try:
            print("Service Account Key Path Tried:", os.path.abspath(cred_path))
        except: pass
        print("!!! (Please ensure 'serviceAccountKey.json' exists in the root folder) !!!")

    # --- تعريف إعدادات CORS الموحدة للـ Blueprints ---
    # نستخدم هذا التكوين لضمان توحيد السياسات لكل الـ Endpoints
    cors_config = {
        "supports_credentials": True,
        "origins": ["http://localhost:5173"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "methods": ["GET", "POST", "OPTIONS"]
    }

    # ============================================================
    # تسجيل الـ Blueprints (وحدات التطبيق)
    # ============================================================

    # 1. المصادقة (Auth)
    try:
        from .blueprints.auth.routes import auth_bp
        # يتم تطبيق CORS هنا أيضاً لضمان عمل تسجيل الدخول
        CORS(auth_bp, resources={r"/api/auth/*": cors_config})
        app.register_blueprint(auth_bp, url_prefix='/api/auth')
    except ImportError:
        print("Warning: auth_bp (Auth) not found.")
        
    # 2. لوحة التحكم (Dashboard)
    try:
        from .blueprints.dashboard.routes import dashboard_bp 
        CORS(dashboard_bp, resources={r"/api/dashboard/*": cors_config})
        app.register_blueprint(dashboard_bp) 
    except ImportError:
        print("Warning: dashboard_bp (Dashboard) not found.")
        
    # 3. الذكاء الاصطناعي (AI Core)
    try:
        from .blueprints.ai_core.routes import ai_core_bp # تأكدت من اسم المجلد ai_core
        CORS(ai_core_bp, resources={r"/api/ai/*": cors_config})
        app.register_blueprint(ai_core_bp, url_prefix='/api/ai')
        print("✅ AI Core Blueprint Registered at /api/ai")
    except ImportError:
        # محاولة بديلة إذا كان الاسم مختلفاً
        try:
            from .blueprints.ai.routes import ai_bp 
            CORS(ai_bp, resources={r"/api/ai/*": cors_config})
            app.register_blueprint(ai_bp, url_prefix='/api/ai')
            print("✅ AI Blueprint (Legacy) Registered at /api/ai")
        except ImportError:
             print("Warning: ai_bp or ai_core_bp (AI) not found.")
        
    # 4. رفع البيانات (Data Upload)
    try:
        from .blueprints.data_upload.routes import data_upload_bp 
        CORS(data_upload_bp, resources={r"/api/data/*": cors_config})
        app.register_blueprint(data_upload_bp, url_prefix='/api/data')
    except ImportError:
        print("Warning: data_upload_bp (Data Upload) not found.")

    # 5. تجهيز البيانات (Data Preparation)
    try:
        from .blueprints.data_preparation.routes import data_preparation_bp
        CORS(data_preparation_bp, resources={r"/api/prepare/*": cors_config})
        app.register_blueprint(data_preparation_bp, url_prefix='/api/prepare')
    except ImportError:
        print("Warning: data_preparation_bp (Data Prep) not found.")

    # 6. (!!!) هام جداً: التحليل الإحصائي والاختبارات (Statistical Tests) (!!!)
    # هذا الجزء هو المسؤول عن مسار /api/analysis/pre/run-test
    try:
        from .blueprints.statistical_tests.routes import pre_analysis_bp
        CORS(pre_analysis_bp, resources={r"/api/analysis/pre/*": cors_config})
        app.register_blueprint(pre_analysis_bp, url_prefix='/api/analysis/pre')
        print("✅ Statistical Tests Blueprint Registered Successfully at /api/analysis/pre")
    except ImportError as e:
        print(f"Warning: pre_analysis_bp (Statistical Tests) not found. Error: {e}")

    # 7. النمذجة (Modeling)
    try:
        from .blueprints.modeling.routes import model_execution_bp
        CORS(model_execution_bp, resources={r"/api/model/*": cors_config})
        app.register_blueprint(model_execution_bp) # يبدو أنه لا يوجد prefix هنا أو معرف داخلياً
    except ImportError:
        print("Warning: model_execution_bp (Modeling) not found.")

    # 8. التقارير (Reports)
    try:
        from .blueprints.reports.routes import reports_bp
        CORS(reports_bp, resources={r"/api/report/*": cors_config})
        app.register_blueprint(reports_bp, url_prefix='/api/report')
    except ImportError:
        print("Warning: reports_bp (Reports) not found.")
        
    # 9. التواصل (Contact)
    try:
        from .blueprints.contact.routes import contact_bp
        CORS(contact_bp, resources={r"/api/contact/*": cors_config})
        app.register_blueprint(contact_bp, url_prefix='/api/contact')
    except ImportError:
        print("Warning: contact_bp (Contact) not found.")
    
    return app