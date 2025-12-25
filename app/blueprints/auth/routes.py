from flask import Blueprint, request, jsonify, session, current_app
from firebase_admin import auth, firestore
import traceback
import time 
import datetime 

# --- (!!!) (إضافة جديدة: استيراد الحارس) (!!!) ---
# (استخدمنا ".." للرجوع من 'auth' إلى 'app')
from app.decorators import login_required
# --- (!!!) (نهاية الإضافة) (!!!) ---


# --- إعداد الـ Blueprint ---
auth_bp = Blueprint('auth_bp', __name__)

# --- دالة معالجة الأخطاء (كما هي) ---
def handle_auth_error(e):
    """ يعالج أخطاء Firebase Authentication بشكل مخصص """
    error_message = str(e)
    code = 400 # Bad Request (خطأ من المستخدم)

    print("--- AUTH ERROR ---")
    traceback.print_exc()
    print("------------------")

    if "EMAIL_EXISTS" in error_message:
        user_message = "This email is already registered."
    elif "WEAK_PASSWORD" in error_message:
        user_message = "Password is too weak. Must be at least 6 characters."
    elif "INVALID_ID_TOKEN" in error_message:
        user_message = "Invalid or expired session token. Please log in again."
        code = 401 # Unauthorized
    elif "USER_NOT_FOUND" in error_message:
        user_message = "No user found with this email."
    elif "INVALID_PASSWORD" in error_message:
        user_message = "Invalid password."
    elif "MAX_SESSIONS_REACHED" in error_message:
        user_message = "Maximum number of allowed devices (4) reached. Please log out from another device."
        code = 403 # Forbidden
    else:
        user_message = "An unexpected authentication error occurred."
        code = 500 # Server Error
    
    return jsonify({"error": user_message}), code
# --- (نهاية) ---


# --- 1. Endpoint: إنشاء حساب جديد (!!!) (مُعدل) (!!!) ---
@auth_bp.route('/register', methods=['POST'])
def register_user():
    """
    يستقبل (email, password) وينشئ مستخدماً جديداً في Firebase.
    (مُعدل) يقوم بإنشاء "مشروع" فارغ للمستخدم في Firestore.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            raise ValueError("Email and password are required.")

        # 1. إنشاء المستخدم في Firebase Authentication
        user = auth.create_user(email=email, password=password)
        
        # 2. (اختياري، لكن موصى به) إنشاء مستند للمستخدم في Firestore
        try:
            db = firestore.client()
            user_doc_ref = db.collection('users').document(user.uid)
            
            # (!!!) (إضافة جديدة) إنشاء بيانات المستخدم الأولية
            user_data = {
                'email': user.email,
                'created_at': firestore.SERVER_TIMESTAMP,
                'active_sessions': [] # (مهم جداً: تمت الإضافة هنا)
            }
            transaction = db.transaction()
            
            @firestore.transactional
            def create_user_and_project(transaction, user_ref):
                # إنشاء مستند المستخدم
                transaction.set(user_ref, user_data)
                
                # إنشاء "مشروع افتراضي" فارغ له
                project_ref = user_ref.collection('projects').document('default_project')
                empty_project = {
                    "analysisData": {
                        "filename": "",
                        "fullDataset": [],
                        "columns": [],
                        "transformationHistory": [],
                        "isPanel": False,
                        "panelIdVar": "",
                        "panelTimeVar": ""
                    },
                    "comparisonBasket": []
                }
                transaction.set(project_ref, empty_project)

            create_user_and_project(transaction, user_doc_ref)
            print(f"Firestore document and default project created for user: {user.uid}")
            # --- (!!!) (نهاية الإضافة) (!!!) ---

        except Exception as db_e:
            print(f"Warning: User {user.uid} created in Auth, but failed to create Firestore document: {db_e}")
            # (لا توقف العملية، يمكن للمستخدم تسجيل الدخول)

        return jsonify({
            "uid": user.uid, 
            "email": user.email,
            "message": "User created successfully."
        }), 201

    except Exception as e:
        return handle_auth_error(e)


# --- 2. Endpoint: إنشاء جلسة (Session) للباك إند (!!!) (مُعدل) (!!!) ---
@auth_bp.route('/login-session', methods=['POST'])
def create_session():
    """
    (مُعدل)
    يتحقق من الـ Token، ويفحص عدد الجلسات النشطة، وينشئ جلسة Flask.
    يسمح بحد أقصى 4 جلسات نشطة.
    """
    MAX_CONCURRENT_SESSIONS = 4 # (الحد الأقصى الذي طلبته)

    try:
        id_token = request.json.get('idToken')
        if not id_token:
            raise ValueError("ID Token is required.")

        # 1. التحقق من الـ Token
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        token_exp_timestamp = decoded_token['exp'] # تاريخ انتهاء هذا التوكن
        
        # 2. فحص الجلسات في Firestore
        db = firestore.client()
        user_doc_ref = db.collection('users').document(uid)
        
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
        current_timestamp = int(current_time_utc.timestamp())

        @firestore.transactional
        def update_sessions_in_transaction(transaction, doc_ref):
            snapshot = doc_ref.get(transaction=transaction)
            
            if not snapshot.exists:
                # (حالة نادرة: المستخدم موجود في Auth وليس في Firestore)
                transaction.set(doc_ref, {
                    'email': decoded_token.get('email', 'N/A'),
                    'active_sessions': [token_exp_timestamp] # إضافة الجلسة الأولى
                })
                return True # (مسموح بالدخول)
            
            # 3. تنظيف الجلسات القديمة (منتهية الصلاحية)
            
            # (!!!) (هذا هو التصحيح للخطأ المطبعي) (!!!)
            # تم حذف 'a' الزائدة من 'aall_sessions'
            all_sessions = snapshot.to_dict().get('active_sessions') or []
            # (!!!) (نهاية التصحيح) (!!!)
            
            fresh_sessions = [
                exp_time for exp_time in all_sessions 
                if isinstance(exp_time, (int, float)) and exp_time > current_timestamp
            ]

            # 4. فحص الحد الأقصى
            if len(fresh_sessions) >= MAX_CONCURRENT_SESSIONS:
                if token_exp_timestamp in fresh_sessions:
                        return True # (مسموح، هو نفس الجهاز يجدد جلسته)
                else:
                    print(f"Session limit reached for user {uid}. Found {len(fresh_sessions)} active sessions.")
                    return False # (غير مسموح بالدخول)
            
            # 5. (غير ممتلئ) أضف الجلسة الجديدة
            if token_exp_timestamp not in fresh_sessions:
                fresh_sessions.append(token_exp_timestamp)
            
            transaction.update(doc_ref, {'active_sessions': fresh_sessions})
            return True # (مسموح بالدخول)

        # 6. تشغيل الـ Transaction
        transaction = db.transaction()
        is_allowed = update_sessions_in_transaction(transaction, user_doc_ref)
        
        if not is_allowed:
            raise ValueError("MAX_SESSIONS_REACHED")

        # 7. (إذا نجح كل شيء) إنشاء جلسة Flask
        session.clear()
        session['user_id'] = uid
        session.permanent = True 
        
        # (تصحيح) pd.Timedelta لا يعمل هنا، استخدم datetime.timedelta
        current_app.permanent_session_lifetime = datetime.timedelta(days=14) 
        
        print(f"Flask session created successfully for user: {uid}")
        
        return jsonify({"status": "success", "uid": uid}), 200

    except Exception as e:
        return handle_auth_error(e)


# --- 3. Endpoint: تسجيل الخروج (لا تغيير) ---
@auth_bp.route('/logout', methods=['POST'])
def logout():
    """
    يقوم بتدمير الجلسة الآمنة (Flask Session) من الباك إند.
    """
    try:
        user_id = session.get('user_id')
        session.clear()
        print(f"Flask session cleared for user: {user_id}")
        return jsonify({"message": "Logged out successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- 4. Endpoint: التحقق من الجلسة (لا تغيير) ---
@auth_bp.route('/check-session', methods=['GET'])
def check_session():
    """
    يتحقق هل لا تزال هناك جلسة (Session) صالحة في الباك إند.
    """
    if 'user_id' in session:
        return jsonify({"isAuthenticated": True, "uid": session['user_id']}), 200
    else:
        return jsonify({"isAuthenticated": False}), 401

# --- (!!!) (إضافة جديدة: حفظ المشروع) (!!!) ---
@auth_bp.route('/save-project', methods=['POST'])
@login_required # (مؤمن)
def save_project_data():
    """
    يستقبل حالة المشروع (analysisData, comparisonBasket) من الفرونت إند
    ويحفظها في Firestore.
    """
    try:
        user_id = session['user_id'] # مضمون بسبب @login_required
        data = request.get_json()
        
        if 'analysisData' not in data or 'comparisonBasket' not in data:
            raise ValueError("Missing 'analysisData' or 'comparisonBasket' in payload.")

        analysis_data = data['analysisData']
        comparison_basket = data['comparisonBasket']

        # (مهم جداً) تحذير بشأن حجم البيانات
        # Firestore يحد المستند بـ 1MB. سنحاول الحفظ
        # إذا كان fullDataset كبيراً جداً، سيفشل هذا الطلب
        # (الحل المستقبلي هو استخدام Firebase Storage للبيانات)
        
        db = firestore.client()
        project_ref = db.collection('users').document(user_id).collection('projects').document('default_project')
        
        # (سنحفظ كل شيء، بما في ذلك fullDataset)
        project_data = {
            "analysisData": analysis_data,
            "comparisonBasket": comparison_basket,
            "last_saved": firestore.SERVER_TIMESTAMP
        }
        
        project_ref.set(project_data) # (استخدم set للكتابة فوق القديم)
        
        print(f"Project saved successfully for user: {user_id}")
        return jsonify({"status": "success", "message": "Project saved."}), 200

    except Exception as e:
        return handle_error(e, "Failed to save project data.")
# --- (!!!) (نهاية الإضافة) (!!!) ---


# --- (!!!) (إضافة جديدة: تحميل المشروع) (!!!) ---
@auth_bp.route('/load-project', methods=['GET'])
@login_required # (مؤمن)
def load_project_data():
    """
    يقوم بتحميل آخر حالة مشروع محفوظة للمستخدم من Firestore.
    """
    try:
        user_id = session['user_id'] # مضمون بسبب @login_required
        
        db = firestore.client()
        project_ref = db.collection('users').document(user_id).collection('projects').document('default_project')
        
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            # (حالة نادرة: إذا تم حذف المشروع)
            print(f"No default project found for user: {user_id}. Returning empty state.")
            return jsonify({
                "analysisData": {"fullDataset": [], "transformationHistory": [], "columns": []},
                "comparisonBasket": []
            }), 200 # إرجاع 200 مع بيانات فارغة

        project_data = project_doc.to_dict()

        # (تنظيف البيانات قبل إرسالها)
        # إزالة 'last_saved' إذا كان موجوداً
        project_data.pop('last_saved', None) 
        
        print(f"Project loaded successfully for user: {user_id}")
        return jsonify(project_data), 200

    except Exception as e:
        return handle_error(e, "Failed to load project data.")
# --- (!!!) (نهاية الإضافة) (!!!)