from functools import wraps
from flask import session, jsonify, request
from firebase_admin import auth

def login_required(f):
    """
    هذا هو "الحارس" (Decorator) الخاص بنا.
    يتحقق مما إذا كان 'user_id' موجوداً في الجلسة (Session).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # (!!!) هام جداً: السماح لطلبات الفحص المسبق (OPTIONS) بالمرور
        # المتصفح يرسل هذا الطلب أولاً بدون الكوكيز للتحقق من الأمان
        # إذا لم نضع هذا الشرط، سيرد الخادم بـ 401 وسيفشل طلب الـ CORS
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        # 1. Check Session (Primary method)
        if 'user_id' in session:
            return f(*args, **kwargs)

        # 2. Check Authorization Header (Fallback method)
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                decoded_token = auth.verify_id_token(token)
                session['user_id'] = decoded_token['uid']
                return f(*args, **kwargs)
            except Exception as e:
                print(f"Token verification failed: {e}")
                return jsonify({"error": "Invalid or expired token"}), 401

        return jsonify({"error": "Authentication required. Please log in."}), 401
    
    return decorated_function