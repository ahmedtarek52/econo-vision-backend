from functools import wraps
from flask import session, jsonify, request

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

        if 'user_id' not in session:
            # يمكننا طباعة رسالة في السيرفر للتأكد
            # print(f"Access denied: No user_id in session. Path: {request.path}")
            return jsonify({"error": "Authentication required. Please log in."}), 401
        
        # إذا كان المستخدم مسجلاً دخوله، قم بتشغيل الدالة الأصلية
        return f(*args, **kwargs)
    
    return decorated_function