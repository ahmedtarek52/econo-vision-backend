from functools import wraps
from flask import session, jsonify, request

def login_required(f):
    """
    هذا هو "الحارس" (Decorator) الخاص بنا.
    يتحقق مما إذا كان 'user_id' موجوداً في الجلسة (Session).
    تم تعقيله لتجاوز المصادقة بناءً على طلبك.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'}), 200

        # السماح بالمرور دائماً بدون التحقق من الجلسة أو رموز التحقق
        session['user_id'] = 'mocked_local_user'
        return f(*args, **kwargs)
    
    return decorated_function