# app/blueprints/security/routes.py
import json
from flask import Blueprint, request, jsonify, current_app

security_bp = Blueprint('security_api', __name__)

# --- (تمت استعادة دالة المساعد) ---
def _build_cors_preflight_response():
    """مساعد للرد على طلبات CORS Preflight"""
    response = jsonify({"status": "preflight-ok"})
    # (مهم) يجب أن تسمح بهذه الهيدرز والأصل
    response.headers.add("Access-Control-Allow-Origin", "*") # أو حدد الدومين الخاص بك
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    # (إضافة السماح بالهيدر المطلوب لـ Reporting API)
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, X-Content-Type-Options") 
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response
# --- (نهاية استعادة الدالة) ---

@security_bp.route('/report-violation', methods=['POST', 'OPTIONS'])
def handle_report_violation():
    """
    هذا هو الـ Endpoint الذي يستقبل تقارير أخطاء الأمان (CSP) 
    والأكواد المهملة (Deprecations) من متصفحات المستخدمين.
    """
    
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    try:
        reports = None
        # القراءة اليدوية لـ reports+json للتغلب على مشاكل المتصفح
        if request.content_type == 'application/reports+json':
            # قراءة البيانات كـ bytes ثم محاولة تحميلها كـ JSON
            reports_data = request.data.decode('utf-8')
            if reports_data:
                 reports = json.loads(reports_data)
        
        if reports:
            current_app.logger.warning(f"CSP/SECURITY REPORT RECEIVED: {json.dumps(reports, indent=2)}")
            # طباعة رسالة واضحة في الـ Terminal
            print(f"\n--- SECURITY ALERT: CSP Violation Detected ---")
            print(f"Violation URL: {reports[0].get('url', 'N/A')}")
            print(f"Blocked URL: {reports[0].get('body', {}).get('blockedURL', 'N/A')}")
            print(f"--------------------------------------------\n")
        else:
             current_app.logger.info("Empty or non-JSON report violation request received.")

        # الرد بـ 204 No Content لإخبار المتصفح أننا استلمنا بنجاح
        return jsonify({"status": "received"}), 204

    except json.JSONDecodeError:
        current_app.logger.error("Failed to decode reports+json payload.")
        return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400
    except Exception as e:
        current_app.logger.error(f"Error handling security report: {e}", exc_info=True)
        return jsonify({"status": "error"}), 500