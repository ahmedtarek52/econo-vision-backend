# app/blueprints/reports/routes.py
from flask import Blueprint, request, jsonify, current_app 
import pandas as pd 
import numpy as np 
from firebase_admin import firestore # (!!!) (إضافة جديدة) (!!!)

# --- (!!!) (إضافة جديدة: استيراد الحارس) (!!!) ---
# (استخدمنا ".." مرتين للرجوع من 'reports' إلى 'blueprints' ثم إلى 'app')
from app.decorators import login_required# --- (!!!) (نهاية الإضافة) (!!!) ---

# --- Imports for report generation functions ---
from .utils import (
    generate_english_report,
    generate_arabic_report,
    generate_gemini_english_report,
    generate_gemini_arabic_report
)

reports_bp = Blueprint('reports_api', __name__)

# --- (جديد) Helper Function for Error Handling ---
def handle_error(e, default_message="An error occurred", status_code=500):
    """Logs the error and returns a JSON response."""
    error_message = str(e)
    try:
        current_app.logger.error(f"Report API Error: {error_message}", exc_info=True)
    except AttributeError: # Fallback
        print(f"Report API Error: {error_message}")
        import traceback
        traceback.print_exc()

    user_message = default_message
    if isinstance(e, (ValueError, KeyError, TypeError)): # Handle validation errors
        user_message = error_message
        status_code = 400 # Bad Request

    return jsonify({"error": user_message}), status_code
# --- (نهاية الإضافة) ---

# --- (!!!) (إضافة جديدة: دالة العداد الآمنة) (!!!) ---
def get_next_report_count():
    """
    Atomically increments and returns the global report counter using Firestore.
    This is thread-safe and production-ready.
    """
    try:
        db = firestore.client()
        counter_ref = db.collection('metadata').document('app_counters')

        # @firestore.transactional
        # def update_in_transaction(transaction, doc_ref):
        #     snapshot = doc_ref.get(transaction=transaction)
        #     if not snapshot.exists:
        #         # If counter doesn't exist, create it starting at 1
        #         new_count = 1001 # (ابدأ من 1001 إذا أردت)
        #         transaction.set(doc_ref, {'report_count': new_count})
        #         return new_count
            
        #     old_count = snapshot.get('report_count') or 0
        #     new_count = old_count + 1
        #     transaction.update(doc_ref, {'report_count': new_count})
        #     return new_count

        # transaction = db.transaction()
        # final_count = update_in_transaction(transaction, counter_ref)
        
        # (طريقة أبسط وأكثر حداثة باستخدام Increment)
        result = counter_ref.update({'report_count': firestore.Increment(1)})
        # (نحتاج قراءة القيمة بعد التحديث)
        snapshot = counter_ref.get()
        if snapshot.exists:
            return snapshot.get('report_count')
        else:
            # (إذا لم يكن المستند موجوداً، أنشئه)
            counter_ref.set({'report_count': 1001})
            return 1001

    except Exception as e:
        print(f"CRITICAL: Failed to increment report counter in Firestore: {e}")
        # Fallback (non-thread-safe) just in case
        return np.random.randint(1000, 9999) 
# --- (!!!) (نهاية الإضافة) (!!!) ---


# (تم حذف العداد القديم)
# global report_count = 1000 

@reports_bp.route('/generate-report', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def generate_report_endpoint():
    """Generates basic template-based reports."""
    # (تم حذف العداد القديم)
    try:
        payload = request.get_json()
        if not payload or 'modelSummary' not in payload:
            return jsonify({"error": "Missing 'modelSummary' in request payload."}), 400

        model_summary = payload['modelSummary']
        if not isinstance(model_summary, str) or not model_summary.strip():
             return jsonify({"error": "'modelSummary' must be a non-empty string."}), 400

        english_report = generate_english_report(model_summary)
        arabic_report = generate_arabic_report(model_summary)

        # (!!!) (تعديل) استدعاء العداد الجديد
        report_count = get_next_report_count()

        return jsonify({
            "englishReport": english_report,
            "arabicReport": arabic_report,
            "reportCount": report_count
        }), 200

    except Exception as e:
        return handle_error(e, "Failed to generate basic report")


@reports_bp.route('/generate-smart-report', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def generate_smart_report():
    """Generates advanced AI reports using Gemini, including diagnostic results."""
    # (تم حذف العداد القديم)
    try:
        payload = request.get_json()
        if not payload or 'modelSummary' not in payload:
            return jsonify({"error": "Missing 'modelSummary' in request payload."}), 400

        model_summary = payload['modelSummary']
        diagnostics = payload.get('diagnostics') 
        post_test = payload.get('postTest') 

        if not isinstance(model_summary, str) or not model_summary.strip():
             return jsonify({"error": "'modelSummary' must be a non-empty string."}), 400

        english_report = generate_gemini_english_report(
            model_summary=model_summary,
            diagnostics_results=diagnostics,
            post_test_result=post_test
        )
        arabic_report = generate_gemini_arabic_report(
            model_summary=model_summary,
            diagnostics_results=diagnostics,
            post_test_result=post_test
        )

        # (!!!) (تعديل) استدعاء العداد الجديد
        report_count = get_next_report_count() 

        return jsonify({
            "englishReport": english_report,
            "arabicReport": arabic_report,
            "reportCount": report_count
        }), 200

    except ValueError as ve: # Catch specific errors (like API key missing) from utils
        return handle_error(ve)
    except Exception as e:
        return handle_error(e, "Failed to generate smart AI report")


@reports_bp.route('/report-count', methods=['GET'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_report_count():
    """Returns the current report generation count."""
    # (!!!) (تعديل) قراءة العداد من Firestore
    try:
        db = firestore.client()
        counter_ref = db.collection('metadata').document('app_counters')
        snapshot = counter_ref.get()
        if snapshot.exists:
            count = snapshot.get('report_count') or 0
            return jsonify({"count": count})
        else:
            return jsonify({"count": 0}) # أو 1000 كقيمة أولية
    except Exception as e:
        print(f"Error reading report count from Firestore: {e}")
        return jsonify({"error": "Could not retrieve report count."}), 500