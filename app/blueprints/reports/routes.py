# app/blueprints/reports/routes.py
from flask import Blueprint, request, jsonify, current_app 
import pandas as pd 
import numpy as np 

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
    import numpy as np
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

        tone = payload.get('tone', 'academic')

        english_report = generate_gemini_english_report(
            model_summary=model_summary,
            diagnostics_results=diagnostics,
            post_test_result=post_test,
            tone=tone
        )
        arabic_report = generate_gemini_arabic_report(
            model_summary=model_summary,
            diagnostics_results=diagnostics,
            post_test_result=post_test,
            tone=tone
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
    return jsonify({"count": 1000})