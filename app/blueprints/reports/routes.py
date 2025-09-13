# app/blueprints/reports/routes.py
from flask import Blueprint, request, jsonify
from .utils import generate_english_report, generate_arabic_report
from .utils import generate_gemini_english_report, generate_gemini_arabic_report

reports_bp = Blueprint('reports_bp', __name__)
report_count = 1000
@reports_bp.route('/generate-report', methods=['POST'])
def generate_report_endpoint():
    global report_count
    try:
        payload = request.get_json()
        
        # This endpoint will receive all the data collected so far
        # For this example, we'll just pass the model summary
        model_summary = payload.get('modelSummary')
        
        if not model_summary:
            return jsonify({"error": "Missing model summary for report generation."}), 400

        # Generate reports in both languages
        english_report = generate_english_report(model_summary)
        arabic_report = generate_arabic_report(model_summary)

        # Increment report count
        report_count += 1

        return jsonify({
            "englishReport": english_report,
            "arabicReport": arabic_report,
            "reportCount": report_count
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@reports_bp.route('/generate-smart-report', methods=['POST'])
def generate_smart_report():
    global report_count
    try:
        payload = request.get_json()
        
        # This endpoint will receive all the data collected so far
        # For this example, we'll just pass the model summary
        model_summary = payload.get('modelSummary')
        
        if not model_summary:
            return jsonify({"error": "Missing model summary for report generation."}), 400

        # Generate reports in both languages
        english_report = generate_gemini_english_report(model_summary)
        arabic_report = generate_gemini_arabic_report(model_summary)

        # Increment report count
        report_count += 1

        return jsonify({
            "englishReport": english_report,
            "arabicReport": arabic_report,
            "reportCount": report_count
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@reports_bp.route('/report-count', methods=['GET'])
def get_report_count():
    global report_count
    return jsonify({"count": report_count})