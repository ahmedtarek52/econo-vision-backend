# app/blueprints/reports/routes.py
from flask import Blueprint, request, jsonify
from .utils import generate_english_report, generate_arabic_report

reports_bp = Blueprint('reports_bp', __name__)

@reports_bp.route('/generate-report', methods=['POST'])
def generate_report_endpoint():
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

        return jsonify({
            "englishReport": english_report,
            "arabicReport": arabic_report
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500