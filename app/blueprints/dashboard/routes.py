# app/blueprints/dashboard/routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from .utils import calculate_summary_statistics

dashboard_bp = Blueprint('dashboard_bp', __name__)

@dashboard_bp.route('/summary', methods=['POST'])
def get_summary():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')

        if not dataset:
            return jsonify({"error": "Missing 'dataset' in request"}), 400

        df = pd.DataFrame(dataset)
        
        # Generate summary stats for all numeric columns
        summary = calculate_summary_statistics(df)

        return jsonify({
            "message": "Summary statistics generated successfully.",
            "summary": summary
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500