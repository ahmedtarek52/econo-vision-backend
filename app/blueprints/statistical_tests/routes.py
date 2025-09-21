# app/blueprints/statistical_tests/routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from .utils import run_adf_test, run_granger_causality_test

statistical_tests_bp = Blueprint('statistical_tests_bp', __name__)

@statistical_tests_bp.route('/run-test', methods=['POST'])
def run_test():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        test_id = payload.get('testId')
        params = payload.get('params', {})

        if not all([dataset, test_id]):
            return jsonify({"error": "Missing dataset or testId"}), 400

        df = pd.DataFrame(dataset)
        
        # Drop rows with NaN values for accurate statistical testing
        df.dropna(inplace=True)

        if test_id == 'adf':
            variable = params.get('variable')
            if not variable:
                return jsonify({"error": "ADF test requires a 'variable' parameter"}), 400
            results = run_adf_test(df, variable)
        
        elif test_id == 'granger':
            max_lags = params.get('max_lags', 2)
            results = run_granger_causality_test(df, max_lags=max_lags)
            
        else:
            return jsonify({"error": f"Unknown testId: {test_id}"}), 400

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500