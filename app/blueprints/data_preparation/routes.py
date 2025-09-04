# app/blueprints/data_preparation/routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from .utils import (
    remove_missing_values, 
    impute_with_average, # <-- IMPORT THE NEW FUNCTION
    handle_outliers, 
    remove_duplicates, 
    normalize_data
)

data_preparation_bp = Blueprint('data_preparation_bp', __name__)

@data_preparation_bp.route('/clean', methods=['POST'])
def clean_data():
    """
    A single endpoint to apply a specified cleaning operation.
    """
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        operation = payload.get('operation')
        params = payload.get('params', {})

        if not dataset or not operation:
            return jsonify({"error": "Missing 'dataset' or 'operation' in request"}), 400

        df = pd.DataFrame(dataset)

        # Apply the requested cleaning function
        if operation == 'remove-missing':
            cleaned_df = remove_missing_values(df)
        elif operation == 'impute-missing':
            # USE THE NEW ALGORITHM
            cleaned_df = impute_with_average(df)
        elif operation == 'handle-outliers':
            cleaned_df = handle_outliers(df, column=params.get('column'), method=params.get('method', 'iqr'))
        elif operation == 'remove-duplicates':
            cleaned_df = remove_duplicates(df)
        elif operation == 'normalize-data':
            cleaned_df = normalize_data(df, column=params.get('column'), method=params.get('method', 'standardize'))
        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400
        
        response_data = cleaned_df.where(pd.notnull(cleaned_df), None).to_dict(orient='records')

        return jsonify({
            "message": f"Operation '{operation}' applied successfully.",
            "cleanedDataset": response_data
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
