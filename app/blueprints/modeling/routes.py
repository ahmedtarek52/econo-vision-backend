# app/blueprints/modeling/routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from .utils import run_ols_model, run_var_model, run_arima_model

modeling_bp = Blueprint('modeling_bp', __name__)

@modeling_bp.route('/run-model', methods=['POST'])
def run_model():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        model_id = payload.get('modelId')
        endog_vars = payload.get('endogenous')
        exog_vars = payload.get('exogenous')

        if not all([dataset, model_id, endog_vars]):
            return jsonify({"error": "Missing dataset, modelId, or endogenous variables"}), 400

        df = pd.DataFrame(dataset)
        df.dropna(inplace=True) # Models require complete data

        if model_id == 'ols':
            if len(endog_vars) != 1:
                return jsonify({"error": "OLS requires exactly one endogenous variable."}), 400
            results = run_ols_model(df, endog_vars[0], exog_vars)
        
        elif model_id == 'var':
            if len(endog_vars) < 2:
                return jsonify({"error": "VAR requires at least two endogenous variables."}), 400
            results = run_var_model(df, endog_vars)
            
        elif model_id == 'arima':
            if len(endog_vars) != 1:
                return jsonify({"error": "ARIMA requires exactly one endogenous variable."}), 400
            results = run_arima_model(df, endog_vars[0])
            
        else:
            return jsonify({"error": f"Unknown modelId: {model_id}"}), 400

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during model fitting: {str(e)}"}), 500