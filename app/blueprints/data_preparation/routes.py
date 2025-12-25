import pandas as pd
import numpy as np

from flask import Blueprint, request, jsonify, current_app
from app.decorators import login_required

# --- تحديث: استخدام الاستيراد النسبي (Relative Imports) ---
from .utils import (
    remove_missing_values,
    impute_with_average,
    handle_outliers,
    remove_duplicates,
    normalize_data,
    apply_transformations, 
    delete_columns,       
    create_dummy_variables 
)

from .scanner import scan_dataset_health, apply_cleaning_fixes

data_preparation_bp = Blueprint('data_prep_api', __name__)

# --- Helper Function for Error Handling ---
def handle_error(e, default_message="An error occurred", status_code=500):
    """Logs the error and returns a JSON response."""
    error_message = str(e)
    try:
        current_app.logger.error(f"Data Prep API Error: {error_message}", exc_info=True)
    except:
        print(f"Data Prep API Error: {error_message}")
        import traceback
        traceback.print_exc()

    user_message = default_message
    if isinstance(e, (ValueError, KeyError, TypeError)):
        user_message = error_message
        status_code = 400

    return jsonify({"error": user_message}), status_code

# --- دالة مساعدة لتطهير البيانات قبل الإرسال (Sanitize for JSON) ---
def sanitize_for_json(df):
    """
    تجهيز الداتا فريم للإرسال كـ JSON:
    1. تحويل التواريخ لنصوص (لتجنب خطأ NaT).
    2. استبدال NaN و Infinity بـ None (لتجنب خطأ JSON NaN).
    """
    df_clean = df.copy()
    
    # 1. معالجة التواريخ (Dates to Strings)
    datetime_cols = df_clean.select_dtypes(include=['datetime64[ns]', 'datetime']).columns
    for col in datetime_cols:
        # تحويل التواريخ لنص ISO، واستبدال NaT بـ None
        df_clean[col] = df_clean[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

    # 2. معالجة الأرقام (NaN/Inf to None)
    # استبدال القيم غير الصالحة JSON بـ None
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    df_clean = df_clean.replace({np.nan: None}) # استبدال صريح لـ NaN
    
    # تأكيد أخير: where للقيم الفارغة
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    
    return df_clean.to_dict(orient='records')

# ==============================================================================
# 1. المسارات الذكية (Smart Scanning & Fixing)
# ==============================================================================

@data_preparation_bp.route('/scan-health', methods=['POST'])
@login_required
def scan_data_health():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        manual_date_col = payload.get('manualDateColumn')
        
        if not dataset: return jsonify({"error": "No dataset provided"}), 400
            
        df = pd.DataFrame(dataset)
        
        health_report = scan_dataset_health(df, manual_date_col=manual_date_col)
        
        return jsonify({
            "status": "success",
            "report": health_report,
            "total_columns": len(df.columns),
            "total_rows": len(df)
        }), 200
    except Exception as e:
        return handle_error(e, "Error scanning data health")

@data_preparation_bp.route('/apply-fixes', methods=['POST'])
@login_required
def apply_auto_fixes():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        cleaning_plan = payload.get('cleaningPlan') 
        
        if not dataset or not cleaning_plan: 
            return jsonify({"error": "Missing data or cleaning plan"}), 400
            
        df = pd.DataFrame(dataset)
        
        # تطبيق الإصلاحات
        df_cleaned = apply_cleaning_fixes(df, cleaning_plan)
        
        # تطهير البيانات للإرسال (الحل الجذري لمشاكل JSON)
        cleaned_data = sanitize_for_json(df_cleaned)
        
        return jsonify({
            "status": "success",
            "cleaned_dataset": cleaned_data,
            "message": "Smart fixes applied successfully."
        }), 200
    except Exception as e:
        return handle_error(e, "Error applying smart fixes")


# ==============================================================================
# 2. المسارات اليدوية (Manual Operations)
# ==============================================================================

@data_preparation_bp.route('/clean', methods=['POST'])
@login_required
def clean_data():
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'operation' not in payload:
            return jsonify({"error": "Missing 'dataset' or 'operation'"}), 400

        dataset = payload['dataset']
        operation = payload['operation']
        params = payload.get('params', {})

        if not dataset:
            return jsonify({"cleanedDataset": [], "columns": [], "message": "Dataset empty."}), 200

        df = pd.DataFrame(dataset)
        cleaned_df = df.copy()

        if operation == 'remove-missing':
            cleaned_df = remove_missing_values(cleaned_df)
        elif operation == 'impute-missing':
            cleaned_df = impute_with_average(cleaned_df)
        elif operation == 'remove-duplicates':
            cleaned_df = remove_duplicates(cleaned_df)
        elif operation == 'handle-outliers' or operation == 'normalize-data':
            columns_to_process = params.get('columns')
            if not columns_to_process or not isinstance(columns_to_process, list):
                raise ValueError(f"Operation '{operation}' requires 'columns' list.")
            func = handle_outliers if operation == 'handle-outliers' else normalize_data
            for col in columns_to_process:
                if col in cleaned_df.columns:
                    cleaned_df = func(cleaned_df, column=col)
        else:
            return jsonify({"error": f"Unknown operation: {operation}"}), 400

        # تطهير البيانات للإرسال
        response_data = sanitize_for_json(cleaned_df)

        return jsonify({
            "message": f"Operation '{operation}' applied.",
            "cleanedDataset": response_data,
            "columns": cleaned_df.columns.tolist()
        }), 200

    except Exception as e:
        return handle_error(e, f"Error during '{payload.get('operation')}'")


@data_preparation_bp.route('/transform', methods=['POST'])
@login_required
def handle_transform_data():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        transformations = payload.get('transformations')

        if not dataset: return jsonify({"transformedDataset": [], "columns": []}), 200
        if not transformations: return jsonify({"error": "No transformations provided"}), 400

        df = pd.DataFrame(dataset)
        df_transformed, new_cols = apply_transformations(df, transformations)

        # تطهير البيانات للإرسال
        response_data = sanitize_for_json(df_transformed)

        return jsonify({
            "message": "Transformations applied.",
            "transformedDataset": response_data,
            "columns": df_transformed.columns.tolist(),
            "newColumns": new_cols
        }), 200
    except Exception as e:
        return handle_error(e, "Error applying transformations")


@data_preparation_bp.route('/delete-columns', methods=['POST'])
@login_required
def handle_delete_columns():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        cols_to_delete = payload.get('columns_to_delete')

        if not dataset: return jsonify({"error": "Empty dataset"}), 400
        if not cols_to_delete: return jsonify({"error": "No columns selected"}), 400

        df = pd.DataFrame(dataset)
        df_deleted, deleted_list = delete_columns(df, cols_to_delete)

        # تطهير البيانات للإرسال
        response_data = sanitize_for_json(df_deleted)

        return jsonify({
            "message": f"Deleted {len(deleted_list)} columns.",
            "dataset_after_delete": response_data,
            "new_columns": df_deleted.columns.tolist()
        }), 200
    except Exception as e:
        return handle_error(e, "Error deleting columns")


@data_preparation_bp.route('/create-dummies', methods=['POST'])
@login_required
def handle_create_dummies():
    try:
        payload = request.get_json()
        dataset = payload.get('dataset')
        cols_to_dummify = payload.get('columns_to_dummify')

        if not dataset: return jsonify({"error": "Empty dataset"}), 400
        if not cols_to_dummify: return jsonify({"error": "No columns selected"}), 400

        df = pd.DataFrame(dataset)
        df_dummy, new_cols = create_dummy_variables(df, cols_to_dummify, drop_first=True)

        # تطهير البيانات للإرسال
        response_data = sanitize_for_json(df_dummy)

        return jsonify({
            "message": f"Created {len(new_cols)} dummy variables.",
            "dataset_after_dummies": response_data,
            "new_columns": df_dummy.columns.tolist()
        }), 200
    except Exception as e:
        return handle_error(e, "Error creating dummy variables")