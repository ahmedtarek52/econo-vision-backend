# app/blueprints/statistical_tests/routes.py

from flask import Blueprint, request, jsonify
import pandas as pd

# استيراد جميع دوال الاختبارات الجديدة من ملف utils.py
from .utils import (
    run_stationarity_tests,
    run_autocorrelation_analysis,
    run_multicollinearity_test,
    run_optimal_lag_selection,
    run_johansen_cointegration_test
)

# إنشاء Blueprint لتنظيم المسارات
statistical_tests_bp = Blueprint('statistical_tests', __name__, url_prefix='/api/tests')

@statistical_tests_bp.route('/run-test', methods=['POST'])
def run_single_test():
    """
    Endpoint مركزي لتشغيل الاختبارات القبلية بناءً على testId.
    """
    try:
        data = request.get_json()
        
        # التأكد من وصول البيانات المطلوبة
        if 'dataset' not in data or 'testId' not in data:
            return jsonify({"error": "Missing 'dataset' or 'testId' in request"}), 400

        df = pd.DataFrame(data['dataset'])
        test_id = data['testId']
        
        # استلام المتغيرات المستقلة (مهم لاختبار VIF)
        independent_vars = data.get('independent_vars', [])
        
        results = {}

        # --- توجيه الطلب إلى الدالة المناسبة بناءً على testId ---

        if test_id == 'stationarity':
            results = run_stationarity_tests(df)
        
        elif test_id == 'autocorrelation':
            results = run_autocorrelation_analysis(df)
            
        elif test_id == 'vif':
            results = run_multicollinearity_test(df, independent_vars)

        elif test_id == 'lag_order':
            html_table = run_optimal_lag_selection(df)
            # نغلف جدول الـ HTML في JSON لإرساله
            results = {"html_table": html_table}

        elif test_id == 'johansen':
            results = run_johansen_cointegration_test(df)
            
        else:
            return jsonify({"error": f"Unknown testId: '{test_id}'"}), 400

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500
