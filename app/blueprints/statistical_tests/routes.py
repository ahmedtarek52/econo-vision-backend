from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import numpy as np

# --- استيراد دوال الاختبارات الحسابية من ملف utils ---
from .utils import (
    run_adf_test,
    run_kpss_test,
    run_multicollinearity_test,
    run_optimal_lag_selection,
    run_johansen_cointegration_test,
    run_granger_causality_test,
    run_panel_unit_root_test,
    run_zivot_andrews_test, run_pesaran_cd_test # (جديد)
)

# --- استيراد ديكوريتور حماية المسارات ---
from app.decorators import login_required

# --- محاولة استيراد وظائف الذكاء الاصطناعي بشكل آمن ---
AI_CORE_AVAILABLE = False
try:
    # نحاول الاستيراد من المسار المفترض لـ AI Core
    from app.blueprints.ai_core.utils import (
        get_stationarity_recommendation,
        get_cointegration_recommendation
    )
    AI_CORE_AVAILABLE = True
except ImportError:
    print("Warning: AI Core utils not found within Statistical Tests routes. AI recommendations will be disabled.")
    # دوال وهمية لكي لا يتوقف الكود في حال عدم وجود الموديول
    def get_stationarity_recommendation(adf_result=None, kpss_result=None): return None
    def get_cointegration_recommendation(johansen_result=None): return None


# --- تعريف الـ Blueprint ---
pre_analysis_bp = Blueprint('pre_analysis_api', __name__)


# --- دالة معالجة الأخطاء الموحدة ---
def handle_error(e, default_message="An error occurred", status_code=500):
    """Logs the error and returns a clean JSON response."""
    error_message = str(e)
    try:
        current_app.logger.error(f"Pre-Analysis API Error: {error_message}", exc_info=True)
    except Exception:
        # في حال فشل الـ logger لأي سبب
        print(f"Pre-Analysis API Error (Logger Failed): {error_message}")
    
    user_message = default_message
    
    # إذا كان الخطأ ناتجاً عن مدخلات غير صالحة، نعيد رمز 400 ورسالة الخطأ الأصلية
    if isinstance(e, (ValueError, KeyError, TypeError, RuntimeError)):
        user_message = error_message
        status_code = 400
    
    return jsonify({"error": user_message}), status_code


# --- المسار الرئيسي لتشغيل الاختبارات ---
@pre_analysis_bp.route('/run-test', methods=['POST', 'OPTIONS'])
@login_required
def run_pre_estimation_test():
    """
    Endpoint لتشغيل الاختبارات التشخيصية القبلية.
    يستقبل JSON يحتوي على 'dataset' و 'testId' و 'params'.
    """
    
    # (1) الرد السريع على طلبات OPTIONS (Preflight check for CORS)
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    # متغيرات لتخزين النتائج لاستخدامها لاحقاً مع الذكاء الاصطناعي
    adf_results_for_ai = None
    kpss_results_for_ai = None

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Request payload is empty."}), 400
            
        dataset = payload.get('dataset')
        test_id = payload.get('testId')
        params = payload.get('params', {})

        if not dataset or not isinstance(dataset, list):
            return jsonify({"error": "'dataset' is missing or must be a list."}), 400
        
        if not test_id:
            return jsonify({"error": "'testId' is required."}), 400

        # تحويل البيانات إلى DataFrame
        df = pd.DataFrame(dataset)
        
        # (!!!) التعديل الأهم: تحويل آمن للأرقام لتجنب FutureWarning (!!!)
        for col in df.columns:
            # نحاول التحويل إلى أرقام، وإذا فشل نترك العمود كما هو (لأننا قد نحتاج الـ ID النصي)
            # لكن للاختبارات الحسابية، الدوال في utils ستقوم بعمل dropna() و select_dtypes()
           # تحويل البيانات إلى DataFrame
            df[col] = pd.to_numeric(df[col], errors='coerce') 
        # تحديد أعمدة الهوية والوقت لحمايتها من التحويل الرقمي القسري
        protected_cols = []
        if 'params' in payload:
            if 'panel_id_var' in payload['params']: protected_cols.append(payload['params']['panel_id_var'])
            if 'panel_time_var' in payload['params']: protected_cols.append(payload['params']['panel_time_var'])
            if 'variable' in payload['params']: 
                 # المتغير المستهدف يجب أن يكون رقمياً، لا نحميه
                 pass

        # تنظيف وتحويل البيانات
        for col in df.columns:
            if col in protected_cols:
                # نترك أعمدة الهوية والوقت كما هي (قد تكون نصوصاً أو تواريخ)
                continue
            
            # محاولة تحويل باقي الأعمدة إلى أرقام
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}

        # --- توجيه الطلب بناءً على testId ---

        # 1. Stationarity Tests (ADF)
        if test_id == 'adf':
            variable = params.get('variable')
            test_level = params.get('test_level', 'level')
            regression_type = params.get('regression_type', 'c')
            
            if not variable:
                raise ValueError("Missing 'variable' parameter for ADF test.")
            
            results = run_adf_test(df, variable, test_level, regression_type)
            adf_results_for_ai = results  # حفظ النتيجة للـ AI

        # 2. Stationarity Tests (KPSS)
        elif test_id == 'kpss':
            variable = params.get('variable')
            test_level = params.get('test_level', 'level')
            regression_type = params.get('regression_type', 'c')
            
            if not variable:
                raise ValueError("Missing 'variable' parameter for KPSS test.")
            
            results = run_kpss_test(df, variable, test_level, regression_type)
            kpss_results_for_ai = results  # حفظ النتيجة للـ AI

        # 3. Multicollinearity (VIF)
        elif test_id == 'vif':
            independent_vars = params.get('independent_vars')
            if not independent_vars:
                raise ValueError("Missing 'independent_vars' parameter for VIF test.")
            
            results = run_multicollinearity_test(df, independent_vars)

        # 4. Lag Order Selection
        elif test_id == 'lag_order':
            variables = params.get('variables')
            max_lags = int(params.get('max_lags', 4))
            
            if not variables:
                raise ValueError("Missing 'variables' parameter for Optimal Lag Selection.")
            
            results = run_optimal_lag_selection(df, variables=variables, maxlags=max_lags)

        # 5. Johansen Cointegration
        elif test_id == 'johansen':
            variables = params.get('variables')
            det_order = int(params.get('det_order', 0))
            k_ar_diff = int(params.get('k_ar_diff', 1))
            
            if not variables:
                raise ValueError("Missing 'variables' parameter for Johansen test.")
            
            results = run_johansen_cointegration_test(
                df, variables=variables, det_order=det_order, k_ar_diff=k_ar_diff
            )
            
            # إضافة توصية AI خاصة باختبار Johansen
            if AI_CORE_AVAILABLE:
                ai_rec = get_cointegration_recommendation(johansen_result=results)
                if ai_rec:
                    results['ai_recommendation'] = ai_rec

        # 6. Granger Causality
        elif test_id == 'granger':
            variables = params.get('variables')
            max_lag = params.get('max_lag')
            
            if not variables:
                raise ValueError("Missing 'variables' parameter for Granger test.")
            if max_lag is None:
                raise ValueError("Missing 'max_lag' parameter for Granger test.")
            
            results = run_granger_causality_test(df, variables=variables, max_lag=int(max_lag))
        
        # 7. Panel Unit Root Tests (New!)
        elif test_id == 'panel_stationarity':
            variable = params.get('variable')
            panel_id_var = params.get('panel_id_var')
            panel_time_var = params.get('panel_time_var')
            panel_test_type = params.get('panel_test_type', 'llc')
            
            if not variable or not panel_id_var or not panel_time_var:
                raise ValueError("Missing 'variable', 'panel_id_var', or 'panel_time_var' for Panel test.")
            
            results = run_panel_unit_root_test(df, variable, panel_id_var, panel_time_var, panel_test_type)

        # 8. Zivot-Andrews
        elif test_id == 'zivot_andrews':
            variable = params.get('variable')
            if not variable:
                raise ValueError("Missing 'variable' parameter for Zivot-Andrews test.")
            results = run_zivot_andrews_test(df, variable)

        # 9. Pesaran CD
        elif test_id == 'pesaran_cd':
            variable = params.get('variable')
            panel_id_var = params.get('panel_id_var')
            panel_time_var = params.get('panel_time_var')
            if not variable or not panel_id_var or not panel_time_var:
                raise ValueError("Missing 'variable', 'panel_id_var', or 'panel_time_var' for Pesaran CD test.")
            results = run_pesaran_cd_test(df, variable, panel_id_var, panel_time_var)

        else:
            return jsonify({"error": f"Unknown testId: '{test_id}'"}), 400
        

        # --- إضافة توصية الذكاء الاصطناعي للاستقرار (ADF/KPSS) ---
        # ملاحظة: التوصية تكون أدق إذا توفر الاختبارين معاً، لكننا هنا نقدم توصية بناءً على الاختبار الحالي
        if AI_CORE_AVAILABLE and (test_id == 'adf' or test_id == 'kpss'):
            ai_rec = get_stationarity_recommendation(
                adf_result=adf_results_for_ai,
                kpss_result=kpss_results_for_ai
            )
            # تأكد من أن التوصية ليست فارغة قبل إضافتها
            if ai_rec and "recommendation unavailable" not in ai_rec.lower():
                results['ai_recommendation'] = ai_rec

        # إرجاع النتائج بنجاح
        return jsonify(results), 200

    except Exception as e:
        # استخدام دالة معالجة الأخطاء الموحدة
        return handle_error(e, f"An error occurred while running test '{locals().get('test_id', 'unknown')}'")