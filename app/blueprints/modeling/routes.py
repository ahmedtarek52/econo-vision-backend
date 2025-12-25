import pandas as pd
import numpy as np
# (!!!) (إضافة g و traceback) (!!!)
from flask import Blueprint, request, jsonify, current_app, g
import statsmodels.api as sm 
import traceback # (لتحسين طباعة الأخطاء)

# (!!!) (حذف: قمنا بحذف استيرادات Redis) (!!!)
# from app.extensions import redis_client # <--- تم الحذف
# import pickle # <--- تم الحذف
# import json # <--- تم الحذف

from app.decorators import login_required

# --- Imports لدوال النماذج (من utils.py) ---
from .utils import (
    run_ols_model, run_ardl_model, run_var_model, run_vecm_model,
    run_arima_sarima, run_garch_model, run_panel_suite,
    run_logit_model, run_probit_model,
    run_ml_model,
    run_lasso_model,
    run_elasticnet_model,
    run_ridge_model,
    run_double_ml_model,
    run_single_equation_diagnostics,
    get_model_suggestion
)

from .utils_code_generator import generate_code_snippet

# --- Import لدوال الـ AI Core ---
try:
    from app.blueprints.ai_core.utils import (
        get_model_diagnostics_recommendations,
        get_panel_model_decision
    )
    AI_CORE_AVAILABLE = True
except ImportError:
    print("Warning: AI Core utils not found in modeling routes. AI recommendations disabled.")
    AI_CORE_AVAILABLE = False
    def get_model_diagnostics_recommendations(diagnostics_list=None): return ["AI recommendations unavailable."]
    def get_panel_model_decision(hausman_result=None, panel_lm_result=None): return "AI decision unavailable."
# ---

# --- Imports للاختبارات البعدية ---
# 
# 
from . import utils_post_tests as utils_post_tests
POST_TESTS_AVAILABLE = True
#try:
   # from . import utils_post_tests
  #  POST_TESTS_AVAILABLE = True
#except ImportError as e:
   # print(f"CRITICAL WARNING: utils_post_tests.py failed to import. Post-tests DISABLED. Error: {e}")
   # POST_TESTS_AVAILABLE = False
  # utils_post_tests = None 
# ---

# (!!!) (إعادة: الذاكرة المؤقتة القديمة) (!!!)
MODEL_CACHE = {
    "fitted_model": None,
    "model_id": None,
    "original_params": None,
    "dataframe": None
}
# ---

model_execution_bp = Blueprint('model_execution_api', __name__, url_prefix='/api/analysis/model')

# --- دالة Error Handling (لا تغيير) ---
def handle_error(e, default_message="An error occurred", status_code=500):
    error_message = str(e)
    try:
        current_app.logger.error(f"Model Execution API Error: {error_message}", exc_info=True)
    except Exception: 
        print(f"Model Execution API Error: {error_message}")
        traceback.print_exc()
    user_message = default_message
    if isinstance(e, (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError)):
        user_message = error_message
        status_code = 400
    return jsonify({"error": user_message}), status_code
# ---

# (!!!) (حذف: دالة get_redis_keys) (!!!)
# def get_redis_keys(uid): ... # <--- تم الحذف

@model_execution_bp.route('/run-model', methods=['POST'])
@login_required 
def run_selected_model():
    """Endpoint لتشغيل النموذج المختار وإضافة توصيات AI."""
    global MODEL_CACHE # (!!!) (إعادة: استخدام 'global') (!!!)

    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'modelId' not in payload or 'params' not in payload:
            return jsonify({"error": "Missing 'dataset', 'modelId', or 'params' in payload"}), 400

        dataset = payload['dataset']
        model_id = payload['modelId']
        params = payload['params']

        # (!!!) (حذف: uid و redis_keys) (!!!)

        if not isinstance(dataset, list) or not dataset:
            return jsonify({"error": "'dataset' must be a non-empty list."}), 400

        df = pd.DataFrame(dataset)
        results = {}
        ai_recommendations = []
        panel_decision = None
        
        # (!!!) (إعادة: تنظيف الـ Cache القديم) (!!!)
        MODEL_CACHE = {} 
        print(f"Running model: {model_id}")

        # --- (تشغيل النموذج المحدد - الكود كما هو تماماً) ---
        if model_id == 'ols':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for OLS.")
            results = run_ols_model(df, dep_var, indep_vars) 
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))
        
        elif model_id == 'lasso':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Lasso.")
            results = run_lasso_model(df, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))
        
        elif model_id == 'elastic_net':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for ElasticNet.")
            results = run_elasticnet_model(df, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        elif model_id == 'ridge':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Ridge.")
            results = run_ridge_model(df, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        elif model_id == 'double_ml':
            dep_var = params.get('dependent_var')
            treatment_var = params.get('treatment_var')
            control_vars = params.get('control_vars')
            ml_method = params.get('ml_method', 'lasso')
            if not dep_var or not treatment_var or not control_vars:
                raise ValueError("Double ML requires a Dependent (Y), a Treatment (d), and Control (X) variables.")
            results = run_double_ml_model(df, dep_var, treatment_var, control_vars, ml_method)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        elif model_id == 'ardl':
            endog = params.get('endog_var') or params.get('dependent_var')
            exog = params.get('exog_vars') or params.get('independent_vars')
            lags = int(params.get('lags', 1))
            
            raw_exog = params.get('exog_lags')
            exog_lags = int(raw_exog) if (raw_exog is not None and raw_exog != "") else 0
            
            trend = params.get('trend', 'c')
            
            # (!!!) استقبال طريقة الاختيار (!!!)
            selection_method = params.get('selection_method', 'fixed')

            if not endog: raise ValueError("Missing dependent variable for ARDL.")

            results = run_ardl_model(df, endog, exog, lags=lags, exog_lags=exog_lags, trend=trend, selection_method=selection_method)
            
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))
        elif model_id == 'var':
            variables = params.get('variables'); maxlags = params.get('maxlags', 4)
            if not variables: raise ValueError("Missing 'variables' for VAR.")
            results = run_var_model(df, variables, maxlags=maxlags)

        elif model_id == 'vecm':
            variables = params.get('variables'); lags = params.get('lags', 2); coint_rank = params.get('coint_rank', 1)
            if not variables: raise ValueError("Missing 'variables' for VECM.")
            results = run_vecm_model(df, variables, lags=lags, coint_rank=coint_rank)

        elif model_id == 'arima':
            endog = params.get('endog_var'); exog = params.get('exog_vars'); order = params.get('order'); seasonal_order = params.get('seasonal_order')
            if not endog or not order: raise ValueError("Missing endog_var or order for ARIMA.")
            seasonal_order_tuple = tuple(seasonal_order) if seasonal_order and len(seasonal_order) == 4 else (0,0,0,0)
            results = run_arima_sarima(df, endog, exog_vars=exog, order=tuple(order), seasonal_order=seasonal_order_tuple)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        elif model_id == 'garch':
            endog = params.get('endog_var'); p = params.get('p', 1); q = params.get('q', 1)
            if not endog: raise ValueError("Missing endog_var for GARCH.")
            results = run_garch_model(df, endog, p=p, q=q)

        elif model_id == 'panel':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars'); panel_id = params.get('panel_id_var'); panel_time = params.get('panel_time_var')
            if not all([dep_var, indep_vars, panel_id, panel_time]): raise ValueError("Missing required parameters for Panel model.")
            results = run_panel_suite(df, dep_var, indep_vars, panel_id, panel_time)
            if AI_CORE_AVAILABLE:
                hausman_res = next((item for item in results.get('diagnostics', []) if 'hausman' in item.get('name', '').lower()), None)
                lm_res = None 
                panel_decision = get_panel_model_decision(hausman_result=hausman_res, panel_lm_result=lm_res)

        elif model_id == 'logit':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Logit.")
            results = run_logit_model(df, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        elif model_id == 'probit':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Probit.")
            results = run_probit_model(df, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))
        
        elif model_id == 'random_forest' or model_id == 'xgboost':
            dep_var = params.get('dependent_var')
            indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError(f"Missing dependent/independent vars for {model_id}.")
            results = run_ml_model(df, model_id, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        else:
            return jsonify({"error": f"Unknown modelId: '{model_id}'"}), 400
         
        # --- (!!!) (إعادة: التخزين في MODEL_CACHE) (!!!) ---
        if results.get("fitted_model_object"):
            MODEL_CACHE["fitted_model"] = results.get("fitted_model_object")
            MODEL_CACHE["model_id"] = model_id
            MODEL_CACHE["original_params"] = params
            MODEL_CACHE["dataframe"] = df 
            print(f"Model {model_id} successfully fitted and cached (in-memory).")
        else:
            print(f"Warning: Model {model_id} ran, but 'fitted_model_object' was not returned for caching.")
        # --- (!!!) (نهاية التعديل) (!!!) ---

        # --- (إضافة توصيات الـ AI للـ Response - لا تغيير) ---
        final_recommendations = []
        if ai_recommendations and isinstance(ai_recommendations, list):
            final_recommendations.extend(ai_recommendations)
        if panel_decision:
            final_recommendations.append(f"Panel Decision Aid: {panel_decision}")
        if final_recommendations:
            results['ai_recommendations_list'] = final_recommendations
        # ---

        response_data = {
            "summary_html": results.get("summary_html"),
            "comparison_html": results.get("comparison_html"),
            "diagnostics": results.get("diagnostics", []),
            "metrics": results.get("metrics", {}), 
            "ai_recommendations_list": results.get("ai_recommendations_list", [])
        }
        if not response_data["summary_html"] and not response_data["comparison_html"]:
            response_data["summary_html"] = "<p>No standard summary table generated or an error occurred during summary generation.</p>"

        return jsonify(response_data), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err) # Returns 400
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running model '{payload.get('modelId', 'N/A')}'", 500)


# (!!!) (إعادة: suggest-model) (!!!)
@model_execution_bp.route('/suggest-model', methods=['POST'])
@login_required 
def suggest_best_model():
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'y_vars' not in payload:
            return jsonify({"error": "Missing 'dataset' or 'y_vars' in payload"}), 400
        dataset = payload['dataset']
        y_vars = payload['y_vars']
        x_vars = payload.get('x_vars', [])
        is_panel = payload.get('is_panel', False)
        panel_id = payload.get('panel_id_var')
        panel_time = payload.get('panel_time_var')
        if not isinstance(dataset, list) or not dataset:
            return jsonify({"error": "'dataset' must be a non-empty list."}), 400
        if not isinstance(y_vars, list) or not y_vars:
            return jsonify({"error": "'y_vars' must be a non-empty list."}), 400
        df = pd.DataFrame(dataset)
        suggestion = get_model_suggestion(
            df=df,
            y_vars=y_vars,
            x_vars=x_vars,
            is_panel=is_panel,
            panel_id=panel_id,
            panel_time=panel_time
        )
        return jsonify(suggestion), 200
    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, "An internal server error occurred while generating suggestion.", 500)
# --- (نهاية الإضافة) ---


# (!!!) (إعادة: جلب البيانات من MODEL_CACHE) (!!!)
@model_execution_bp.route('/run-post-test', methods=['POST'])
@login_required 
def run_post_estimation_test():
    """Endpoint لتشغيل اختبار بعدي محدد (يستخدم الـ Cache الآن)."""
    global MODEL_CACHE 
    
    if not POST_TESTS_AVAILABLE or utils_post_tests is None:
        return jsonify({"error": "Post-estimation testing module is not available."}), 501

    try:
        payload = request.get_json()
        required_keys = ['originalModelId', 'testId', 'originalParams']
        if not payload or not all(key in payload for key in required_keys):
            return jsonify({"error": "Missing required keys ('originalModelId', 'testId', 'originalParams')."}), 400

        original_model_id = payload['originalModelId']
        test_id = payload['testId']
        params = payload.get('params', {})
        original_params_from_frontend = payload['originalParams']

        # 1. جلب البيانات من الـ Cache القديم
        cached_model = MODEL_CACHE.get("fitted_model")
        cached_model_id = MODEL_CACHE.get("model_id")

        if cached_model is None or cached_model_id != original_model_id:
            print(f"Cache miss. Requested: {original_model_id}, Cached: {cached_model_id}")
            return handle_error(ValueError("Model cache is empty or mismatched. Please re-run the main model first before running post-tests."), 400)
        
        print(f"Cache hit. Running post-test '{test_id}' on cached {cached_model_id} model.")

        # 2. تشغيل الاختبار (كما كان)
        post_test_func_name = f"run_{test_id}"
        post_test_func = getattr(utils_post_tests, post_test_func_name, None)

        if post_test_func and callable(post_test_func):
            results = post_test_func(
                model_id=cached_model_id, 
                results=cached_model, 
                test_params=params,
                original_df=MODEL_CACHE.get("dataframe"),
                original_params=MODEL_CACHE.get("original_params") 
            )
        else:
            raise NotImplementedError(f"Test '{test_id}' function '{post_test_func_name}' not found in utils_post_tests.py.")
            #raise NotImplementedError(f"Unknown or not implemented post-estimation testId: '{test_id}'.")

        if AI_CORE_AVAILABLE and isinstance(results, dict) and results.get('interpretation'):
            if test_id != 'forecast' and test_id != 'irf' and test_id != 'fevd':
                recs = get_model_diagnostics_recommendations([results])
                if recs and isinstance(recs, list) and len(recs) > 0:
                    if isinstance(recs[0], dict) and "unavailable" not in recs[0].get("text", "") and "model diagnostics" not in recs[0].get("text", "").lower():
                        results['ai_recommendation'] = recs[0].get("text")
                    elif isinstance(recs[0], str):
                        print(f"AI returned a string, not an object: {recs[0]}")
        
        return jsonify(results), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running post-test '{payload.get('testId', 'N/A')}'", 500)


# (!!!) (إعادة: جلب البيانات من MODEL_CACHE) (!!!)
@model_execution_bp.route('/run-quick-fix', methods=['POST'])
@login_required 
def run_quick_fix():
    """
    Runs a 'quick fix' on the cached model (e.g., re-fits OLS/ARDL with robust SE).
    """
    global MODEL_CACHE
    try:
        payload = request.get_json()
        fix_code = payload.get('fixCode')
        
        # 1. جلب البيانات من الـ Cache القديم
        cached_model_id = MODEL_CACHE.get("model_id")
        original_params = MODEL_CACHE.get("original_params")
        original_df = MODEL_CACHE.get("dataframe")

        if not all([cached_model_id, original_params, original_df is not None]):
            raise ValueError("Cache is invalid. Cannot apply fix. Please re-run the main model.")

        print(f"Applying Quick Fix: {fix_code} to model {cached_model_id}")

        ai_recommendations = []
        new_params = original_params.copy()
        results_for_frontend = {}

        # 2. تشغيل الإصلاح (كما كان)
        if cached_model_id == 'ols':
            dep_var = original_params.get('dependent_var')
            indep_vars = original_params.get('independent_vars')
            cov_type = 'nonrobust'; cov_kwds = None
            if fix_code == 'USE_HAC':
                cov_type = 'HAC'; cov_kwds = {'maxlags': 5}; new_params['cov_type'] = 'HAC'
            elif fix_code == 'USE_ROBUST_HC3':
                cov_type = 'HC3'; new_params['cov_type'] = 'HC3'
            else:
                raise ValueError(f"Unknown fix_code: {fix_code}")
            
            results_fixed = run_ols_model(original_df, dep_var, indep_vars, cov_type=cov_type, cov_kwds=cov_kwds)
            
        elif cached_model_id == 'ardl':
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)
            cov_type = 'nonrobust'
            if fix_code == 'USE_HAC':
                cov_type = 'HAC'; new_params['cov_type'] = 'HAC'
            elif fix_code == 'USE_ROBUST_HC3':
                cov_type = 'HC3'; new_params['cov_type'] = 'HC3'
            else:
                raise ValueError(f"Unknown fix_code: {fix_code}")
                
            results_fixed = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type=cov_type)
        
        else:
            raise ValueError(f"Quick Fix is not implemented for model type: {cached_model_id}")
        
        if AI_CORE_AVAILABLE:
            ai_recommendations = get_model_diagnostics_recommendations(results_fixed.get("diagnostics", []))
        ai_recommendations.append(
            {"text": f"**Quick Fix Applied:** Model re-estimated using **{cov_type}** robust standard errors."}
        )

        # 3. (!!!) (إعادة: تخزين النموذج المُعدل في Cache) (!!!)
        MODEL_CACHE["fitted_model"] = results_fixed.get("fitted_model_object")
        MODEL_CACHE["original_params"] = new_params
        print(f"Quick Fix {fix_code} applied and model re-cached (in-memory).")

        # 4. إرسال الرد
        results_for_frontend = {
            "summary_html": results_fixed.get("summary_html"),
            "diagnostics": results_fixed.get("diagnostics", []),
            "metrics": results_fixed.get("metrics", {}),
            "ai_recommendations_list": ai_recommendations,
            "newParams": new_params
        }
        return jsonify(results_for_frontend), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running Quick Fix '{payload.get('fixCode', 'N/A')}'", 500)


# (!!!) (إعادة: جلب البيانات من MODEL_CACHE) (!!!)
@model_execution_bp.route('/run-robustness-comparison', methods=['POST'])
@login_required 
def run_robustness_comparison():
    global MODEL_CACHE
    try:
        # 1. جلب البيانات من الـ Cache القديم
        original_model_id = MODEL_CACHE.get("model_id")
        original_params = MODEL_CACHE.get("original_params")
        original_df = MODEL_CACHE.get("dataframe")

        if not all([original_model_id, original_params, original_df is not None]):
            raise ValueError("Model cache is invalid. Cannot run comparison. Please re-run the main model.")

        print(f"Running Robustness Comparison for model {original_model_id}")
        comparison_results = []
        
        # 2. تشغيل النماذج الثلاثة (كما كان)
        if original_model_id == 'ols':
            dep_var = original_params.get('dependent_var')
            indep_vars = original_params.get('independent_vars')
            
            results_orig = run_ols_model(original_df, dep_var, indep_vars, cov_type='nonrobust')
            comparison_results.append({ "modelName": f"OLS (Non-Robust)", "metrics": results_orig.get("metrics", {}), "diagnostics": results_orig.get("diagnostics", []) })
            
            results_hac = run_ols_model(original_df, dep_var, indep_vars, cov_type='HAC', cov_kwds={'maxlags': 5})
            comparison_results.append({ "modelName": f"OLS (HAC Robust)", "metrics": results_hac.get("metrics", {}), "diagnostics": results_hac.get("diagnostics", []) })

            results_hc3 = run_ols_model(original_df, dep_var, indep_vars, cov_type='HC3')
            comparison_results.append({ "modelName": f"OLS (HC3 Robust)", "metrics": results_hc3.get("metrics", {}), "diagnostics": results_hc3.get("diagnostics", []) })

        elif original_model_id == 'ardl':
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)

            results_orig = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='nonrobust')
            comparison_results.append({ "modelName": f"ARDL (Non-Robust)", "metrics": results_orig.get("metrics", {}), "diagnostics": results_orig.get("diagnostics", []) })
            
            results_hac = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='HAC')
            comparison_results.append({ "modelName": f"ARDL (HAC Robust)", "metrics": results_hac.get("metrics", {}), "diagnostics": results_hac.get("diagnostics", []) })

            results_hc3 = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='HC3')
            comparison_results.append({ "modelName": f"ARDL (HC3 Robust)", "metrics": results_hc3.get("metrics", {}), "diagnostics": results_hc3.get("diagnostics", []) })
        
        else:
            raise ValueError(f"Robustness comparison not implemented for {original_model_id}")

        return jsonify(comparison_results), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running robustness comparison", 500)


# (!!!) (إعادة: جلب البيانات من MODEL_CACHE) (!!!)
@model_execution_bp.route('/get-code', methods=['POST'])
@login_required 
def get_model_code_snippet():
    global MODEL_CACHE
    try:
        payload = request.get_json()
        transformation_history = payload.get('transformationHistory', []) 

        # 1. جلب البيانات من الـ Cache القديم
        model_id = MODEL_CACHE.get("model_id")
        params = MODEL_CACHE.get("original_params")
        df = MODEL_CACHE.get("dataframe")

        if not all([model_id, params, df is not None]):
             raise ValueError("Model cache is invalid. Cannot generate code. Please re-run the main model.")

        # 2. إنشاء الكود (كما كان)
        code_snippet = generate_code_snippet(model_id, params, df, transformation_history)

        return jsonify({"code_snippet": code_snippet}), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while generating code", 500)