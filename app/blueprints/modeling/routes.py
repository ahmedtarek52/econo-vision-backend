import pandas as pd
import numpy as np
from flask import Blueprint, request, jsonify, current_app, g
import statsmodels.api as sm 
import traceback 

from app.decorators import login_required
from .utils_code_generator import generate_code_snippet

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
    get_model_suggestion ,
    run_dynamic_panel_gmm  # 👈 ضِف هذه هنا بنظافة
)

# =========================================================================
# 🏛️ محرك الـ Stargazer المدمج محلياً لضمان عدم حدوث أي خطأ في المسارات 🏛️
# =========================================================================
def format_to_stargazer_html(model_results_dict, dep_var_name="Dependent Variable"):
    """
    Formats single or multiple econometric model results into a publication-ready
    Stargazer-style HTML table. Supports both statsmodels and linearmodels.
    """
    try:
        all_indep_vars = []
        for name, res in model_results_dict.items():
            if res is not None:
                vars_list = res.params.index.tolist()
                for v in vars_list:
                    if v not in all_indep_vars and v != 'const':
                        all_indep_vars.append(v)
                        
        if any('const' in res.params.index for res in model_results_dict.values() if res is not None):
            all_indep_vars.append('const')

        html = '<table class="stargazer-academic-table">'
        html += f'<caption>Model Estimation Matrix (Dependent Variable: {dep_var_name})</caption>'
        
        html += '<thead>'
        html += '<tr class="top-border"><th>Regressor</th>'
        for model_name in model_results_dict.keys():
            html += f'<th>{model_name}</th>'
        html += '</tr>'
        
        html += '<tr class="bottom-border"><td></td>'
        for res in model_results_dict.values():
            estimator_name = res.__class__.__name__ if res is not None else "N/A"
            if "Pooled" in estimator_name: estimator_name = "Pooled OLS"
            elif "Panel" in estimator_name: estimator_name = "Fixed Effects"
            elif "Random" in estimator_name: estimator_name = "Random Effects"
            html += f'<td><span class="estimator-sub font-mono text-xs text-gray-500">({estimator_name})</span></td>'
        html += '</tr></thead><tbody>'

        for var in all_indep_vars:
            display_var = "Constant" if var == 'const' else var
            html += f'<tr><td class="var-name font-semibold text-left">{display_var}</td>'
            for res in model_results_dict.values():
                if res is not None and var in res.params.index:
                    coef = res.params[var]
                    pval = res.pvalues[var]
                    
                    stars = ""
                    if pval <= 0.01: stars = "***"
                    elif pval <= 0.05: stars = "**"
                    elif pval <= 0.1: stars = "*"
                    
                    html += f'<td>{coef:.4f}{stars}</td>'
                else:
                    html += '<td></td>'
            html += '</tr>'

            html += '<tr class="t-stat-row"><td></td>'
            for res in model_results_dict.values():
                tstats = getattr(res, 'tvalues', getattr(res, 'tstats', None)) if res is not None else None
                if res is not None and tstats is not None and var in tstats.index:
                    t_stat = tstats[var]
                    html += f'<td><span class="text-gray-500">({t_stat:.4f})</span></td>'
                else:
                    html += '<td></td>'
            html += '</tr>'

        html += '<tr class="top-border"><td class="font-medium text-left">Observations</td>'
        for res in model_results_dict.values():
            nobs = getattr(res, 'nobs', 'N/A')
            html += f'<td>{nobs}</td>'
        html += '</tr>'

        html += '<tr><td class="font-medium text-left">R-squared</td>'
        for res in model_results_dict.values():
            rsq = getattr(res, 'rsquared', None)
            rsq_str = f"{rsq:.4f}" if rsq is not None else "N/A"
            html += f'<td>{rsq_str}</td>'
        html += '</tr>'
        
        html += '<tr class="bottom-border"><td class="font-medium text-left">F-Statistic / Prob</td>'
        for res in model_results_dict.values():
            f_stat = getattr(res, 'f_statistic', None)
            if f_stat is not None and hasattr(f_stat, 'stat'):
                html += f'<td>{f_stat.stat:.2f} <span class="text-xs">({f_stat.pvalue:.4f})</span></td>'
            elif hasattr(res, 'fvalue'):
                html += f'<td>{res.fvalue:.2f} <span class="text-xs">({res.f_pvalue:.4f})</span></td>'
            else:
                html += '<td>N/A</td>'
        html += '</tr>'

        html += '</tbody></table>'
        html += '<div class="stargazer-legend text-xs mt-2 text-gray-500 italic text-left">Note: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1. T-statistics reported in parentheses.</div>'
        
        return html
    except Exception as e:
        return f"<p class='text-red-500'>Failed to format Stargazer Table: {str(e)}</p>"
# =========================================================================

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

# --- Imports للاختبارات البعدية ---
from . import utils_post_tests as utils_post_tests
POST_TESTS_AVAILABLE = True

def extract_coefficients_from_model(model_id, fitted_model, params):
    coefs_list = []
    if fitted_model is None:
        return coefs_list
        
    try:
        # 1. Double ML
        if model_id == 'double_ml':
            if isinstance(fitted_model, dict):
                treatment_var = fitted_model.get('treatment_var', params.get('treatment_var', 'D'))
                final_ols = fitted_model.get('final_ols')
                if final_ols and hasattr(final_ols, 'params'):
                    if len(final_ols.params) > 1:
                        coefs_list.append({"name": str(treatment_var), "value": float(final_ols.params[1])})
                    if len(final_ols.params) > 0:
                        coefs_list.append({"name": "const", "value": float(final_ols.params[0])})
                        
        # 2. Statsmodels / Linearmodels
        elif hasattr(fitted_model, 'params'):
            if hasattr(fitted_model.params, 'items'):
                for name, val in fitted_model.params.items():
                    coefs_list.append({"name": str(name), "value": float(val)})
            else:
                for idx, val in enumerate(fitted_model.params):
                    coefs_list.append({"name": f"x{idx}", "value": float(val)})
                    
        # 3. Scikit-learn Pipelines
        elif model_id in ['lasso', 'ridge', 'elastic_net']:
            step_name = 'elasticnet' if model_id == 'elastic_net' else model_id
            if hasattr(fitted_model, 'named_steps') and step_name in fitted_model.named_steps:
                model_obj = fitted_model.named_steps[step_name]
                if hasattr(model_obj, 'coef_') and hasattr(model_obj, 'intercept_'):
                    coefs_list.append({"name": "const", "value": float(model_obj.intercept_)})
                    indep_vars = params.get('independent_vars', [])
                    for name, val in zip(indep_vars, model_obj.coef_):
                        coefs_list.append({"name": str(name), "value": float(val)})
    except Exception as e:
        print(f"Error extracting coefficients: {e}")
        
    return coefs_list

# الذاكرة المؤقتة القديمة للموديلات داخل السيرفر
MODEL_CACHE = {
    "fitted_model": None,
    "model_id": None,
    "original_params": None,
    "dataframe": None
}

model_execution_bp = Blueprint('model_execution_api', __name__, url_prefix='/api/analysis/model')

# --- دالة Error Handling ---
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


@model_execution_bp.route('/run-model', methods=['POST'])
@login_required 
def run_selected_model():
    """Endpoint لتشغيل النموذج المختار وإضافة توصيات AI وجداول Stargazer الأكاديمية."""
    global MODEL_CACHE 

    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'modelId' not in payload or 'params' not in payload:
            return jsonify({"error": "Missing 'dataset', 'modelId', or 'params' in payload"}), 400

        dataset = payload['dataset']
        model_id = payload['modelId']
        params = payload['params']

        if not isinstance(dataset, list) or not dataset:
            return jsonify({"error": "'dataset' must be a non-empty list."}), 400

        df = pd.DataFrame(dataset)
        results = {}
        ai_recommendations = []
        panel_decision = None
        
        MODEL_CACHE = {} 
        print(f"Running model: {model_id}")

        # --- تشغيل النموذج المحدد بناءً على الـ ID ---
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

        # 🟢 معالجة البانل وحقن الـ Stargazer بأمان مطلق داخلياً 🟢
        elif model_id == 'panel':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars'); panel_id = params.get('panel_id_var'); panel_time = params.get('panel_time_var')
            if not all([dep_var, indep_vars, panel_id, panel_time]): raise ValueError("Missing required parameters for Panel model.")
            
            results = run_panel_suite(df, dep_var, indep_vars, panel_id, panel_time)
            
            pooled_res = results.get('pooled_model_object') or results.get('pooled_res')
            fe_res = results.get('fe_model_object') or results.get('fe_res') or results.get('fitted_model_object')
            re_res = results.get('re_model_object') or results.get('re_res')
            
            if pooled_res and fe_res and re_res:
                models_to_compare = {
                    "Pooled OLS": pooled_res,
                    "Fixed Effects (FE)": fe_res,
                    "Random Effects (RE)": re_res
                }
                stargazer_table = format_to_stargazer_html(models_to_compare, dep_var_name=dep_var)
                results["comparison_html"] = stargazer_table
                results["summary_html"] = stargazer_table  
            
            if AI_CORE_AVAILABLE:
                hausman_res = next((item for item in results.get('diagnostics', []) if 'hausman' in item.get('name', '').lower()), None)
                lm_res = next((item for item in results.get('diagnostics', []) if 'breusch-pagan lm' in item.get('name', '').lower()), None)
                panel_decision = get_panel_model_decision(hausman_result=hausman_res, panel_lm_result=lm_res)
                
                hausman_p = hausman_res.get('p_value', 1.0) if hausman_res else 1.0
                lm_p = lm_res.get('p_value', 1.0) if lm_res else 1.0
                
                if not panel_decision or "defaulting to pooled" in panel_decision.lower() or "pooled ols" in panel_decision.lower():
                    if lm_p > 0.05:
                        panel_decision = "Recommended Panel Model: **Pooled OLS**. Reasoning: Breusch-Pagan LM test p-value > 0.05 suggests no significant cross-sectional variance."
                    else:
                        if hausman_p <= 0.05:
                            panel_decision = "Recommended Panel Model: **Fixed Effects (FE)**. Reasoning: Breusch-Pagan LM rejects Pooled OLS, and Hausman test strictly rejects H0 (p<=0.05), indicating individual effects are correlated with regressors."
                        else:
                            panel_decision = "Recommended Panel Model: **Random Effects (RE)**. Reasoning: Breusch-Pagan LM indicates significant panel variance over Pooled OLS, and Hausman test fails to reject H0 (p>0.05), making RE the most efficient estimator."
        # 🟢 المسار الجديد والمحكم لتشغيل موديل البانل الديناميكي GMM
        # 🟢 المسار الجديد والمحكم لتشغيل موديل البانل الديناميكي GMM مع التنسيق الأكاديمي
        elif model_id == 'panel_gmm':
            dep_var = params.get('dependent_var')
            indep_vars = params.get('independent_vars')
            panel_id = params.get('panel_id_var')
            panel_time = params.get('panel_time_var')
            
            if not all([dep_var, indep_vars, panel_id, panel_time]): 
                raise ValueError("Missing required panel architecture parameters (Y, X, Entity ID, Time ID) for Dynamic GMM.")
                
            results = run_dynamic_panel_gmm(df, dep_var, indep_vars, panel_id, panel_time)

            # 🏛️ اللمسة العبقرية المضافة: تمرير مخرجات GMM لـ Stargazer لتوحيد المظهر الدولي 🏛️
            gmm_fitted_obj = results.get('fitted_model_object')
            if gmm_fitted_obj:
                models_to_format = {
                    "Dynamic Panel GMM": gmm_fitted_obj
                }
                # توليد جدول النشر الفاخر للـ GMM وحقنه في الواجهة فوراً
                stargazer_table = format_to_stargazer_html(models_to_format, dep_var_name=dep_var)
                results["comparison_html"] = stargazer_table
                results["summary_html"] = stargazer_table
                
        elif model_id == 'logit':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Logit.")
            results = run_logit_model(df, dep_var, indep_vars)
            
            if AI_CORE_AVAILABLE:
                lr_p = next((item.get('p_value', 1.0) for item in results.get('diagnostics', []) if 'lr test' in item.get('name', '').lower()), 0.0)
                pseudo_r2 = next((item.get('statistic', 0.0) for item in results.get('diagnostics', []) if 'mcfadden' in item.get('name', '').lower()), 0.0)
                
                if lr_p <= 0.05:
                    status_text = "The model is jointly significant (p<=0.05)."
                    if pseudo_r2 < 0.10:
                        fit_text = f"However, the McFadden Pseudo R2 is low ({pseudo_r2:.4f}), indicating weak explanatory power. We highly recommend adding more structural control variables to reduce Omitted Variable Bias (OVB)."
                    else:
                        fit_text = f"The McFadden Pseudo R2 ({pseudo_r2:.4f}) indicates a strong and reliable goodness-of-fit for binary choice data."
                else:
                    status_text = "Warning: The model is NOT jointly significant (p>0.05)."
                    fit_text = "The selected independent variables fail to explain the probability of the outcome. Consider revising the model specification entirely."
                
                ai_recommendations = [f"Model Specification & Fit: {status_text} {fit_text}"]

        elif model_id == 'probit':
            dep_var = params.get('dependent_var'); indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError("Missing dependent_var or independent_vars for Probit.")
            results = run_probit_model(df, dep_var, indep_vars)
            
            if AI_CORE_AVAILABLE:
                lr_p = next((item.get('p_value', 1.0) for item in results.get('diagnostics', []) if 'lr test' in item.get('name', '').lower()), 0.0)
                pseudo_r2 = next((item.get('statistic', 0.0) for item in results.get('diagnostics', []) if 'mcfadden' in item.get('name', '').lower()), 0.0)
                
                if lr_p <= 0.05:
                    status_text = "The model is jointly significant (p<=0.05)."
                    if pseudo_r2 < 0.10:
                        fit_text = f"However, the McFadden Pseudo R2 is low ({pseudo_r2:.4f}), indicating weak explanatory power. We highly recommend adding more structural control variables to reduce Omitted Variable Bias (OVB)."
                    else:
                        fit_text = f"The McFadden Pseudo R2 ({pseudo_r2:.4f}) indicates a strong and reliable goodness-of-fit for binary choice data."
                else:
                    status_text = "Warning: The model is NOT jointly significant (p>0.05)."
                    fit_text = "The selected independent variables fail to explain the probability of the outcome. Consider revising the model specification entirely."
                
                ai_recommendations = [f"Model Specification & Fit: {status_text} {fit_text}"]
        
        elif model_id in ['random_forest', 'xgboost']:
            dep_var = params.get('dependent_var')
            indep_vars = params.get('independent_vars')
            if not dep_var or not indep_vars: raise ValueError(f"Missing dependent/independent vars for {model_id}.")
            results = run_ml_model(df, model_id, dep_var, indep_vars)
            if AI_CORE_AVAILABLE: ai_recommendations = get_model_diagnostics_recommendations(results.get('diagnostics'))

        else:
            return jsonify({"error": f"Unknown modelId: '{model_id}'"}), 400
         
        if results.get("fitted_model_object"):
            MODEL_CACHE["fitted_model"] = results.get("fitted_model_object")
            MODEL_CACHE["model_id"] = model_id
            MODEL_CACHE["original_params"] = params
            MODEL_CACHE["dataframe"] = df 
            print(f"Model {model_id} successfully fitted and cached (in-memory).")
        else:
            print(f"Warning: Model {model_id} ran, but 'fitted_model_object' was not returned for caching.")

        final_recommendations = []
        if ai_recommendations and isinstance(ai_recommendations, list):
            final_recommendations.extend(ai_recommendations)
        if panel_decision:
            final_recommendations.append(f"Panel Decision Aid: {panel_decision}")
        if final_recommendations:
            results['ai_recommendations_list'] = final_recommendations

        response_data = {
            "summary_html": results.get("summary_html"),
            "comparison_html": results.get("comparison_html"),
            "diagnostics": results.get("diagnostics", []),
            "metrics": results.get("metrics", {}), 
            "ai_recommendations_list": results.get("ai_recommendations_list", []),
            "coefficients": extract_coefficients_from_model(model_id, results.get("fitted_model_object"), params)
        }
        if not response_data["summary_html"] and not response_data["comparison_html"]:
            response_data["summary_html"] = "<p>No standard summary table generated or an error occurred.</p>"

        return jsonify(response_data), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running model '{payload.get('modelId', 'N/A')}'", 500)


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
        suggestion = get_model_suggestion(df=df, y_vars=y_vars, x_vars=x_vars, is_panel=is_panel, panel_id=panel_id, panel_time=panel_time)
        return jsonify(suggestion), 200
    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, "An internal server error occurred while generating suggestion.", 500)


@model_execution_bp.route('/run-post-test', methods=['POST'])
@login_required 
def run_post_estimation_test():
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

        cached_model = MODEL_CACHE.get("fitted_model")
        cached_model_id = MODEL_CACHE.get("model_id")

        if cached_model is None or cached_model_id != original_model_id:
            return handle_error(ValueError("Model cache is empty or mismatched. Please re-run the main model first."), 400)
        
        print(f"Cache hit. Running post-test '{test_id}' on cached {cached_model_id} model.")

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
            raise NotImplementedError(f"Test '{test_id}' function '{post_test_func_name}' not found.")

        if AI_CORE_AVAILABLE and isinstance(results, dict) and results.get('interpretation'):
            if test_id not in ['forecast', 'irf', 'fevd']:
                recs = get_model_diagnostics_recommendations([results])
                if recs and isinstance(recs, list) and len(recs) > 0:
                    if isinstance(recs[0], dict) and "unavailable" not in recs[0].get("text", ""):
                        results['ai_recommendation'] = recs[0].get("text")
        
        return jsonify(results), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running post-test '{payload.get('testId', 'N/A')}'", 500)


@model_execution_bp.route('/run-quick-fix', methods=['POST'])
@login_required 
def run_quick_fix():
    global MODEL_CACHE
    try:
        payload = request.get_json()
        fix_code = payload.get('fixCode')
        
        cached_model_id = MODEL_CACHE.get("model_id")
        original_params = MODEL_CACHE.get("original_params")
        original_df = MODEL_CACHE.get("dataframe")

        if not all([cached_model_id, original_params, original_df is not None]):
            raise ValueError("Cache is invalid. Please re-run the main model.")

        ai_recommendations = []
        new_params = original_params.copy()

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
            cov_type = 'nonrobust'; cov_kwds = None
            if fix_code == 'USE_HAC':
                cov_type = 'HAC'; cov_kwds = {'maxlags': 5}; new_params['cov_type'] = 'HAC'
            elif fix_code == 'USE_ROBUST_HC3':
                cov_type = 'HC3'; new_params['cov_type'] = 'HC3'
            else:
                raise ValueError(f"Unknown fix_code: {fix_code}")
                
            results_fixed = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type=cov_type, cov_kwds=cov_kwds)
        
        elif cached_model_id == 'arima':
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            order = original_params.get('order')
            seasonal_order = original_params.get('seasonal_order')
            
            cov_type = 'robust'
            cov_kwds = None
            new_params['cov_type'] = 'robust'
            
            order_tuple = tuple(order) if order else (1,0,1)
            seasonal_tuple = tuple(seasonal_order) if seasonal_order and len(seasonal_order) == 4 else (0,0,0,0)
            
            results_fixed = run_arima_sarima(
                original_df, 
                endog_var, 
                exog_vars=exog_vars, 
                order=order_tuple, 
                seasonal_order=seasonal_tuple,
                cov_type=cov_type,
                cov_kwds=cov_kwds
            )
            
        else:
            raise ValueError(f"Quick Fix is not implemented for model type: {cached_model_id}")
        
        if AI_CORE_AVAILABLE:
            ai_recommendations = get_model_diagnostics_recommendations(results_fixed.get("diagnostics", []))
        ai_recommendations.append({"text": f"**Quick Fix Applied:** Model re-estimated using **{cov_type}** standard errors."})

        MODEL_CACHE["fitted_model"] = results_fixed.get("fitted_model_object")
        MODEL_CACHE["original_params"] = new_params

        results_for_frontend = {
            "summary_html": results_fixed.get("summary_html"),
            "diagnostics": results_fixed.get("diagnostics", []),
            "metrics": results_fixed.get("metrics", {}),
            "ai_recommendations_list": ai_recommendations,
            "newParams": new_params,
            "coefficients": extract_coefficients_from_model(cached_model_id, results_fixed.get("fitted_model_object"), new_params)
        }
        return jsonify(results_for_frontend), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, f"An internal server error occurred while running Quick Fix", 500)


@model_execution_bp.route('/run-robustness-comparison', methods=['POST'])
@login_required 
def run_robustness_comparison():
    global MODEL_CACHE
    try:
        original_model_id = MODEL_CACHE.get("model_id")
        original_params = MODEL_CACHE.get("original_params")
        original_df = MODEL_CACHE.get("dataframe")

        if not all([original_model_id, original_params, original_df is not None]):
            raise ValueError("Model cache is invalid. Please re-run the main model.")

        comparison_results = []
        
        if original_model_id == 'ols':
            dep_var = original_params.get('dependent_var')
            indep_vars = original_params.get('independent_vars')
            
            results_orig = run_ols_model(original_df, dep_var, indep_vars, cov_type='nonrobust')
            comparison_results.append({ "modelName": "OLS (Non-Robust)", "metrics": results_orig.get("metrics", {}), "diagnostics": results_orig.get("diagnostics", []) })
            
            results_hac = run_ols_model(original_df, dep_var, indep_vars, cov_type='HAC', cov_kwds={'maxlags': 5})
            comparison_results.append({ "modelName": "OLS (HAC Robust)", "metrics": results_hac.get("metrics", {}), "diagnostics": results_hac.get("diagnostics", []) })

            results_hc3 = run_ols_model(original_df, dep_var, indep_vars, cov_type='HC3')
            comparison_results.append({ "modelName": "OLS (HC3 Robust)", "metrics": results_hc3.get("metrics", {}), "diagnostics": results_hc3.get("diagnostics", []) })

        elif original_model_id == 'ardl':
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)

            results_orig = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='nonrobust')
            comparison_results.append({ "modelName": "ARDL (Non-Robust)", "metrics": results_orig.get("metrics", {}), "diagnostics": results_orig.get("diagnostics", []) })
            
            results_hac = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='HAC')
            comparison_results.append({ "modelName": "ARDL (HAC Robust)", "metrics": results_hac.get("metrics", {}), "diagnostics": results_hac.get("diagnostics", []) })

            results_hc3 = run_ardl_model(original_df, endog_var, exog_vars, lags=lags, cov_type='HC3')
            comparison_results.append({ "modelName": "ARDL (HC3 Robust)", "metrics": results_hc3.get("metrics", {}), "diagnostics": results_hc3.get("diagnostics", []) })
        
        else:
            raise ValueError(f"Robustness comparison not implemented for {original_model_id}")

        return jsonify(comparison_results), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, "An internal server error occurred while running robustness comparison", 500)


@model_execution_bp.route('/get-code', methods=['POST'])
@login_required 
def get_model_code_snippet():
    global MODEL_CACHE
    try:
        payload = request.get_json()
        transformation_history = payload.get('transformationHistory', []) 

        model_id = MODEL_CACHE.get("model_id")
        params = MODEL_CACHE.get("original_params")
        df = MODEL_CACHE.get("dataframe")

        if not all([model_id, params, df is not None]):
             raise ValueError("Model cache is invalid. Cannot generate code.")

        code_snippet = generate_code_snippet(model_id, params, df, transformation_history)
        return jsonify({"code_snippet": code_snippet}), 200

    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as user_err:
        return handle_error(user_err)
    except Exception as e:
        return handle_error(e, "An internal server error occurred while generating code", 500)