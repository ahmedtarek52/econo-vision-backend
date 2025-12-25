import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback

def run_forecast(model_id, results, test_params={'periods': 10, 'alpha': 0.05}, **kwargs):
    supported_models = ['arima', 'var', 'vecm', 'ols', 'ardl', 'lasso', 'ridge', 'elastic_net', 'random_forest', 'xgboost', 'garch']
    
    if model_id not in supported_models:
        raise ValueError(f"Forecast generation is not supported for model type: '{model_id}'.")
        
    periods = int(test_params.get('periods', 10))
    alpha = float(test_params.get('alpha', 0.05))
    
    try:
        # --- 0. Prepare Historical Data (Robust Extraction) ---
        # استخراج البيانات التاريخية بطريقة آمنة سواء كانت Pandas أو Numpy
        hist_data_series_or_df = None
        
        # محاولة عامة لجلب البيانات الأصلية
        if hasattr(results, 'model') and hasattr(results.model, 'endog'):
             hist_data_series_or_df = results.model.endog
        elif hasattr(results, 'model') and hasattr(results.model, 'data') and hasattr(results.model.data, 'endog'):
             hist_data_series_or_df = results.model.data.endog
        elif isinstance(results, dict) and "Y_train" in results: # ML Models
             hist_data_series_or_df = pd.concat([results["Y_train"], results["Y_test"]])
        
        # تحويل البيانات التاريخية إلى قائمة أرقام نظيفة (List of Floats) فوراً
        hist_data_values = []
        if hist_data_series_or_df is not None:
            # تحويلها لمصفوفة مسطحة أولاً
            arr = np.array(hist_data_series_or_df).flatten()
            # تحويلها لقائمة بايثون عادية مع استبعاد القيم الفارغة
            hist_data_values = [float(x) for x in arr if pd.notna(x)]

        original_params = kwargs.get('original_params', {})
        plot_data = {}
        interp = ""

        # --- 1. Generate Future Forecast ---
        try:
            # === ARIMA ===
            if model_id == 'arima':
                endog_name = original_params.get('endog_var', 'Y') 
                exog_hist_array = getattr(results.model, 'exog', None)
                original_exog_cols = getattr(results.model, 'exog_names', None)
                exog_future_df_or_array = None
                
                if exog_hist_array is not None and original_exog_cols is not None:
                    future_exog_data = test_params.get('future_exog')
                    if future_exog_data is None or not isinstance(future_exog_data, list) or len(future_exog_data) == 0:
                        raise ValueError(f"Forecasting for ARIMAX requires future values for exogenous variables.")
                    try:
                        exog_future = pd.DataFrame(future_exog_data)
                        future_cols_provided = list(exog_future.columns)
                        # Filter valid columns
                        valid_cols = [col for col in original_exog_cols if col in future_cols_provided]
                        exog_future = exog_future[valid_cols] 

                        if len(exog_future) < periods:
                            raise ValueError(f"Not enough future exog values provided. Need {periods}, got {len(exog_future)}.")
                        exog_future = exog_future.head(periods)
                        for col in exog_future.columns:
                            exog_future[col] = pd.to_numeric(exog_future[col], errors='coerce')
                        exog_future_df_or_array = exog_future.values
                    except Exception as e:
                        raise ValueError(f"Error processing 'future_exog' data: {e}")
                
                fc_result = results.get_forecast(steps=periods, exog=exog_future_df_or_array) 
                fc_mean = fc_result.predicted_mean
                conf_int = fc_result.conf_int(alpha=alpha)
                
                # إعداد البيانات للرسم (Clean Lists)
                fc_data = [round(float(v), 4) for v in fc_mean.values]
                fc_lower = [round(float(v), 4) for v in conf_int.iloc[:, 0].values]
                fc_upper = [round(float(v), 4) for v in conf_int.iloc[:, 1].values]
                
                hist_labels = list(range(1, len(hist_data_values) + 1))
                fc_labels = list(range(len(hist_data_values) + 1, len(hist_data_values) + 1 + periods))
                
                plot_data = {
                    "univariate": True, "variable_name": endog_name,
                    "hist_labels": hist_labels, "hist_data": hist_data_values,
                    "fc_labels": fc_labels, "fc_data": fc_data,
                    "fc_lower": fc_lower, "fc_upper": fc_upper
                }
                interp = f"Generated {periods}-step forecast for {endog_name}."

            # === VAR / VECM ===
            elif model_id == 'var' or model_id == 'vecm':
                var_names = original_params.get('variables', [])
                if not var_names and hasattr(results.model, 'endog_names'):
                    var_names = results.model.endog_names

                # For VAR/VECM, hist_data needs to be shaped correctly (not flattened yet)
                # إعادة جلب البيانات كـ Array للأبعاد المتعددة
                hist_arr = np.array(hist_data_series_or_df)
                
                fc_data_array = None
                fc_lower_array = None
                fc_upper_array = None
                
                if model_id == 'var':
                    y_last_lags = hist_arr[-results.k_ar:] 
                    fc_data_array = results.forecast(y=y_last_lags, steps=periods)
                    try:
                        if hasattr(results, 'forecast_interval'):
                            fc_intervals = results.forecast_interval(y=y_last_lags, steps=periods, alpha=alpha)
                            fc_lower_array, fc_upper_array = fc_intervals[0], fc_intervals[1]
                    except Exception: pass
                else: # VECM
                    fc_data_array = results.predict(steps=periods) 

                hist_labels = list(range(1, len(hist_arr) + 1))
                # تحويل كل عمود إلى قائمة نظيفة
                hist_data = {col: [round(float(v), 4) for v in hist_arr[:, i]] for i, col in enumerate(var_names)}
                fc_labels = list(range(len(hist_arr) + 1, len(hist_arr) + 1 + periods))
                fc_data = {col: [round(float(v), 4) for v in fc_data_array[:, i]] for i, col in enumerate(var_names)}
                
                fc_lower = {}
                fc_upper = {}
                for i, col in enumerate(var_names):
                    fc_lower[col] = [round(float(v), 4) for v in fc_lower_array[:, i]] if fc_lower_array is not None else [None]*periods
                    fc_upper[col] = [round(float(v), 4) for v in fc_upper_array[:, i]] if fc_upper_array is not None else [None]*periods
                
                plot_data = {
                    "univariate": False, "variables": var_names,
                    "hist_labels": hist_labels, "hist_data": hist_data,
                    "fc_labels": fc_labels, "fc_data": fc_data,
                    "fc_lower": fc_lower, "fc_upper": fc_upper
                }
                interp = f"Generated {periods}-step forecast for system."

            # === GARCH ===
            elif model_id == 'garch':
                forecast_result = results.forecast(horizon=periods)
                mean_forecast = forecast_result.mean.iloc[-1].values
                variance_forecast = forecast_result.variance.iloc[-1].values
                endog_name = "Y"
                
                plot_data = {
                    "univariate": True, 
                    "variable_name": f"{endog_name} (Vol)",
                    "hist_labels": [], "hist_data": [],
                    "fc_labels": list(range(1, periods + 1)), 
                    "fc_data": [round(float(v), 4) for v in mean_forecast],
                    "fc_lower": [round(float(v), 4) for v in variance_forecast],
                    "fc_upper": [None] * periods
                }
                interp = f"Generated GARCH forecast."

            # === Conditional Models (OLS, ARDL, ML) ===
            elif model_id in ['ols', 'ardl', 'lasso', 'ridge', 'elastic_net', 'random_forest', 'xgboost']:
                print(f"Forecast: Conditional model ({model_id}) detected.")
                
                future_exog_data = test_params.get('future_exog')
                model_exog_vars = original_params.get('independent_vars') or original_params.get('exog_vars')
                
                if future_exog_data is None:
                    raise ValueError(f"Forecast for {model_id} requires future_exog values.")

                # معالجة البيانات الخارجية
                exog_future_df = pd.DataFrame(future_exog_data)
                
                # التأكد من الأعمدة
                if model_exog_vars:
                    # تساهل في المطابقة إذا كان العدد صحيحاً
                    if not all(col in exog_future_df.columns for col in model_exog_vars):
                         if len(exog_future_df.columns) >= len(model_exog_vars):
                             exog_future_df = exog_future_df.iloc[:, :len(model_exog_vars)]
                             exog_future_df.columns = model_exog_vars
                    else:
                        exog_future_df = exog_future_df[model_exog_vars]
                
                # تنظيف الأرقام
                for col in exog_future_df.columns:
                    exog_future_df[col] = pd.to_numeric(exog_future_df[col], errors='coerce')
                exog_future_df = exog_future_df.fillna(0)

                Y_pred_mean = np.zeros(periods)
                Y_pred_lower = [None] * periods
                Y_pred_upper = [None] * periods
                interp = f"Generated {periods}-step conditional forecast."

                # --- A. OLS ---
                if model_id == 'ols':
                    exog_future_df_with_const = exog_future_df.copy()
                    if hasattr(results.model, 'exog_names') and 'const' in results.model.exog_names:
                        exog_future_df_with_const = sm.add_constant(exog_future_df_with_const, prepend=True, has_constant='add')
                    
                    pred_results = results.get_prediction(exog=exog_future_df_with_const)
                    Y_pred_mean = pred_results.predicted_mean.flatten()
                    conf_int = pred_results.conf_int(alpha=alpha)
                    Y_pred_lower = conf_int[:, 0].flatten()
                    Y_pred_upper = conf_int[:, 1].flatten()

                # --- B. ARDL (Robust Fix) ---
                elif model_id == 'ardl':
                    n_obs = int(results.nobs)
                    
                    # استخدام Index رقمي بسيط للمستقبل لتجنب مشاكل التواريخ المعقدة و numpy attributes
                    start_idx = n_obs
                    end_idx = n_obs + periods - 1
                    
                    # تعيين Index للبيانات الخارجية ليتوافق مع التوقع
                    exog_future_df.index = pd.RangeIndex(start=start_idx, stop=start_idx + periods)

                    try:
                        # التوقع باستخدام أرقام الـ Index
                        Y_pred_mean_series = results.predict(start=start_idx, end=end_idx, exog_oos=exog_future_df)
                        Y_pred_mean = Y_pred_mean_series.values.flatten()
                    except Exception as ardl_e:
                        print(f"ARDL Predict fallback: {ardl_e}")
                        # محاولة بديلة: تمرير القيم فقط
                        Y_pred_mean_series = results.predict(start=start_idx, end=end_idx, exog_oos=exog_future_df.values)
                        Y_pred_mean = Y_pred_mean_series.flatten()

                # --- C. ML Models ---
                else: 
                    model_to_predict = results.get("model") if isinstance(results, dict) else results
                    try:
                        Y_pred_mean = model_to_predict.predict(exog_future_df)
                    except:
                        Y_pred_mean = model_to_predict.predict(exog_future_df.values)

                # --- تجهيز بيانات الرسم (Construction) ---
                # التأكد من أن Y_pred_mean هو قائمة أرقام (List) وليس Numpy Array
                fc_data_clean = [float(x) for x in np.array(Y_pred_mean).flatten() if pd.notna(x)]
                
                # التعامل مع الحدود (Upper/Lower)
                fc_lower_clean = []
                fc_upper_clean = []
                for val in Y_pred_lower:
                    fc_lower_clean.append(float(val) if val is not None and pd.notna(val) else None)
                for val in Y_pred_upper:
                    fc_upper_clean.append(float(val) if val is not None and pd.notna(val) else None)

                # بناء كائن البيانات النهائي
                hist_labels = list(range(1, len(hist_data_values) + 1))
                fc_labels = list(range(len(hist_data_values) + 1, len(hist_data_values) + 1 + len(fc_data_clean)))
                
                plot_data = {
                    "univariate": True, 
                    "variable_name": original_params.get('dependent_var') or 'Y',
                    "hist_labels": hist_labels,
                    "hist_data": hist_data_values,     
                    "fc_labels": fc_labels,
                    "fc_data": fc_data_clean,
                    "fc_lower": fc_lower_clean,
                    "fc_upper": fc_upper_clean
                }

        except Exception as forecast_e:
            print(f"Error during forecast *step*: {forecast_e}")
            traceback.print_exc()
            raise RuntimeError(f"Model forecast generation failed: {forecast_e}")

        # --- 2. Calculate In-Sample Metrics (RMSE, MAE, MAPE) ---
        metrics_html = ""
        try:
            y_true = None
            y_pred = None

            if hasattr(results, 'fittedvalues'): # OLS, ARDL, ARIMA
                 y_pred = results.fittedvalues
                 # محاولة جلب y_true بأمان
                 if hasattr(results.model, 'endog'):
                     y_true = results.model.endog
                 elif hasattr(results.model, 'data') and hasattr(results.model.data, 'endog'):
                     y_true = results.model.data.endog
                 
                 # Align lengths
                 if y_true is not None and len(y_pred) != len(y_true):
                     min_len = min(len(y_pred), len(y_true))
                     y_pred = y_pred[-min_len:]
                     y_true = y_true[-min_len:]
            
            elif isinstance(results, dict) and "Y_train" in results:
                 # ML models
                 model_ml = results.get("model")
                 X_full = pd.concat([results["X_train"], results["X_test"]])
                 y_true = pd.concat([results["Y_train"], results["Y_test"]])
                 y_pred = model_ml.predict(X_full)

            if y_true is not None and y_pred is not None:
                 y_true = np.asarray(y_true).flatten()
                 y_pred = np.asarray(y_pred).flatten()
                 
                 rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                 mae = mean_absolute_error(y_true, y_pred)
                 
                 with np.errstate(divide='ignore', invalid='ignore'):
                    mape_array = np.abs((y_true - y_pred) / y_true)
                    mape_array = mape_array[np.isfinite(mape_array)]
                    mape = np.mean(mape_array) * 100 if len(mape_array) > 0 else np.nan

                 metrics_html = f"""
                 <h4 style="margin-bottom: 10px; font-weight: bold;">In-Sample Accuracy Metrics</h4>
                 <table border="1" style="border-collapse: collapse; width: 100%; font-size: 0.85rem; border-color: #e5e7eb;">
                    <thead>
                        <tr style="background-color: #f3f4f6; color: #374151;">
                            <th style="padding: 8px; border: 1px solid #e5e7eb;">Metric</th>
                            <th style="padding: 8px; border: 1px solid #e5e7eb;">Value</th>
                            <th style="padding: 8px; border: 1px solid #e5e7eb;">Interpretation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>RMSE</strong></td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;">{rmse:.4f}</td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb; color: #6b7280;">Root Mean Squared Error</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>MAE</strong></td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;">{mae:.4f}</td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb; color: #6b7280;">Mean Absolute Error</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;"><strong>MAPE</strong></td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb;">{mape:.2f}%</td>
                            <td style="padding: 8px; border: 1px solid #e5e7eb; color: #6b7280;">Mean Absolute Percentage Error</td>
                        </tr>
                    </tbody>
                 </table>
                 """
        except Exception as e:
            print(f"Could not calc metrics: {e}")
            
        return {
            "plot_data_forecast": plot_data, 
            "interpretation": interp,
            "metrics_html": metrics_html
        }
        
    except (ValueError, KeyError, TypeError, NotImplementedError, RuntimeError) as refit_err:
        print(f"Error during forecast setup: {refit_err}")
        traceback.print_exc()
        raise refit_err
    except Exception as e:
        print(f"Error during forecast setup (outer): {e}")
        traceback.print_exc()
        raise RuntimeError(f"An unexpected error occurred during forecast preparation: {e}") from e

def run_system_equations(model_id, results, test_params={}, **kwargs):
    """
    Generates system equations for VECM, VAR, OLS, ARDL.
    Fixed VECM logic to use alpha/gamma arrays instead of missing 'params'.
    """
    try:
        output = ""

        # --- CASE 1: VECM (Fixed) ---
        if model_id == 'vecm':
            var_names = getattr(results.model, 'endog_names', [])
            n_vars = len(var_names)
            
            # استخراج المصفوفات
            alpha = results.alpha # (n_vars x rank)
            beta = results.beta   # (n_vars x rank)
            gamma = results.gamma # (n_vars x (n_vars * k_ar_diff)) - قد تكون غير موجودة إذا Lags=1
            
            # 1. معادلة التوازن طويلة الأجل (Long Run / ECT)
            coint_eq_str = ""
            # نفترض Rank=1 للتبسيط في العرض (العمود الأول من بيتا)
            # نقوم بالتطبيع (Normalization) بقسمة الكل على معامل المتغير الأول
            norm_factor = beta[0, 0] if beta[0, 0] != 0 else 1.0
            
            for i, name in enumerate(var_names):
                coef = beta[i, 0]
                norm_coef = coef / norm_factor
                
                # في معادلة ECT، ننقل المتغيرات للطرف الآخر (عكس الإشارة) ما عدا المتغير التابع
                val = -1 * norm_coef 
                
                if i == 0: 
                    coint_eq_str += f"{name}"
                else:
                    sign = "+" if val >= 0 else "-"
                    coint_eq_str += f" {sign} {abs(val):.6f}*{name}(-1)"
            
            # إضافة الثابت داخل العلاقة (إن وجد)
            if results.deterministic == 'ci':
                 coint_eq_str += " + Constant(LongRun)"

            output += "<b>System Equations (VECM)</b><br>"
            output += "--------------------------------------------------<br>"
            output += f"<b>Cointegrating Eq (ECT):</b> = {coint_eq_str}<br><br>"

            # 2. معادلات الأجل القصير (Short Run)
            # D(Y_t) = Alpha * ECT + Gamma * D(Y_t-1) ...
            
            # تحديد عدد اللاجات في الفروق
            k_ar_diff = 0
            if gamma is not None and gamma.size > 0:
                # حجم غاما هو (K, K * lags)
                k_ar_diff = gamma.shape[1] // n_vars

            for i, target_var in enumerate(var_names):
                eq_str = f"<b>D({target_var})</b> = "
                
                # أ. حد تصحيح الخطأ (Alpha)
                alpha_val = alpha[i, 0] # لأول علاقة تكامل
                eq_str += f"{alpha_val:.6f} * (ECT)"
                
                # ب. المعاملات المتأخرة (Gamma)
                if k_ar_diff > 0:
                    col_idx = 0
                    for lag in range(1, k_ar_diff + 1):
                        for j, lag_var in enumerate(var_names):
                            # معامل المتغير j عند الإبطاء lag للمعادلة i
                            coef = gamma[i, col_idx]
                            
                            sign = "+" if coef >= 0 else "-"
                            eq_str += f" {sign} {abs(coef):.6f}*D({lag_var}(-{lag}))"
                            
                            col_idx += 1
                
                # ج. الثوابت الخارجية (Constant Outside)
                # في statsmodels، الثابت الخارجي غالباً يظهر في det_coef_coint أو det_coef
                # للتبسيط سنفحص det_coef إذا وجد
                if hasattr(results, 'det_coef') and results.det_coef.size > 0:
                    # نفترض وجود ثابت واحد
                     if results.det_coef.shape[0] > i:
                        const_val = results.det_coef[i]
                        if isinstance(const_val, (list, np.ndarray)): const_val = const_val[0] # safety
                        sign = "+" if const_val >= 0 else "-"
                        eq_str += f" {sign} {abs(const_val):.6f}*C"
                
                output += eq_str + "<br><br>"

        # --- CASE 2: VAR ---
        elif model_id == 'var':
            output += "<b>Vector Autoregression (VAR) System</b><br>"
            output += "--------------------------------------------------<br>"
            
            params_df = results.params
            var_names = results.names
            
            for target in var_names:
                eq_str = f"<b>{target}</b> = "
                coeffs = params_df[target]
                
                first_term = True
                for lag_name, val in coeffs.items():
                    formatted_name = str(lag_name)
                    if 'const' in formatted_name: 
                        formatted_name = "C"
                    elif 'L' in formatted_name and '.' in formatted_name:
                        # تحويل L1.y -> y(-1)
                        parts = formatted_name.split('.')
                        lag_num = parts[0].replace('L', '')
                        var_part = parts[1]
                        formatted_name = f"{var_part}(-{lag_num})"
                    
                    if first_term:
                        eq_str += f"{val:.6f}*{formatted_name}"
                        first_term = False
                    else:
                        sign = "+" if val >= 0 else "-"
                        eq_str += f" {sign} {abs(val):.6f}*{formatted_name}"
                
                output += eq_str + "<br><br>"

        # --- CASE 3: OLS / ARDL / Others ---
        elif hasattr(results, 'params'):
            output += f"<b>Estimation Equation ({model_id.upper()})</b><br>"
            output += "--------------------------------------------------<br>"
            
            # محاولة تحديد اسم المتغير التابع
            dep_var = "Y"
            if hasattr(results.model, 'endog_names'):
                dep_var = results.model.endog_names
            
            eq_str = f"<b>{dep_var}</b> = "
            
            params_series = results.params
            first_term = True
            
            for name, val in params_series.items():
                formatted_name = str(name)
                if formatted_name == 'const': formatted_name = "C"
                
                if first_term:
                    eq_str += f"{val:.6f}*{formatted_name}"
                    first_term = False
                else:
                    sign = "+" if val >= 0 else "-"
                    eq_str += f" {sign} {abs(val):.6f}*{formatted_name}"
            
            output += eq_str

        else:
            return {"formatted_results": "Equation view not available for this model type.", "interpretation": ""}

        return {"formatted_results": output, "interpretation": "These are the estimated equations."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"formatted_results": f"Error generating equations: {e}", "interpretation": "Error"}
    