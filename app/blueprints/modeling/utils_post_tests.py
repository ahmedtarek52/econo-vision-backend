import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import (
    acorr_ljungbox, het_breuschpagan, acorr_breusch_godfrey,
    het_white, linear_reset, het_arch, recursive_olsresiduals
)
from scipy.stats import f, chi2
from sklearn.metrics import mean_squared_error, mean_absolute_error
import traceback
import io
import sys
import warnings

# --- (1) Panel Data Support Setup ---
try:
    # نتحقق من وجود المكتبة الأساسية
    import linearmodels
    from linearmodels.panel import PanelOLS
    PANEL_TESTS_AVAILABLE = True
    print("✅ Successfully imported linearmodels core.")
except ImportError:
    print("WARNING: linearmodels library not found. Panel data post-tests will be disabled.")
    PANEL_TESTS_AVAILABLE = False

# محاولة استيراد دالة الارتباط التسلسلي بشكل منفصل
panel_serial_correlation = None
if PANEL_TESTS_AVAILABLE:
    try:
        from linearmodels.panel.diagnostics import panel_serial_correlation
    except ImportError:
        print("WARNING: 'panel_serial_correlation' not found in linearmodels.")


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import xgboost as xgb
import traceback
import warnings

def train_advanced_ml_model(df, target_col, feature_cols, model_type='random_forest', lags=1, tune=True):
    # تجهيز البيانات
    data = df.copy()
    
    # إنشاء اللاجات (Lags)
    cols_to_lag = [target_col] + feature_cols
    for col in cols_to_lag:
        for lag in range(1, lags + 1):
            data[f'{col}_lag{lag}'] = data[col].shift(lag)
            
    data = data.dropna()
    
    # تحديد Features (اللاجات فقط للتنبؤ التكراري)
    features = [c for c in data.columns if '_lag' in c]
    X = data[features]
    y = data[target_col]
    
    # تقسيم زمني (آخر 20% للاختبار)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # إعداد الـ Pipeline
    pipeline_steps = [('scaler', StandardScaler())]
    params = {}
    model = None
    
    if model_type == 'lasso':
        model = Lasso(random_state=42)
        params = {'model__alpha': [0.001, 0.01, 0.1, 1]}
    elif model_type == 'ridge':
        model = Ridge(random_state=42)
        params = {'model__alpha': [0.01, 0.1, 1, 10]}
    elif model_type == 'elastic_net':
        model = ElasticNet(random_state=42)
        params = {'model__alpha': [0.01, 0.1], 'model__l1_ratio': [0.2, 0.8]}
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        params = {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        params = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
        
    pipeline_steps.append(('model', model))
    pipe = Pipeline(pipeline_steps)
    
    final_model = pipe
    if tune:
        print(f"Training {model_type} with optimization...")
        tscv = TimeSeriesSplit(n_splits=3) # تقليل التقسيم للسرعة
        grid = GridSearchCV(pipe, param_grid=params, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        final_model = grid.best_estimator_
    else:
        final_model.fit(X_train, y_train)
        
    # إرجاع حزمة النتائج (Bundle)
    return {
        "model": final_model,
        "model_type": model_type,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "features": features,
        "last_lags": X.iloc[-1:].values # الصف الأخير للتنبؤ
    }
# --- VAR / VECM Specific Tests ---

def run_var_stability(model_id, results, test_params={}, **kwargs):
    """
    Checks Stability.
    Includes a dynamic manual calculation for VECM that adapts to gamma dimensions.
    """
    try:
        if model_id == 'var':
            # --- VAR Logic (Standard) ---
            roots = results.roots
            is_stable = np.all(np.abs(roots) < 1)
            
            f_roots = "\n".join([f" Root {i+1}: {np.abs(r):.4f}" for i, r in enumerate(roots[:5])])
            if len(roots) > 5: f_roots += "\n ..."
            
            output = f"VAR Stability Check (Roots of Characteristic Polynomial)\n{'-'*50}\n{f_roots}\n"
            interp = f"Model is **{'STABLE' if is_stable else 'UNSTABLE'}**.\n{'All roots lie inside the unit circle (< 1).' if is_stable else 'Warning: At least one root lies outside the unit circle.'}"
            
            return {"formatted_results": output, "interpretation": interp, "is_stable": bool(is_stable)}
            
        elif model_id == 'vecm':
            # --- VECM Logic (Dynamic Manual Calculation) ---
            try:
                # 1. محاولة استخدام الخواص الجاهزة أولاً (الأضمن)
                if hasattr(results, 'var_rep'):
                    roots = results.var_rep.roots
                elif hasattr(results, 'to_var'):
                    roots = results.to_var().roots
                else:
                    # 2. الحساب اليدوي (إذا فشلت الخواص الجاهزة)
                    
                    # أ. استخراج المعاملات
                    alpha = results.alpha
                    beta = results.beta
                    gamma = results.gamma # قد تكون None أو بأبعاد مختلفة
                    
                    K = alpha.shape[0] # عدد المتغيرات
                    
                    # ب. تحديد رتبة التأخير (p)
                    # p هو عدد لاجات VAR (مستوى) = لاجات VECM (فروق) + 1
                    if hasattr(results, 'k_ar'):
                        p = results.k_ar
                    else:
                        # استنتاج p من حجم gamma
                        if gamma is not None and gamma.size > 0:
                            # gamma size = K * K * (p-1)
                            p = (gamma.size // (K*K)) + 1
                        else:
                            p = 1

                    # ج. حساب مصفوفة Pi
                    pi = np.dot(alpha, beta.T)
                    
                    # د. تجهيز Gamma (معالجة الأبعاد الديناميكية)
                    gammas_list = []
                    if p > 1:
                        if gamma is None:
                            raise ValueError("p > 1 but gamma is None")
                        
                        # التأكد من الشكل الصحيح: نريد قائمة مصفوفات (K, K)
                        # gamma قد تكون (K, K*(p-1)) أو (K*(p-1), K)
                        
                        raw_gamma = gamma
                        
                        # الحالة 1: مكدسة أفقياً (K, K*(p-1)) -> الشائع في statsmodels
                        if raw_gamma.shape[0] == K and raw_gamma.shape[1] == K*(p-1):
                            for i in range(p-1):
                                gammas_list.append(raw_gamma[:, i*K : (i+1)*K])
                                
                        # الحالة 2: مكدسة عمودياً (K*(p-1), K)
                        elif raw_gamma.shape[0] == K*(p-1) and raw_gamma.shape[1] == K:
                            for i in range(p-1):
                                gammas_list.append(raw_gamma[i*K : (i+1)*K, :])
                        
                        else:
                            # محاولة Reshape يائسة إذا كانت الأبعاد غير متوقعة
                            try:
                                flat = raw_gamma.flatten()
                                for i in range(p-1):
                                    start = i*K*K
                                    end = (i+1)*K*K
                                    gammas_list.append(flat[start:end].reshape(K, K))
                            except:
                                raise ValueError(f"Unknown gamma shape: {raw_gamma.shape}")

                    # هـ. تحويل VECM إلى VAR Matrices (A1...Ap)
                    # المعادلات:
                    # A1 = Pi + I + Gamma_1
                    # Ai = Gamma_i - Gamma_{i-1}
                    # Ap = -Gamma_{p-1}
                    
                    A_matrices = []
                    I = np.eye(K)
                    
                    if p == 1:
                        A_matrices.append(pi + I)
                    else:
                        # A1
                        A_matrices.append(pi + I + gammas_list[0])
                        
                        # A2 to Ap-1
                        for i in range(1, p-1):
                            A_matrices.append(gammas_list[i] - gammas_list[i-1])
                            
                        # Ap
                        A_matrices.append(-gammas_list[-1])

                    # و. بناء المصفوفة المصاحبة (Companion Matrix)
                    comp_dim = p * K
                    comp_matrix = np.zeros((comp_dim, comp_dim))
                    
                    # الصف الأول: A1, A2, ..., Ap
                    for i in range(p):
                        comp_matrix[:K, i*K:(i+1)*K] = A_matrices[i]
                    
                    # القطر الفرعي: مصفوفات الوحدة
                    if p > 1:
                        comp_matrix[K:, :(p-1)*K] = np.eye((p-1)*K)
                    
                    # ز. حساب الجذور
                    eigvals = np.linalg.eigvals(comp_matrix)
                    roots = np.abs(eigvals)

            except Exception as e:
                # طباعة الخطأ الحقيقي للمستخدم لكي نتمكن من تشخيصه إذا فشل مرة أخرى
                return {
                    "formatted_results": "Stability Check Failed", 
                    "interpretation": f"Error calculating roots: {str(e)}. \n\nFallback Advice: Since you are using VECM, rely on the Johansen Cointegration Test result. If Rank > 0, the system is constrained and stable in the long run."
                }

            # 3. عرض النتائج
            roots = np.sort(roots)[::-1] # ترتيب تنازلي
            max_root = np.max(roots)
            is_stable = max_root < 1.001 # السماح بجذور الوحدة
            
            f_roots = "\n".join([f" Root {i+1}: {r:.4f}" for i, r in enumerate(roots[:5])])
            if len(roots) > 5: f_roots += "\n ..."

            output = f"VECM Stability Check\n(Roots of Companion Matrix)\n{'-'*50}\nMax Modulus: {max_root:.6f}\n\nTop Roots:\n{f_roots}"
            
            interp = "VECM Stability:\n"
            if is_stable:
                interp += f"**Model appears STABLE**.\nMax root ({max_root:.4f}) is effectively 1 (Unit Root) or less."
            else:
                interp += f"**Warning: UNSTABLE**.\nMax root ({max_root:.4f}) is > 1."
                
            return {"formatted_results": output, "interpretation": interp, "is_stable": bool(is_stable)}

    except Exception as e:
        return {"formatted_results": "Stability check failed.", "interpretation": f"Critical Error: {e}"}

def run_irf(model_id, results, test_params={'periods': 10, 'orth': True, 'signif': 0.05}, **kwargs):
    """Generates IRF data with Confidence Intervals."""
    try:
        periods = test_params.get('periods', 10)
        orth = test_params.get('orth', True)
        signif = test_params.get('signif', 0.05) # مستوى المعنوية (0.05 = 95%)
        variables = getattr(results.model, 'endog_names', [])
        
        if model_id not in ['var', 'vecm']:
            raise ValueError("IRF test only valid for VAR/VECM models.")

        # حساب IRF
        irf_res = results.irf(periods=periods)
        
        # القيم المتوسطة (Mean)
        irf_values = irf_res.orth_irfs if orth else irf_res.irfs
        
        # القيم المعيارية للأخطاء (Standard Errors)
        # ملاحظة: statsmodels يحسبها تلقائياً عند طلب irf_res
        # نحتاج لاستخراجها لحساب الحدود
        # stderr() تعيد مصفوفة الأخطاء المعيارية
        irf_stderr = irf_res.stderr(orth=orth)
        
        # حساب القيمة الحرجة (Critical Value) للتوزيع الطبيعي
        from scipy.stats import norm
        crit = norm.ppf(1 - signif / 2)

        plot_data = {}; irf_type = "Orthogonalized" if orth else "Standard"
        
        for i, imp_var in enumerate(variables): # الصدمة من
            plot_data[imp_var] = {}
            for j, res_var in enumerate(variables): # الاستجابة لـ
                
                # استخراج البيانات
                mean_val = irf_values[:, j, i] # الاستجابة
                std_err = irf_stderr[:, j, i]  # الخطأ المعياري
                
                # حساب الحدود (Upper & Lower Bounds)
                lower = mean_val - crit * std_err
                upper = mean_val + crit * std_err

                plot_data[imp_var][res_var] = {
                    'periods': list(range(periods + 1)),
                    'response': [round(v, 5) for v in mean_val],
                    'lower': [round(v, 5) for v in lower], # (إضافة)
                    'upper': [round(v, 5) for v in upper]  # (إضافة)
                }

        interp = f"Generated {irf_type} Impulse Response Functions (IRFs) with {int((1-signif)*100)}% Confidence Intervals."
        return {"plot_data_irf": plot_data, "interpretation": interp}
        
    except Exception as e:
        print(f"IRF Error: {e}")
        return {"plot_data_irf": None, "interpretation": f"Error generating IRF: {e}"}

def run_fevd(model_id, results, test_params={'periods': 10}, **kwargs):
    """
    Calculates FEVD with S.E. column (EViews Style).
    Fixes 'Index out of bounds' error by adjusting array indexing.
    """
    try:
        # التأكد من أن المدخل رقم صحيح
        periods = int(test_params.get('periods', 10))
        variable_names = getattr(results.model, 'endog_names', [])
        n_vars = len(variable_names)
        
        # مصفوفات لتخزين النتائج
        decomp = None
        mse_values = None # لتخزين S.E.

        # -------------------------------------------------------
        # المسار 1: VAR
        # -------------------------------------------------------
        if model_id == 'var':
            # 1. حساب FEVD
            # statsmodels .fevd(periods) عادة يعيد فترات من 0 إلى periods-1
            fevd_obj = results.fevd(periods=periods)
            decomp = fevd_obj.decomp 
            
            # 2. حساب S.E. (Standard Error)
            # results.mse(periods) يعيد مصفوفة بحجم periods (index 0 = Period 1)
            mse = results.mse(periods) 
            mse_values = np.sqrt(np.diagonal(mse, axis1=1, axis2=2)) # (periods, k)

        # -------------------------------------------------------
        # المسار 2: VECM
        # -------------------------------------------------------
        elif model_id == 'vecm':
            irf = results.irf(periods=periods)
            orth_irfs = irf.orth_irfs 
            sq_irfs = orth_irfs ** 2
            cum_sq_irfs = np.cumsum(sq_irfs, axis=0)
            
            # حساب التباين الكلي (MSE)
            total_variance = np.sum(cum_sq_irfs, axis=2) # (Time x Vars)
            mse_values = np.sqrt(total_variance) # S.E.
            
            # حساب النسب المئوية
            decomp = np.zeros_like(cum_sq_irfs)
            
            # تجنب القسمة على صفر
            with np.errstate(divide='ignore', invalid='ignore'):
                 for t_idx in range(total_variance.shape[0]):
                     for j in range(n_vars):
                         tv = total_variance[t_idx, j]
                         if tv > 1e-9:
                             decomp[t_idx, j, :] = cum_sq_irfs[t_idx, j, :] / tv
                         else:
                             decomp[t_idx, j, :] = 0.0

        # -------------------------------------------------------
        # بناء جدول HTML بنمط EViews
        # -------------------------------------------------------
        html_rows = ""
        
        # EViews Style CSS
        table_style = "border-collapse: collapse; width: 100%; font-family: 'Courier New', Courier, monospace; font-size: 0.9em;"
        header_style = "border-bottom: 2px solid #000; text-align: right; padding: 5px;"
        cell_style = "text-align: right; padding: 4px 8px; border: none;"
        row_border = "border-bottom: 1px solid #ddd;"

        # العرض: نبدأ من الفترة 1 إلى periods
        display_range = range(1, periods + 1)

        for j, target_var in enumerate(variable_names):
            html_rows += f"<div style='margin-top:20px; font-weight:bold; text-align:left;'>Variance Decomposition of {target_var}:</div>"
            html_rows += f"<table style='{table_style}'>"
            
            # Header
            html_rows += "<tr>"
            html_rows += f"<th style='{header_style} width:50px;'>Period</th>"
            html_rows += f"<th style='{header_style} width:80px;'>S.E.</th>"
            for v in variable_names:
                html_rows += f"<th style='{header_style}'>{v}</th>"
            html_rows += "</tr>"
            
            # Rows
            for t in display_range:
                # معالجة الفهرس (Index Handling) لتجنب الخطأ
                # t يبدأ من 1. المصفوفات في بايثون تبدأ من 0.
                
                idx = t - 1 # الافتراضي لـ VAR (index 0 هو period 1)
                
                if model_id == 'vecm':
                    # VECM IRF usually includes period 0 at index 0
                    # So Period 1 is at index 1
                    idx = t 
                    if idx >= mse_values.shape[0]: 
                         idx = mse_values.shape[0] - 1 # Safety cap

                elif model_id == 'var':
                     # Safety cap for VAR
                     if idx >= mse_values.shape[0]:
                         idx = mse_values.shape[0] - 1
                
                # S.E. Value extraction
                se_val = 0.0
                if mse_values is not None and idx < mse_values.shape[0]:
                    se_val = mse_values[idx, j]
                
                # Percentage extraction
                row_percentages = []
                for i in range(n_vars):
                    pct = 0.0
                    if decomp is not None and idx < decomp.shape[0]:
                        pct = decomp[idx, j, i] * 100
                    row_percentages.append(pct)

                html_rows += f"<tr style='{row_border}'>"
                html_rows += f"<td style='{cell_style} font-weight:bold;'>{t}</td>"
                html_rows += f"<td style='{cell_style}'>{se_val:.6f}</td>"
                
                for pct in row_percentages:
                    html_rows += f"<td style='{cell_style}'>{pct:.6f}</td>"
                html_rows += "</tr>"
            
            html_rows += "</table><br>"

        # تجهيز بيانات الرسم (Chart Data)
        plot_data_fevd = {}
        if decomp is not None:
            available_steps = decomp.shape[0]
            for j, target_var in enumerate(variable_names):
                data_for_var = []
                matrix_for_var = decomp[:, j, :]
                for step_idx in range(available_steps):
                    # تخطي الفترة 0 في الرسم لـ VECM إذا أردت، أو عرضها
                    row_dict = {"period": step_idx} # 0, 1, 2...
                    if model_id == 'var': row_dict['period'] = step_idx + 1 # VAR usually starts at 1
                    
                    for i, shock_var in enumerate(variable_names):
                        row_dict[shock_var] = float(round(matrix_for_var[step_idx, i] * 100, 2))
                    data_for_var.append(row_dict)
                plot_data_fevd[target_var] = data_for_var

        interp = f"Variance Decomposition (Cholesky) for {periods} periods. <br><b>Ordering:</b> {' '.join(variable_names)}"
        
        return {
            "html_table": html_rows, 
            "plot_data_fevd": plot_data_fevd, 
            "interpretation": interp
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"formatted_results": f"Error: {e}", "interpretation": f"Analysis failed: {e}"}
    
    
def run_var_normality(model_id, results, test_params={}, **kwargs):
    """Runs Jarque-Bera on cached VAR/VECM residuals."""
    try:
        if model_id not in ['var', 'vecm']:
            raise ValueError("Normality test only valid for VAR/VECM models.")
            
        residuals = results.resid
        var_names = getattr(results.model, 'endog_names', [f'Var{i+1}' for i in range(residuals.shape[1])])

        if residuals is None or residuals.shape[0] < 2: raise ValueError("Insufficient residuals.")
        if isinstance(residuals, pd.DataFrame): residuals = residuals.values
        if residuals.ndim == 1: residuals = residuals.reshape(-1, 1)

        jb_stat, jb_pvalue, skew, kurt = jarque_bera(residuals, axis=0)

        output = f"Jarque-Bera Normality Test (VAR/VECM Residuals)\nNull: Residuals are normally distributed\n{'-'*50}\n"
        interp = "Overall Residual Normality:\n"; all_normal = True
        for i, name in enumerate(var_names):
            p_val = jb_pvalue[i] if isinstance(jb_pvalue, (np.ndarray, list)) and i < len(jb_pvalue) else jb_pvalue
            stat_val = jb_stat[i] if isinstance(jb_stat, (np.ndarray, list)) and i < len(jb_stat) else jb_stat
            is_normal = p_val > 0.05
            if not is_normal: all_normal = False
            output += f"- {name}: Stat={stat_val:.2f}, P-Val={p_val:.4f} -> {'Normal' if is_normal else 'Non-Normal'}\n"
            interp += f"- Residuals for {name} {'appear normal' if is_normal else '**do NOT appear normal**'}.\n"
        interp += f"\nConclusion: {'The residuals appear to conform to normality assumptions.' if all_normal else 'Non-normality detected in residuals.'}"
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "VAR/VECM Normality test failed.", "interpretation": f"Error: {e}"}

def run_var_autocorr(model_id, results, test_params={'lags': 5}, **kwargs):
    """Checks for Serial Correlation (Robust for VAR & VECM)."""
    try:
        if model_id not in ['var', 'vecm']:
            raise ValueError("Autocorrelation test only valid for VAR/VECM models.")

        lags_test = test_params.get('lags', 5)
        variable_names = getattr(results.model, 'endog_names', [])
        
        # (!!!) استخراج البواقي بشكل آمن (!!!)
        # statsmodels VAR results.resid is typically a numpy array or DataFrame
        residuals = results.resid
        
        # التحويل إلى numpy array لضمان الفهرسة الصحيحة
        if hasattr(residuals, 'values'):
            residuals = residuals.values
        elif isinstance(residuals, list):
            residuals = np.array(residuals)
            
        # التأكد من الأبعاد (N_obs x N_vars)
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)
            
        n_eqs = residuals.shape[1]
        
        output = f"Residual Portmanteau Test for Autocorrelation\n(Ljung-Box Test per Equation)\n{'-'*60}\n"
        output += f"{'Equation':<20} | {'Lag':<5} | {'Stat':<10} | {'P-Value':<10} | {'Result'}\n"
        output += "-"*60 + "\n"
        
        significant_corr_found = False
        
        for i in range(n_eqs):
            var_name = variable_names[i] if i < len(variable_names) else f"Eq_{i+1}"
            
            # الحصول على البواقي للمتغير i (العمود i)
            res_series = residuals[:, i] 
            
            # تشغيل Ljung-Box
            # (التأكد من إزالة NaN إن وجد)
            res_series = res_series[~np.isnan(res_series)]
            
            if len(res_series) <= lags_test:
                 output += f"{var_name:<20} | {lags_test:<5} | {'N/A':<10} | {'N/A':<10} | Insuff Data\n"
                 continue

            lb_df = acorr_ljungbox(res_series, lags=[lags_test], return_df=True)
            lb_stat = lb_df.iloc[0]['lb_stat']
            lb_pval = lb_df.iloc[0]['lb_pvalue']
            
            is_corr = lb_pval <= 0.05
            if is_corr: significant_corr_found = True
            
            res_str = "**Corr**" if is_corr else "No Corr"
            output += f"{var_name:<20} | {lags_test:<5} | {lb_stat:<10.4f} | {lb_pval:<10.4f} | {res_str}\n"

        interp = ""
        if significant_corr_found:
            interp = "Conclusion: **REJECT H0**. Significant serial correlation detected in at least one equation. Consider increasing lags."
        else:
            interp = "Conclusion: **FAIL TO REJECT H0**. No significant serial correlation detected at this lag order. The model captures dynamics well."

        return {"formatted_results": output, "interpretation": interp}
        
    except Exception as e:
        print(f"Autocorr Error: {e}")
        import traceback
        traceback.print_exc()
        return {"formatted_results": "Autocorr check failed.", "interpretation": f"Error: {e}"}

def run_vecm_long_run(model_id, results, test_params={}, **kwargs):
    """Extracts the Normalized Long-Run Equation."""
    if model_id != 'vecm':
        return {"interpretation": "Only for VECM models."}
    
    try:
        # الحصول على معاملات بيتا (Beta)
        beta = results.beta 
        if beta is None: raise ValueError("No beta matrix found.")
        
        var_names = getattr(results.model, 'endog_names', [])
        
        # التطبيع على المتغير الأول (عادة هو المتغير التابع في المعادلة الأولى)
        # ملاحظة: statsmodels قد لا يطبعها بشكل تلقائي مثل EViews
        # نقسم كل المعاملات على معامل المتغير الأول
        norm_factor = beta[0, 0] 
        normalized_beta = beta[:, 0] / norm_factor # العمود الأول يمثل العلاقة الأولى
        
        equation_parts = []
        target_var = var_names[0]
        
        # بناء نص المعادلة: Y = -beta_1*X1 - beta_2*X2 ...
        # (نقلب الإشارة لأن statsmodels يعطي المعادلة الصفرية Beta*Y = 0)
        for i, coef in enumerate(normalized_beta):
            if i == 0: continue # تخطي المتغير التابع
            
            var_name = var_names[i] if i < len(var_names) else "Const"
            # عكس الإشارة للنقل للطرف الآخر
            val = -1 * coef 
            sign = "+" if val >= 0 else "-"
            equation_parts.append(f"{sign} {abs(val):.4f} ({var_name})")
            
        # البحث عن الثابت (Constant) إذا كان موجوداً داخل علاقة التكامل
        if results.deterministic == 'ci':
             # الثابت يكون عادة آخر عنصر في مصفوفة بيتا الممتدة
             # لكن الوصول إليه معقد قليلاً في statsmodels، لذا سنكتفي بالمتغيرات
             pass

        eq_str = f"{target_var} = " + " ".join(equation_parts)
        
        output = "Estimated Long-Run Relationship (Normalized):\n" + "-"*40 + "\n"
        output += eq_str
        
        return {"formatted_results": output, "interpretation": "This equation represents the long-term equilibrium. Coefficients show the impact of each variable on the target."}
        
    except Exception as e:
        return {"interpretation": f"Error extracting equation: {e}"}
# --- OLS / ARDL Specific Post-Tests ---

def run_ramsey_reset(model_id, results, test_params={'power': 2}, **kwargs):
    try:
        if model_id not in ['ols', 'ardl', 'lasso', 'elastic_net', 'ridge']:
            raise ValueError("Ramsey RESET test is typically applied to OLS/ARDL models.")
        
        results_to_use = results
        
        # ARDL Reconstruction Logic
        if model_id == 'ardl':
            original_df = kwargs.get('original_df')
            original_params = kwargs.get('original_params')
            if original_df is None or original_params is None:
                raise ValueError("Could not retrieve original data/params from cache for RESET test.")

            from .utils import ARDL 
            if ARDL is None: raise ImportError("ARDL model not found.")

            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)
            
            model_data = original_df[[endog_var] + exog_vars]
            endog = model_data[endog_var]
            exog = model_data[exog_vars]

            ardl_model_dummy = ARDL(endog, lags=lags, exog=exog, order=lags, trend='c')
            
            Y_ols_dummy = ardl_model_dummy.endog
            X_ols_dummy = ardl_model_dummy.exog
            
            ols_model_for_reset = sm.OLS(Y_ols_dummy, X_ols_dummy)
            results_to_use = ols_model_for_reset.fit()
        
        power = max(2, test_params.get('power', 2))
        reset_test = linear_reset(results_to_use, power=power, use_f=True)
        
        output = f"Ramsey RESET Test (Model Specification)\nNull: Model has no omitted variables (specification is adequate)\n{'-'*50}\nF-Statistic: {reset_test.fvalue:.4f}, P-Value: {reset_test.pvalue:.4f}, Powers: 2 to {power}"
        correct_spec = reset_test.pvalue > 0.05
        interp = f"Conclusion (p={reset_test.pvalue:.4f}): {'FAIL TO REJECT H0 -> No significant evidence of misspecification found.' if correct_spec else 'REJECT H0 -> **Model misspecification likely present**.'}"
        if not correct_spec: interp += " Consider adding non-linear terms (e.g., squares of predictors) or checking for omitted relevant variables."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        print(f"Error during Ramsey RESET test: {e}")
        traceback.print_exc()
        return {"formatted_results": "RESET test failed.", "interpretation": f"Error: {e}"}

def run_white_test(model_id, results, test_params={}, **kwargs):
    try:
        if model_id not in ['ols', 'ardl', 'lasso', 'elastic_net', 'ridge']:
            raise ValueError("White test is typically applied to OLS/ARDL models.")
        
        results_to_use = results
        
        # ARDL Reconstruction
        if model_id == 'ardl':
            original_df = kwargs.get('original_df')
            original_params = kwargs.get('original_params')
            if original_df is None or original_params is None:
                raise ValueError("Could not retrieve original data/params from cache for White test.")
            from .utils import ARDL 
            if ARDL is None: raise ImportError("ARDL model not found.")
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)
            model_data = original_df[[endog_var] + exog_vars]
            endog = model_data[endog_var]
            exog = model_data[exog_vars]
            ardl_model_dummy = ARDL(endog, lags=lags, exog=exog, order=lags, trend='c')
            Y_ols_dummy = ardl_model_dummy.endog
            X_ols_dummy = ardl_model_dummy.exog
            ols_model_for_white = sm.OLS(Y_ols_dummy, X_ols_dummy)
            results_to_use = ols_model_for_white.fit()
        
        resid = results_to_use.resid
        exog = results_to_use.model.exog
        
        if resid is None or exog is None: raise ValueError("Residuals or Exog missing.")

        if exog.ndim == 1:
            exog = exog[:, np.newaxis]
        
        # Check for constant
        has_const = np.any(np.isclose(np.var(exog, axis=0), 0) & np.isclose(np.mean(exog, axis=0), 1))
        
        exog_for_test = exog
        if not has_const:
            exog_for_test = sm.add_constant(exog, prepend=True)
            
        if exog_for_test.shape[1] < 2:
             raise ValueError(f"White's test requires at least one regressor (non-constant). Model exog shape: {exog.shape}")

        n_obs = len(resid)
        n_regressors_original = exog_for_test.shape[1]
        k_vars = n_regressors_original - 1 
        n_regressors_white = k_vars + (k_vars * (k_vars + 1) // 2) + 1 
        
        if n_obs <= n_regressors_white :
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, exog_for_test)
            output = f"Breusch-Pagan Test (Heteroskedasticity - due to insufficient obs for White)\nNull: Homoskedasticity\n{'-'*50}\nLM Stat: {bp_stat:.4f}, P-Value: {bp_pvalue:.4f}"
            homoskedastic = bp_pvalue > 0.05
            interp = f"Conclusion (Breusch-Pagan, p={bp_pvalue:.4f}): {'FAIL TO REJECT H0 -> No evidence of heteroskedasticity found.' if homoskedastic else 'REJECT H0 -> **Heteroskedasticity likely present**.'}"
        else:
            white_stat, white_pvalue, _, _ = het_white(resid, exog_for_test)
            output = f"White Test (Heteroskedasticity - General Form)\nNull: Homoskedasticity (constant variance)\n{'-'*50}\nLM Statistic: {white_stat:.4f}, P-Value: {white_pvalue:.4f}"
            homoskedastic = white_pvalue > 0.05
            interp = f"Conclusion (White, p={white_pvalue:.4f}): {'FAIL TO REJECT H0 -> No significant evidence of heteroskedasticity found.' if homoskedastic else 'REJECT H0 -> **Heteroskedasticity likely present**.'}"
        if not homoskedastic: interp += " Consider using Robust Standard Errors for inference."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        print(f"Error during White test: {e}")
        traceback.print_exc()
        return {"formatted_results": "White/Breusch-Pagan test failed.", "interpretation": f"Error: {e}"}

def run_jarque_bera_resid(model_id, results, test_params={}, **kwargs):
    try:
        jb_stat, jb_pvalue, skew, kurt = (None, None, None, None)
        if hasattr(results, 'test_normality'):
            try:
                test_results_list = results.test_normality(method='jarquebera')
                if test_results_list:
                    jb_stat, jb_pvalue, skew, kurt = test_results_list[0]
            except Exception as sarimax_e:
                print(f"model_results.test_normality() failed: {sarimax_e}. Falling back.")
        if jb_stat is None:
            print("Falling back to manual JB test.")
            resid = getattr(results, 'standardized_residuals', None)
            if resid is None: resid = getattr(results, 'std_resid', None)
            if resid is None: resid = results.resid
            resid = resid.dropna()
            if resid.ndim > 1: resid = resid.flatten()
            if resid.shape[0] < 3: raise ValueError("Not enough non-NaN residuals for Jarque-Bera.")
            jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)
        if jb_stat is None: raise ValueError("Could not determine JB statistic.")
        skew_val = skew if skew is not None else np.nan
        kurt_val = kurt if kurt is not None else np.nan
        output = f"Residual Normality Test (Jarque-Bera)\nNull: Residuals are normally distributed\n{'-'*50}\n"
        output += f"Statistic: {float(jb_stat):.4f}\nP-Value:      {float(jb_pvalue):.4f}\nSkewness:    {float(skew_val):.4f}\nKurtosis:    {float(kurt_val):.4f}"
        is_normal = float(jb_pvalue) > 0.05
        interp = f"Conclusion (p={float(jb_pvalue):.4f}): {'FAIL TO REJECT H0 -> Residuals appear normally distributed.' if is_normal else 'REJECT H0 -> **Do NOT appear normally distributed**.'}"
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "Jarque-Bera residual test failed.", "interpretation": f"Error: {e}"}

def run_durbin_watson(model_id, results, test_params={}, **kwargs):
    try:
        if not hasattr(results, 'resid'):
            raise ValueError("Durbin-Watson test is only applicable to models with .resid attribute.")
        resid = results.resid
        if resid is None or resid.empty: raise ValueError("Could not get residuals.")
        dw_stat = durbin_watson(resid)
        output = f"Durbin-Watson Test (First-Order Serial Correlation)\n{'-'*50}\nStatistic: {dw_stat:.4f}"
        interp = f"DW Statistic = {dw_stat:.4f}.\n"
        interp += "Interpretation Guide (approximate):\n"
        interp += "- Values near 2.0 indicate no first-order autocorrelation.\n"
        interp += "- Values towards 0.0 suggest positive serial correlation.\n"
        interp += "- Values towards 4.0 suggest negative serial correlation.\n"
        interp += "Formal conclusion requires comparing to critical values (d_L, d_U)."
        if 1.5 < dw_stat < 2.5: interp += "\nRule of Thumb: Little evidence of first-order autocorrelation."
        elif dw_stat <= 1.5: interp += "\nRule of Thumb: Suggests potential **positive serial correlation**."
        else: interp += "\nRule of Thumb: Suggests potential **negative serial correlation**."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "Durbin-Watson test failed.", "interpretation": f"Error: {e}"}


# --- ARIMA / GARCH Tests ---
def run_ljung_box(model_id, results, test_params={'lags': 10}, **kwargs):
    try:
        if model_id != 'arima':
            raise ValueError("Ljung-Box test here is intended for ARIMA/SARIMA models.")
        resid = getattr(results, 'standardized_residuals', None)
        if resid is None: resid = results.resid
        if resid is None or not hasattr(resid, 'shape') or resid.shape[0] == 0:
            raise ValueError("Could not get residuals.")
        resid = resid.dropna()
        if resid.ndim > 1: resid = resid.flatten()
        lags_input = test_params.get('lags', 10)
        lags_list = [lags_input] if isinstance(lags_input, int) else lags_list 
        if not lags_list or lags_list[-1] < 1: raise ValueError("Lags must be positive.")
        max_allowed_lags = len(resid) - 1
        if lags_list[-1] >= max_allowed_lags:
            lags_list = [max(1, max_allowed_lags -1)]
            print(f"Warning: Reduced Ljung-Box lags to {lags_list[0]} due to sample size.")
        if len(resid) <= lags_list[-1]:
            raise ValueError(f"Not enough residuals ({len(resid)}) for Ljung-Box test with lags={lags_list[-1]}.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb_results_df = acorr_ljungbox(resid, lags=lags_list, return_df=True, boxpierce=False)
        if lb_results_df.empty: raise ValueError("Could not calculate Ljung-Box statistic.")
        lb_pval = lb_results_df['lb_pvalue'].iloc[-1]
        lb_stat = lb_results_df['lb_stat'].iloc[-1]
        output = f"Ljung-Box Test (ARIMA Residual Autocorrelation)\nNull: No autocorrelation up to lag {lags_list[-1]}\n{'-'*50}\n"
        output += lb_results_df.round(4).to_string()
        no_autocorr = lb_pval > 0.05
        interp = f"Conclusion (p={lb_pval:.4f} at lag {lags_list[-1]}): {'FAIL TO REJECT H0 -> Residuals appear uncorrelated (White Noise).' if no_autocorr else 'REJECT H0 -> Residuals **appear serially CORRELATED**.'}"
        if not no_autocorr: interp += " The model may not fully capture the autocorrelation. Consider adjusting ARIMA/SARIMA orders (p,q, P,Q)."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "Ljung-Box test failed.", "interpretation": f"Error: {e}"}

def run_arch_lm(model_id, results, test_params={'lags': 5}, **kwargs):
    try:
        if not hasattr(results, 'resid'):
            raise ValueError("ARCH-LM test requires a model with residuals.")
        resid = getattr(results, 'standardized_residuals', None)
        if resid is None: resid = results.resid
        if resid is None or not hasattr(resid, 'shape') or resid.shape[0] == 0:
            raise ValueError("Could not get residuals.")
        resid = resid.dropna()
        if resid.ndim > 1: resid = resid.flatten()
        lags = test_params.get('lags', 5)
        if lags < 1: raise ValueError("Lags must be positive.")
        resid_clean = resid[~np.isnan(resid)]
        if len(resid_clean) <= lags: raise ValueError(f"Not enough non-NaN residuals ({len(resid_clean)}) for ARCH-LM test with lags={lags}.")
        arch_test_stat, arch_pvalue, _, _ = het_arch(resid_clean, nlags=lags)
        output = f"ARCH-LM Test (Heteroskedasticity in Residuals)\nNull: No ARCH effects up to lag {lags}\n{'-'*50}\nLM Statistic: {arch_test_stat:.4f}, P-Value: {arch_pvalue:.4f}"
        arch_present = arch_pvalue <= 0.05
        interp = f"Conclusion (p={arch_pvalue:.4f}): {'FAIL TO REJECT H0 -> No significant ARCH effects detected.' if not arch_present else 'REJECT H0 -> **ARCH effects likely present**.'}"
        if arch_present: interp += " The volatility of the series appears time-varying. Consider using a GARCH model."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "ARCH-LM test failed.", "interpretation": f"Error: {e}"}

def run_ljung_box_std(model_id, results, test_params={'lags': 10}, **kwargs):
    try:
        if model_id != 'garch':
            raise ValueError("This test is intended for GARCH model standardized residuals.")
        lags_input = test_params.get('lags', 10)
        lags_list = [lags_input] if isinstance(lags_input, int) else lags_list 
        if not lags_list or lags_list[-1] < 1: raise ValueError("Lags must be positive.")
        std_resid = getattr(results, 'std_resid', None) 
        if std_resid is None:
            raise ValueError("Could not access or calculate standardized residuals.")
        std_resid = std_resid.dropna()
        if std_resid.empty: raise ValueError("Standardized residuals are empty.")
        max_allowed_lags = len(std_resid) - 1
        if lags_list[-1] >= max_allowed_lags:
            lags_list = [max(1, max_allowed_lags - 1)]
            print(f"Warning: Reduced Ljung-Box (Std Resid) lags to {lags_list[0]} due to sample size.")
        if len(std_resid) <= lags_list[-1]: raise ValueError(f"Not enough standardized residuals ({len(std_resid)}) for test.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lb_results_df = acorr_ljungbox(std_resid**2, lags=lags_list, return_df=True, boxpierce=False)
        if lb_results_df.empty: raise ValueError("Could not calculate Ljung-Box statistic.")
        lb_pval = lb_results_df['lb_pvalue'].iloc[-1]
        lb_stat = lb_results_df['lb_stat'].iloc[-1]
        output = f"Ljung-Box Test (Squared GARCH Standardized Residuals)\nNull: No autocorrelation up to lag {lags_list[-1]}\n{'-'*60}\n"
        output += lb_results_df.round(4).to_string()
        no_autocorr = lb_pval > 0.05
        interp = f"Conclusion (p={lb_pval:.4f} at lag {lags_list[-1]}): {'FAIL TO REJECT H0 -> GARCH model appears to capture volatility dynamics (no remaining ARCH).' if no_autocorr else 'REJECT H0 -> **Autocorrelation detected in squared standardized residuals (remaining ARCH effects).**'}"
        if not no_autocorr: interp += " Consider adjusting GARCH orders (p,q) or trying different GARCH variants (e.g., EGARCH, GJR-GARCH)."
        return {"formatted_results": output, "interpretation": interp}
    except Exception as e:
        return {"formatted_results": "Ljung-Box (GARCH Std Resid^2) test failed.", "interpretation": f"Error: {e}"}


# --- Panel Specific Post-Tests ---

def run_panel_serial_corr(model_id, results, test_params={}, **kwargs):
    """
    Performs the Wooldridge test for serial correlation in panel data models.
    """
    if model_id != 'panel' or not PANEL_TESTS_AVAILABLE:
        raise ValueError("Panel serial correlation test requires a fitted Panel model and 'linearmodels' library.")
    
    if panel_serial_correlation is None:
         return {"formatted_results": "Not Available", "interpretation": "The 'panel_serial_correlation' function is not available in this version of linearmodels (v6+). Please check residuals manually or use Durbin-Watson on Pooled OLS as a proxy."}

    try:
        # This function expects the 'linearmodels' results object directly
        test_result = panel_serial_correlation(results)
        
        stat = test_result.stat
        p_val = test_result.pval
        
        output = f"Wooldridge Test for Serial Correlation in Panel Data\n"
        output += f"Null Hypothesis: No first-order serial correlation\n"
        output += "----------------------------------------\n"
        output += f"Statistic:     {stat:.4f}\n"
        output += f"P-Value:        {p_val:.4f}\n"
        
        no_corr = p_val > 0.05
        interpretation = (
            f"Conclusion (p={p_val:.4f}): {'FAIL TO REJECT H0' if no_corr else 'REJECT H0'}. "
            f"{'No significant serial correlation detected.' if no_corr else 'Significant **serial correlation IS present**.'}"
        )
        
        return {"formatted_results": output, "interpretation": interpretation}
        
    except Exception as e:
        print(f"Error during Panel Serial Correlation test: {e}")
        return {"formatted_results": f"Test failed: {e}", "interpretation": f"Error: {e}"}

def run_panel_hetero(model_id, results, test_params={}, **kwargs):
    """
    Performs the Breusch-Pagan test for heteroskedasticity in panel data models.
    Uses statsmodels 'het_breuschpagan' on the panel residuals.
    """
    if model_id != 'panel' or not PANEL_TESTS_AVAILABLE:
        raise ValueError("Panel heteroskedasticity test requires a fitted Panel model and 'linearmodels' library.")
        
    try:
        # 1. Get Residuals (Pandas Series)
        if not hasattr(results, 'resids'):
            raise ValueError("Model results object missing 'resids'.")
        residuals = results.resids
        
        # 2. Get Exogenous Variables (DataFrame)
        # (FIXED: Handle both PanelData object and DataFrame)
        exog_obj = results.model.exog
        exog_data = exog_obj.dataframe if hasattr(exog_obj, 'dataframe') else exog_obj

        # 3. Run Test using statsmodels
        # het_breuschpagan(resid, exog_het)
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, exog_data)
        
        output = f"Panel Breusch-Pagan Test for Heteroskedasticity\n"
        output += f"Null Hypothesis: Homoskedasticity (constant variance)\n"
        output += "----------------------------------------\n"
        output += f"LM Statistic:   {bp_stat:.4f}\n"
        output += f"P-Value:        {bp_pvalue:.4f}\n"
        
        is_homo = bp_pvalue > 0.05
        interpretation = (
            f"Conclusion (p={bp_pvalue:.4f}): {'FAIL TO REJECT H0' if is_homo else 'REJECT H0'}. "
            f"{'No significant heteroskedasticity detected.' if is_homo else 'Significant **heteroskedasticity IS present**.'}"
        )
        if not is_homo:
                interpretation += "\n(Note: Your model was fitted with 'cov_type=robust' which already corrects standard errors for this.)"

        return {"formatted_results": output, "interpretation": interpretation}
        
    except Exception as e:
        print(f"Error during Panel Heteroskedasticity test: {e}")
        traceback.print_exc()
        return {"formatted_results": f"Test failed: {e}", "interpretation": f"Error: {e}"}


# --- Logit / Probit / ML Tests ---
def run_classification_report(model_id, results, test_params={'threshold': 0.5}, **kwargs):
    try:
        if model_id not in ['logit', 'probit']:
            raise ValueError("Classification Report only applicable to Logit/Probit models.")
        y_true = results.model.endog
        threshold = test_params.get('threshold', 0.5)
        pred_prob = results.predict()
        y_true_aligned = y_true.loc[pred_prob.index]
        pred_class = (pred_prob >= threshold).astype(int)
        true_labels = np.unique(y_true_aligned)
        pred_labels = np.unique(pred_class)
        all_labels = np.union1d(true_labels, pred_labels)
        if set(all_labels).issubset({0, 1}): all_labels = [0, 1]
        elif 0 in all_labels: all_labels = [0] 
        elif 1 in all_labels: all_labels = [1]
        else: all_labels = all_labels 
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_true_aligned, pred_class, zero_division=0, labels=all_labels, output_dict=False)
        matrix = confusion_matrix(y_true_aligned, pred_class, labels=all_labels)
        output = f"Classification Report ({model_id.capitalize()}, Threshold={threshold})\n"
        output += "----------------------------------------\n"
        output += report
        output += "\nConfusion Matrix:\n"
        if len(all_labels) == 1:
            label = all_labels[0]
            output += f"              Predicted {label}\n"
            output += f"Actual {label}:    {matrix[0,0]}\n"
            accuracy = 1.0 
        elif len(all_labels) == 2:
            output += f"              Predicted {all_labels[0]} | Predicted {all_labels[1]}\n"
            output += f"Actual {all_labels[0]}:    {matrix[0,0]:<8} | {matrix[0,1]:<8}\n"
            output += f"Actual {all_labels[1]}:    {matrix[1,0]:<8} | {matrix[1,1]:<8}\n"
            accuracy = np.trace(matrix) / np.sum(matrix) if np.sum(matrix) > 0 else 0
        else:
            output += "Matrix format error (unexpected labels)."
            accuracy = np.nan
        interpretation = f"Evaluates the model's predictive accuracy on the training data.\nOverall Accuracy: {accuracy:.2%}."
        return {"formatted_results": output, "interpretation": interpretation}
    except Exception as e:
        print(f"Error during Classification Report: {e}")
        traceback.print_exc()
        return {"formatted_results": "Classification report failed.", "interpretation": f"Error: {e}"}

def run_hosmer_lemeshow(model_id, results, test_params={'groups': 10}, **kwargs):
    if model_id not in ['logit', 'probit']:
        raise ValueError("Hosmer-Lemeshow test is only for binary choice models (Logit/Probit).")
    
    try:
        y_true = results.model.endog
        y_prob = results.predict()
        
        data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
        
        groups = test_params.get('groups', 10)
        try:
            data['group'] = pd.qcut(y_prob, q=groups, duplicates='drop')
        except Exception:
                groups = max(2, min(groups, len(data['y_prob'].unique()) - 1))
                if groups < 2: raise ValueError("Not enough unique probabilities to form groups.")
                data['group'] = pd.qcut(y_prob, q=groups, duplicates='drop')
                groups = data['group'].nunique() 

        
        hl_table = data.groupby('group', observed=True).agg(
            total_obs=pd.NamedAgg(column='y_true', aggfunc='count'),
            observed_1=pd.NamedAgg(column='y_true', aggfunc='sum'),
            expected_1=pd.NamedAgg(column='y_prob', aggfunc='sum')
        )
        
        hl_table['observed_0'] = hl_table['total_obs'] - hl_table['observed_1']
        hl_table['expected_0'] = hl_table['total_obs'] - hl_table['expected_1']
        
        # Add small constant to avoid division by zero if expected count is 0
        epsilon = 1e-9
        hl_stat = (
            ((hl_table['observed_0'] - hl_table['expected_0'])**2) / (hl_table['expected_0'] + epsilon) +
            ((hl_table['observed_1'] - hl_table['expected_1'])**2) / (hl_table['expected_1'] + epsilon)
        ).sum()
        
        df = groups - 2 
        p_value = float(chi2.sf(hl_stat, df))
        
        output = f"Hosmer-Lemeshow Goodness-of-Fit Test\n"
        output += f"Null Hypothesis: Model provides a good fit (Observed = Expected)\n"
        output += "----------------------------------------\n"
        output += f"Chi-Squared Statistic: {hl_stat:.4f}\n"
        output += f"P-Value:               {p_value:.4f}\n"
        output += f"Degrees of Freedom (g-2): {df}\n"
        
        good_fit = p_value > 0.05
        interpretation = (
            f"Conclusion (p={p_value:.4f}): {'FAIL TO REJECT H0' if good_fit else 'REJECT H0'}. "
            f"{'The model appears to fit the data well.' if good_fit else 'The model **does NOT** appear to fit the data well.'}"
        )
        
        return {"formatted_results": output, "interpretation": interpretation}
        
    except Exception as e:
        print(f"Error during Hosmer-Lemeshow test: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error generating Hosmer-Lemeshow test: {e}") from e

def run_cusum_plot(model_id, results, test_params={'of_squares': False}, **kwargs):
    if model_id not in ['ols', 'ardl']:
        raise ValueError("CUSUM tests are only applicable to OLS/ARDL models.")
    
    if not hasattr(results, 'model') or not hasattr(results, 'resid'):
        raise ValueError("Cached model results object is invalid or missing residuals.")

    try:
        results_to_use = results 
        
        if model_id == 'ardl':
            original_df = kwargs.get('original_df')
            original_params = kwargs.get('original_params')
            if original_df is None or original_params is None:
                raise ValueError("Could not retrieve original data/params from cache for CUSUM test.")

            from .utils import ARDL 
            # Note: Ensure ARDL class is available or imported correctly in your main script
            # If using statsmodels ARDL directly, adjust import. Assuming local utils based on previous context.
            if 'ARDL' not in globals(): # Fallback check
                 from statsmodels.tsa.ardl import ARDL
                 
            endog_var = original_params.get('endog_var')
            exog_vars = original_params.get('exog_vars')
            lags = original_params.get('lags', 1)
            
            model_data = original_df[[endog_var] + exog_vars]
            endog = model_data[endog_var]
            exog = model_data[exog_vars]

            # Re-estimate as OLS for CUSUM
            ardl_model_dummy = ARDL(endog, lags=lags, exog=exog, order=lags, trend='c')
            
            Y_ols_dummy = ardl_model_dummy.endog 
            X_ols_dummy = ardl_model_dummy.exog 
            
            ols_model_for_cusum = sm.OLS(Y_ols_dummy, X_ols_dummy)
            results_to_use = ols_model_for_cusum.fit()
        
        Y_np = np.asarray(results_to_use.model.endog)
        X_np = np.asarray(results_to_use.model.exog)
        
        k = X_np.shape[1] 
        n = Y_np.shape[0] 

        if n <= k:
            raise ValueError(f"Not enough observations (N={n}) to skip (K={k}) parameters.")
        
        temp_model = sm.OLS(Y_np, X_np)
        temp_results_for_cusum = temp_model.fit() 
        
        # --- FIX: Dynamic Skip Loop ---
        # نحاول العثور على أول قيمة لـ skip تجعل المصفوفة قابلة للحل
        # نبدأ من k (عدد المتغيرات) ونزيد تدريجياً
        rec_res = None
        used_skip = k
        
        # الحد الأقصى للمحاولة هو نصف البيانات (حتى لا نأكل كل العينة)
        max_skip_attempt = int(n * 0.75) 
        
        for attempt_skip in range(k + 1, max_skip_attempt):
            try:
                # محاولة الحساب مع زيادة الـ skip
                rec_res = recursive_olsresiduals(temp_results_for_cusum, skip=attempt_skip)
                used_skip = attempt_skip
                # إذا نجح، نخرج من اللوب
                break 
            except Exception:
                # إذا فشل (singular matrix)، نكمل للرقم التالي
                continue
                
        if rec_res is None:
            raise ValueError("Could not find a valid initialization period (skip) where the regressor matrix is non-singular. This usually happens when Dummy variables are all zeros for a large initial period.")

        of_squares = test_params.get('of_squares', False)
        
        if of_squares:
            test_name = "CUSUM of Squares"
            cusum_sq_stat = (rec_res[3] / rec_res[3][-1])
            time_index = np.arange(used_skip, n) 
            
            alpha = 0.05
            a = (alpha / 2)
            upper_bound = a + (time_index - used_skip) * (1 - alpha) / (n - used_skip)
            lower_bound = -a + (time_index - used_skip) * (1 - alpha) / (n - used_skip)

            plot_data = {
                "x_axis": time_index.tolist(),
                "cusum_stat": cusum_sq_stat.tolist(),
                "upper_band": upper_bound.tolist(),
                "lower_band": lower_bound.tolist(),
            }
            interp = f"CUSUM of Squares test generated (Initialized at obs {used_skip})."
            
        else:
            test_name = "CUSUM"
            rec_resid = rec_res[0]
            
            if temp_results_for_cusum.df_resid <= 0 or temp_results_for_cusum.ssr is None:
                raise ValueError("Cannot compute residual standard deviation for CUSUM test.")
            std_dev = np.sqrt(temp_results_for_cusum.ssr / temp_results_for_cusum.df_resid)
            
            if std_dev == 0:
                raise ValueError("Residual standard deviation is zero, cannot perform CUSUM test.")
            
            cusum_stat = np.cumsum(rec_resid) / (std_dev * np.sqrt(n - used_skip))
            
            time_index = np.arange(used_skip, n)
            
            alpha = 0.05
            crit_val = 0.948 
            upper_bound = crit_val * np.sqrt(n - used_skip) + 2 * crit_val * (time_index - used_skip) / np.sqrt(n - used_skip)
            lower_bound = -upper_bound
            
            plot_data = {
                "x_axis": time_index.tolist(),
                "cusum_stat": cusum_stat.tolist(),
                "upper_band": upper_bound.tolist(),
                "lower_band": lower_bound.tolist(),
            }
            interp = f"CUSUM test generated (Initialized at obs {used_skip}). If the blue line crosses the red bands, parameters are unstable."

        for key in ['cusum_stat', 'upper_band', 'lower_band']:
            plot_data[key] = [None if pd.isna(v) else v for v in plot_data[key]]

        return {"plot_data_cusum": plot_data, "test_name": test_name, "interpretation": interp}

    except Exception as e:
        print(f"Error during CUSUM calculation: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error generating CUSUM test data: {e}") from e


def run_chow_test(model_id, results, test_params={'breakpoint': None}, **kwargs):
    try:
        if model_id not in ['ols', 'ardl', 'lasso', 'elastic_net', 'ridge']:
            raise ValueError("Chow test is only applicable to OLS/ARDL/Lasso/Ridge models.")
        
        breakpoint_row_str = test_params.get('breakpoint')
        if breakpoint_row_str is None or str(breakpoint_row_str).strip() == '':
            raise ValueError("Breakpoint (row number) was not provided.")
        breakpoint_row = int(breakpoint_row_str)
        
        original_df = kwargs.get('original_df')
        original_params = kwargs.get('original_params')
        if original_df is None or original_params is None:
                raise ValueError("Could not retrieve original data/params from cache for Chow test.")

        Y = None
        X = None

        if model_id == 'ardl':
            Y_var = original_params.get('endog_var')
            X_vars = original_params.get('exog_vars')
            if not Y_var or not X_vars:
                raise ValueError("Could not determine Y and X variables from cached ARDL parameters.")
            
            from .utils import ARDL 
            if ARDL is None: raise ImportError("ARDL model not found.")
            lags = original_params.get('lags', 1)
            
            # (!!!) CORRECTED: Use Y_var here, NOT endog_var
            model_data = original_df[[Y_var] + X_vars].dropna()
            endog = model_data[Y_var]
            exog = model_data[X_vars]

            ardl_model_dummy = ARDL(endog, lags=lags, exog=exog, order=lags, trend='c')
            
            Y = ardl_model_dummy.endog 
            X = ardl_model_dummy.exog 

        elif model_id in ['ols', 'lasso', 'elastic_net', 'ridge']:
            Y_var = original_params.get('dependent_var')
            X_vars = original_params.get('independent_vars')
            if not Y_var or not X_vars:
                raise ValueError("Could not determine Y and X variables from cached OLS/Lasso parameters.")

            model_data = original_df[[Y_var] + X_vars].dropna()
            Y = model_data[Y_var]
            X = model_data[X_vars]
            X = sm.add_constant(X, has_constant='raise')

        if Y is None or X is None:
                raise ValueError("Failed to construct Y and X for Chow test.")

        n_obs = len(Y)
        k = X.shape[1] 

        if breakpoint_row <= k or breakpoint_row >= (n_obs - k):
            raise ValueError(f"Invalid breakpoint ({breakpoint_row}). Must be between {k} and {n_obs - k - 1} to have enough data in both sub-samples for {k} parameters.")

        model_full = sm.OLS(Y, X).fit()
        ssr_full = model_full.ssr 

        Y_pre = Y[:breakpoint_row]; X_pre = X[:breakpoint_row]
        model_1 = sm.OLS(Y_pre, X_pre).fit()
        ssr_1 = model_1.ssr
        n1 = len(Y_pre)
        
        Y_post = Y[breakpoint_row:]; X_post = X[breakpoint_row:]
        model_2 = sm.OLS(Y_post, X_post).fit()
        ssr_2 = model_2.ssr
        n2 = len(Y_post)

        ssr_sum_sub = ssr_1 + ssr_2
        if ssr_sum_sub == 0:
            f_statistic = 0.0
            p_value = 1.0
        else:
            numerator = (ssr_full - ssr_sum_sub) / k
            denominator = ssr_sum_sub / (n1 + n2 - (2 * k))
            if denominator <= 0:
                raise RuntimeError(f"Calculation error: Denominator is zero or negative.")
            f_statistic = numerator / denominator
            p_value = f.sf(f_statistic, k, (n1 + n2 - (2 * k)))

        output = f"Chow Test (Structural Break)\n"
        output += f"H0: No structural break at row {breakpoint_row}\n"
        output += "----------------------------------------\n"
        output += f"F-Statistic:            {f_statistic:.4f}\n"
        output += f"P-Value:                {p_value:.4f}\n"
        is_significant = p_value <= 0.05
        interpretation = (
            f"Conclusion (p={p_value:.4f}): "
            f"{'REJECT H0' if is_significant else 'FAIL TO REJECT H0'}. "
            f"{'Significant structural break detected.' if is_significant else 'No structural break detected.'}"
        )
        return {"formatted_results": output, "interpretation": interpretation}
        
    except Exception as e:
        print(f"Error during Chow test: {e}")
        return {"interpretation": f"Error generating Chow test: {e}"}



def run_ml_residuals_plot(model_id, results, test_params={}, **kwargs):
    try:
        model_to_predict = None
        X_train, X_test, Y_train, Y_test = None, None, None, None
        
        if isinstance(results, dict) and "model" in results and "X_train" in results:
            model_to_predict = results["model"]
            X_train = results["X_train"]
            Y_train = results["Y_train"]
            X_test = results["X_test"]
            Y_test = results["Y_test"]
        
        elif hasattr(results, 'predict') and hasattr(results, 'named_steps'):
            model_to_predict = results 
            
            original_df = kwargs.get('original_df')
            original_params = kwargs.get('original_params')
            if original_df is None or original_params is None:
                raise ValueError("Could not retrieve original data/params from cache for Pipeline residuals.")
                
            dependent_var = original_params.get('dependent_var')
            independent_vars = original_params.get('independent_vars')
            
            required_cols = [dependent_var] + independent_vars
            model_data = original_df[required_cols].dropna()
            Y = model_data[dependent_var]
            X = model_data[independent_vars]
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
            
            if X_train.empty or X_test.empty:
                raise ValueError("Not enough data to split for residual calculation.")
        
        else:
                raise ValueError("Cached model object is not a valid ML model bundle or Pipeline.")

        Y_pred_train = model_to_predict.predict(X_train)
        residuals_train = (Y_train - Y_pred_train).values
        
        Y_pred_test = model_to_predict.predict(X_test)
        residuals_test = (Y_test - Y_pred_test).values
        
        train_data = [{"x": float(pred), "y": float(res)} for pred, res in zip(Y_pred_train, residuals_train)]
        test_data = [{"x": float(pred), "y": float(res)} for pred, res in zip(Y_pred_test, residuals_test)]
        
        plot_data = {
            "train": train_data,
            "test": test_data
        }
        
        interp = "Residuals vs. Fitted Plot generated. Ideally, points should be randomly scattered around the zero line with no clear pattern (e.g., no 'U' shape or 'funnel' shape)."

        return {"plot_data_residuals": plot_data, "interpretation": interp}

    except Exception as e:
        print(f"Error during ML Residuals Plot generation: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Error generating ML Residuals Plot data: {e}") from e

def run_dml_sensitivity(model_id, results, test_params={}, **kwargs):
    """
    Performs Sensitivity Analysis for DML using 'Robustness Value'.
    """
    if model_id != 'double_ml':
        return {"interpretation": "Only for Double ML models."}
    
    # (!!!) Check for valid bundle (!!!)
    if not isinstance(results, dict) or "final_ols" not in results:
         return {"interpretation": "⚠️ Error: Invalid model cache. Please click 'Run Analysis' again to enable this test."}

    try:
        final_ols = results.get("final_ols")
        
        t_stat = final_ols.tvalues[1]
        df_resid = final_ols.df_resid
        
        # Partial R2
        partial_r2 = (t_stat**2) / (t_stat**2 + df_resid)
        
        # Robustness Value
        rv_percent = np.sqrt(partial_r2) * 100
        
        output = f"Sensitivity Analysis (Robustness Value)\n{'-'*50}\n"
        output += f"T-Statistic:    {t_stat:.4f}\n"
        output += f"Partial R2:     {partial_r2:.4f}\n"
        output += f"Robustness Val: {rv_percent:.2f}%\n"
        
        interp = f"Robustness Value = {rv_percent:.2f}%.\n\n"
        interp += "Interpretation:\n"
        interp += f"To invalidate this result, an unobserved confounder would need to explain at least **{rv_percent:.1f}%** of the remaining variance.\n"
        
        if rv_percent > 10:
            interp += "✅ **Result is Robust.**"
        else:
            interp += "⚠️ **Result is Sensitive** to hidden bias."
            
        return {"formatted_results": output, "interpretation": interp}

    except Exception as e:
        return {"interpretation": f"Sensitivity analysis failed: {e}"}


def run_dml_heterogeneity(model_id, results, test_params={}, **kwargs):
    """
    Checks for Heterogeneous Treatment Effects.
    """
    if model_id != 'double_ml':
        return {"interpretation": "Only for Double ML models."}

    # (!!!) Check for valid bundle (!!!)
    if not isinstance(results, dict) or "residuals_y" not in results:
         return {"interpretation": "⚠️ Error: Invalid model cache. Please click 'Run Analysis' again to enable this test."}

    try:
        res_y = results.get("residuals_y")
        res_d = results.get("residuals_d")
        original_X = results.get("original_X")
        
        target_col = test_params.get('interaction_var')
        if not target_col and original_X is not None and not original_X.empty:
            target_col = original_X.columns[0]
            
        if target_col not in original_X.columns:
             return {"interpretation": f"Variable '{target_col}' not found in controls."}
             
        Z = original_X[target_col].values
        
        # Interaction term: Y_res ~ D_res + D_res*Z
        interaction = res_d * Z
        X_new = np.column_stack((res_d, interaction))
        X_new = sm.add_constant(X_new)
        
        het_model = sm.OLS(res_y, X_new).fit()
        
        beta_interaction = het_model.params[2]
        p_val_interaction = het_model.pvalues[2]
        
        output = f"Heterogeneity Test (Interaction with '{target_col}')\n{'-'*50}\n"
        output += f"Interaction Coef: {beta_interaction:.4f}\n"
        output += f"P-Value:          {p_val_interaction:.4f}\n"
        
        interp = ""
        if p_val_interaction <= 0.05:
            interp += f"⚠️ **Heterogeneity Detected:** Effect varies significantly with '{target_col}'."
        else:
            interp += f"✅ **Homogeneous Effect:** Effect is stable across '{target_col}'."
            
        return {"formatted_results": output, "interpretation": interp}

    except Exception as e:
        return {"interpretation": f"Heterogeneity check failed: {e}"}
    
# --- (!!!) Updated Forecast Function with Metrics (!!!) ---
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
    
# 2. دالة التنبؤ والتحليل (run_forecast)
# ==========================================
def run_forecast(model_id, results, test_params={'periods': 10, 'alpha': 0.05}, **kwargs):
    
    supported_models = ['arima', 'var', 'vecm', 'ols', 'ardl', 'lasso', 'ridge', 'elastic_net', 'random_forest', 'xgboost', 'garch']
    
    if model_id not in supported_models:
        raise ValueError(f"Forecast generation is not supported for model type: '{model_id}'.")
        
    periods = int(test_params.get('periods', 10))
    alpha = float(test_params.get('alpha', 0.05))
    
    try:
        # =======================================================
        # 1. استخراج وتجهيز البيانات التاريخية (تم التصحيح هنا)
        # =======================================================
        hist_data_values = []
        
        # الحالة 1: نماذج الـ ML (RandomForest, XGBoost...) المخزنة في Dictionary
        if isinstance(results, dict):
            # محاولة استخراج y_train و y_test (مع مراعاة الحروف الكبيرة والصغيرة)
            y_t = results.get("y_train") if "y_train" in results else results.get("Y_train")
            y_s = results.get("y_test") if "y_test" in results else results.get("Y_test")
            
            if y_t is not None and y_s is not None:
                # تحويل كل شيء إلى مصفوفة numpy مسطحة (1D Array) لتجنب مشاكل الـ Pandas Index
                arr_train = np.array(y_t).flatten()
                arr_test = np.array(y_s).flatten()
                # دمجهم معاً
                full_hist = np.concatenate([arr_train, arr_test])
                hist_data_values = [float(x) for x in full_hist if pd.notna(x)]
        
        # الحالة 2: نماذج Statsmodels (ARIMA, OLS...)
        elif hasattr(results, 'model') and hasattr(results.model, 'endog'):
             arr = np.array(results.model.endog).flatten()
             hist_data_values = [float(x) for x in arr if pd.notna(x)]
             
        # الحالة 3: بدائل أخرى لـ Statsmodels
        elif hasattr(results, 'model') and hasattr(results.model, 'data') and hasattr(results.model.data, 'endog'):
             arr = np.array(results.model.data.endog).flatten()
             hist_data_values = [float(x) for x in arr if pd.notna(x)]

        # تأكيد: إذا فشل كل شيء، نرسل قائمة فارغة (لكن الكود أعلاه يغطي 99% من الحالات)
        if not hist_data_values:
            print("Warning: Could not extract historical data.")

        # =======================================================
        # 2. التنبؤ المستقبلي (Forecast Generation)
        # =======================================================
        original_params = kwargs.get('original_params', {})
        plot_data = {}
        interp = ""
        metrics_html = ""

        try:
            # === A. Advanced ML Models (RECURSIVE FORECASTING) ===
            if model_id in ['random_forest', 'xgboost', 'lasso', 'ridge', 'elastic_net']:
                pipeline = results.get("model")
                
                # استخراج آخر لاجات (يجب أن تكون موجودة الآن)
                raw_last_lags = results.get("last_lags")
                if raw_last_lags is None:
                     # Fallback في حالة البيانات القديمة
                     if "X_test" in results:
                         raw_last_lags = results["X_test"].iloc[-1:].values
                     else:
                         raise ValueError("Model cache missing 'last_lags'. Please Retrain.")

                current_input = np.array(raw_last_lags).flatten().copy()
                predictions = []
                
                for _ in range(periods):
                    pred = pipeline.predict(current_input.reshape(1, -1))[0]
                    predictions.append(pred)
                    current_input = np.roll(current_input, 1)
                    current_input[0] = pred 
                
                fc_data_clean = [float(x) for x in predictions]
                fc_lower_clean = [None] * periods
                fc_upper_clean = [None] * periods

                # ضبط الـ Labels للرسم
                hist_len = len(hist_data_values)
                hist_labels = list(range(1, hist_len + 1))
                fc_labels = list(range(hist_len + 1, hist_len + 1 + periods))

                plot_data = {
                    "univariate": True, 
                    "variable_name": original_params.get('dependent_var') or 'Y',
                    "hist_labels": hist_labels,
                    "hist_data": hist_data_values, # <--- البيانات التاريخية هنا     
                    "fc_labels": fc_labels,
                    "fc_data": fc_data_clean,
                    "fc_lower": fc_lower_clean,
                    "fc_upper": fc_upper_clean
                }
                interp = f"Generated {periods}-step recursive forecast using {model_id}."

                # حساب الدقة (In-Sample Metrics) للـ ML
                # نستخدم predict على كل البيانات التاريخية لحساب الخطأ
                if "X_train" in results and "X_test" in results:
                    X_full = pd.concat([results["X_train"], results["X_test"]])
                    y_full_pred = pipeline.predict(X_full)
                    # y_full_true هو نفسه hist_data_values الذي حسبناه في البداية
                    # ولكن للتأكد من توافق الأطوال (بسبب اللاجات، أول كام صف بيطيروا)
                    # سنقارن فقط الأطوال المتوفرة
                    
                    y_full_true = np.concatenate([results["y_train"], results["y_test"]])
                    
                    # Trim to match lengths (Models with lags lose initial rows)
                    min_len = min(len(y_full_true), len(y_full_pred))
                    y_true_m = y_full_true[-min_len:]
                    y_pred_m = y_full_pred[-min_len:]
                    
                    rmse = np.sqrt(mean_squared_error(y_true_m, y_pred_m))
                    mae = mean_absolute_error(y_true_m, y_pred_m)
                    mape = np.mean(np.abs((y_true_m - y_pred_m) / y_true_m)) * 100
                    
                    metrics_html = f"""
                     <table border="1" style="width:100%; border-collapse: collapse; margin-top:10px;">
                        <tr style="background:#f8f9fa;">
                            <th style="padding:8px;">Metric</th>
                            <th style="padding:8px;">Value</th>
                        </tr>
                        <tr><td style="padding:8px;">RMSE</td><td style="padding:8px;">{rmse:.4f}</td></tr>
                        <tr><td style="padding:8px;">MAE</td><td style="padding:8px;">{mae:.4f}</td></tr>
                        <tr><td style="padding:8px;">MAPE</td><td style="padding:8px;">{mape:.2f}%</td></tr>
                     </table>
                     """
        except Exception as ml_e:
            # Ensure this try block has an except so the function compiles,
            # and set safe defaults so downstream forecasting logic can continue.
            print(f"Error during ML recursive forecast: {ml_e}")
            traceback.print_exc()
            plot_data = {}
            interp = f"ML recursive forecasting failed: {ml_e}"
            metrics_html = ""
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

                # (!!!) Ensure historical data array (2D) available
                hist_data_series_or_df = None
                if hasattr(results, 'model') and hasattr(results.model, 'endog'):
                    hist_data_series_or_df = results.model.endog
                elif hasattr(results, 'model') and hasattr(results.model, 'data') and hasattr(results.model.data, 'endog'):
                    hist_data_series_or_df = results.model.data.endog
                elif hist_data_values:
                    # Fallback to flattened history (univariate series)
                    hist_data_series_or_df = np.array(hist_data_values)
                else:
                    raise ValueError("Historical data unavailable for VAR/VECM forecast.")

                hist_arr = np.array(hist_data_series_or_df)
                
                fc_data_array = None
                fc_lower_array, fc_upper_array = None, None
                
                if model_id == 'var':
                    # VAR needs the last k lags to forecast
                    # Ensure results has k_ar and hist_arr has enough rows
                    if not hasattr(results, 'k_ar'):
                        raise ValueError("VAR results missing 'k_ar' attribute required for forecasting.")
                    k_ar = int(results.k_ar)
                    if hist_arr.ndim == 1:
                        # If 1D, treat as single variable series
                        if len(hist_arr) < k_ar:
                            raise ValueError("Not enough historical observations for VAR forecasting.")
                        y_last_lags = hist_arr[-k_ar:]
                    else:
                        if hist_arr.shape[0] < k_ar:
                            raise ValueError("Not enough historical observations for VAR forecasting.")
                        y_last_lags = hist_arr[-k_ar:, :]
                    fc_data_array = results.forecast(y=y_last_lags, steps=periods)
                    try:
                        if hasattr(results, 'forecast_interval'):
                            fc_intervals = results.forecast_interval(y=y_last_lags, steps=periods, alpha=alpha)
                            fc_lower_array, fc_upper_array = fc_intervals[0], fc_intervals[1]
                    except Exception:
                        pass
                else: # VECM
                    # VECM prediction interface may differ; try .predict
                    fc_data_array = results.predict(steps=periods) 

                # Prepare plotting data (multi-series)
                hist_len = hist_arr.shape[0] if hasattr(hist_arr, 'shape') and len(hist_arr.shape) > 0 else len(hist_arr)
                hist_labels = list(range(1, hist_len + 1))
                fc_labels = list(range(hist_len + 1, hist_len + 1 + periods))
                
                # If hist_arr is 1D, convert to 2D for consistent indexing
                if hist_arr.ndim == 1:
                    hist_arr_2d = hist_arr.reshape(-1, 1)
                else:
                    hist_arr_2d = hist_arr

                # Ensure var_names length matches number of columns; fallback to generic names
                if not var_names or len(var_names) != hist_arr_2d.shape[1]:
                    var_names = [f"Var{i+1}" for i in range(hist_arr_2d.shape[1])]

                hist_data = {col: [round(float(v), 4) for v in hist_arr_2d[:, i]] for i, col in enumerate(var_names)}
                fc_data = {col: [round(float(v), 4) for v in fc_data_array[:, i]] for i, col in enumerate(var_names)}
                
                fc_lower, fc_upper = {}, {}
                for i, col in enumerate(var_names):
                    fc_lower[col] = [round(float(v), 4) for v in fc_lower_array[:, i]] if (fc_lower_array is not None and np.array(fc_lower_array).size > 0) else [None]*periods
                    fc_upper[col] = [round(float(v), 4) for v in fc_upper_array[:, i]] if (fc_upper_array is not None and np.array(fc_upper_array).size > 0) else [None]*periods
                
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
# ==========================================
# 3. كود اختبار سريع (Test Script)
# ==========================================
if __name__ == "__main__":
    # إنشاء بيانات وهمية للتجربة
    print("Generating dummy data...")
    df = pd.DataFrame({
        'Y': np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100),
        'X1': np.linspace(0, 10, 100),
        'X2': np.random.random(100)
    })
    
    # 1. تدريب النموذج
    print("1. Training Model...")
    results_bundle = train_advanced_ml_model(
        df, 
        target_col='Y', 
        feature_cols=['X1', 'X2'], 
        model_type='random_forest', 
        lags=2, 
        tune=False # False للسرعة في التجربة
    )
    
    # 2. تشغيل التنبؤ
    print("2. Running Forecast...")
    forecast_output = run_forecast(
        model_id='random_forest', 
        results=results_bundle, 
        test_params={'periods': 12},
        original_params={'dependent_var': 'Y'}
    )
    
    # 3. عرض النتائج
    print("\n--- Forecast Output ---")
    if "error" in forecast_output:
        print("FAILED:", forecast_output["error"])
    else:
        print("SUCCESS!")
        print("Interpretation:", forecast_output['interpretation'])
        print("First 5 Forecast Values:", forecast_output['plot_data_forecast']['fc_data'][:5])
        print("Metrics HTML Generated:", bool(forecast_output['metrics_html']))    
