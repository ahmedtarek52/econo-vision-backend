import numpy as np
import pandas as pd
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import traceback
import io
import sys
import warnings

# --- VAR / VECM Tests ---

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

