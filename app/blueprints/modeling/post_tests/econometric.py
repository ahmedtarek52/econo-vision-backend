import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset, recursive_olsresiduals
from scipy.stats import chi2, f
import traceback

# --- Panel Support ---
try:
    from linearmodels.panel.diagnostics import panel_serial_correlation
    PANEL_TESTS_AVAILABLE = True
except ImportError:
    PANEL_TESTS_AVAILABLE = False
    panel_serial_correlation = None

# --- OLS / ARDL Tests ---

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
        
        rec_res = recursive_olsresiduals(temp_results_for_cusum, skip=k)
        
        of_squares = test_params.get('of_squares', False)
        
        if of_squares:
            test_name = "CUSUM of Squares"
            cusum_sq_stat = (rec_res[3] / rec_res[3][-1])
            time_index = np.arange(k, n) 
            
            alpha = 0.05
            a = (alpha / 2)
            upper_bound = a + (time_index - k) * (1 - alpha) / (n - k)
            lower_bound = -a + (time_index - k) * (1 - alpha) / (n - k)

            plot_data = {
                "x_axis": time_index.tolist(),
                "cusum_stat": cusum_sq_stat.tolist(),
                "upper_band": upper_bound.tolist(),
                "lower_band": lower_bound.tolist(),
            }
            interp = "CUSUM of Squares test generated. If the blue line (statistic) crosses the red confidence bands, it indicates parameter instability."
            
        else:
            test_name = "CUSUM"
            rec_resid = rec_res[0]
            
            if temp_results_for_cusum.df_resid <= 0 or temp_results_for_cusum.ssr is None:
                raise ValueError("Cannot compute residual standard deviation for CUSUM test.")
            std_dev = np.sqrt(temp_results_for_cusum.ssr / temp_results_for_cusum.df_resid)
            
            if std_dev == 0:
                raise ValueError("Residual standard deviation is zero, cannot perform CUSUM test.")
            
            cusum_stat = np.cumsum(rec_resid) / (std_dev * np.sqrt(n - k))
            
            time_index = np.arange(k, n)
            
            alpha = 0.05
            crit_val = 0.948 
            upper_bound = crit_val * np.sqrt(n - k) + 2 * crit_val * (time_index - k) / np.sqrt(n - k)
            lower_bound = -upper_bound
            
            plot_data = {
                "x_axis": time_index.tolist(),
                "cusum_stat": cusum_stat.tolist(),
                "upper_band": upper_bound.tolist(),
                "lower_band": lower_bound.tolist(),
            }
            interp = "CUSUM test generated. If the blue line (statistic) crosses the red confidence bands, it indicates parameter instability."

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

# --- Panel Tests ---

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


# --- Logit / Probit Tests ---

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
