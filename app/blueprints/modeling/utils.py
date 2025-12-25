from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from arch import arch_model
from statsmodels.tsa.api import VAR, VECM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.discrete.discrete_model import Logit, Probit
import statsmodels.api as sm
import numpy as np
import pandas as pd
import time
import warnings
import io
import sys
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.api import VAR
try:
    from statsmodels.tsa.ardl import ARDL
except ImportError:
    try:
        from statsmodels.tsa.api import ARDL
    except ImportError:
        print("Warning: ARDL not found. Please install or check statsmodels version.")
        ARDL = None
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

# (!!!) (هذا هو الإصلاح: تم تغليف linearmodels) (!!!)
try:
    from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects, compare
except ImportError:
    print("WARNING: linearmodels not found. Panel models will be disabled.")
    # (تعريف متغيرات وهمية لتجنب انهيار الدوال الأخرى)
    PooledOLS, PanelOLS, RandomEffects, compare = (None, None, None, None)

from statsmodels.discrete.discrete_model import Logit, Probit

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import time

from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey

import warnings
import traceback

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from firebase_admin import storage

def get_dataframe_from_cloud(file_path):
    """
    تحميل ملف من Firebase Storage وتحويله لـ DataFrame عند الطلب
    """
    try:
        print(f"Downloading file from cloud: {file_path}")
        bucket = storage.bucket()
        blob = bucket.blob(file_path)

        # تحميل الملف في الذاكرة (RAM)
        data_bytes = blob.download_as_string()

        # تحديد الامتداد
        if file_path.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(data_bytes))
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(data_bytes))
        elif file_path.endswith('.json'):
            df = pd.read_json(io.BytesIO(data_bytes))
        else:
            # محاولة افتراضية
            try:
                df = pd.read_csv(io.BytesIO(data_bytes))
            except:
                df = pd.read_excel(io.BytesIO(data_bytes))

        return df
    except Exception as e:
        print(f"Cloud Download Error: {e}")
        raise ValueError(f"Failed to load data from cloud: {str(e)}")

# ... (باقي الدوال كما هي) ...


# --- دالة الاختبارات التشخيصية التلقائية ---
def run_single_equation_diagnostics(model_results):
    """
    Runs common post-estimation diagnostic tests suitable for single-equation models.
    """
    diagnostics = []
    try:
        resid = model_results.resid
        exog = getattr(model_results.model, 'exog', None)
        if exog is not None and not isinstance(exog, np.ndarray):
            exog = np.asarray(exog)
        if exog is not None and exog.ndim == 1:
            exog = exog[:, np.newaxis]
    except Exception as e:
        print(f"Diagnostics Error: Could not get residuals/exog: {e}")
        diagnostics.append({"name": "Diagnostics Note", "interpretation": f"Could not retrieve residuals or exog: {e}"})
        return diagnostics

    if resid is None or not hasattr(resid, 'shape') or resid.shape[0] == 0:
        diagnostics.append({"name": "Diagnostics Note", "interpretation": "Residuals are missing or empty."})
        return diagnostics
    if not isinstance(resid, (pd.Series, np.ndarray)):
        try: resid = np.asarray(resid)
        except Exception:
            diagnostics.append({"name": "Diagnostics Note", "interpretation": "Could not convert residuals to usable format."})
            return diagnostics
    if resid.ndim > 1: resid = resid.flatten()
    has_exog = exog is not None and exog.shape[1] > 0

    # 1. Normality (Jarque-Bera)
    try:
        jb_stat, jb_pvalue, skew, kurt = (None, None, None, None)
        
        if hasattr(model_results, 'test_normality'):
            try:
                test_results_list = model_results.test_normality(method='jarquebera')
                if test_results_list:
                    jb_stat, jb_pvalue, skew, kurt = test_results_list[0]
            except Exception as sarimax_e:
                print(f"model_results.test_normality() failed: {sarimax_e}. Falling back.")
        
        if jb_stat is None:
            print("Falling back to manual JB test.")
            resid_for_jb = getattr(model_results, 'standardized_residuals', None)
            if resid_for_jb is None:
                resid_for_jb = getattr(model_results, 'std_resid', None)
            if resid_for_jb is None:
                resid_for_jb = model_results.resid
            
            resid_for_jb = resid_for_jb.dropna()
            if len(resid_for_jb) > 2:
                jb_stat_calc, jb_pvalue_calc, _, _ = jarque_bera(resid_for_jb)
                jb_stat = jb_stat_calc
                jb_pvalue = jb_pvalue_calc
            else:
                raise ValueError("Not enough data for manual JB test.")

        if jb_stat is not None and jb_pvalue is not None:
            diagnostics.append({ 
                "name": "Normality (Jarque-Bera)", 
                "statistic": round(float(jb_stat), 4), 
                "p_value": round(float(jb_pvalue), 4), 
                "interpretation": "Pass (p>0.05)" if float(jb_pvalue) > 0.05 else "Fail (p<=0.05) - Residuals may not be normal" 
            })
        else:
            diagnostics.append({"name": "Normality (Jarque-Bera)", "interpretation": "Could not be calculated."})
    except Exception as e:
        print(f"Diagnostics Error (JB): {e}")
        diagnostics.append({"name": "Normality (Jarque-Bera)", "interpretation": f"Test failed: {e}"})

    # Tests requiring exogenous variables
    if has_exog:
        # 2. Heteroskedasticity (Breusch-Pagan)
        try:
            if len(resid) > exog.shape[1]:
                exog_for_test = exog
                if not np.all(np.isclose(exog_for_test[:, 0], 1.0)):
                   exog_for_test = sm.add_constant(exog, prepend=True)
                
                if exog_for_test.shape[1] > 1:
                    non_const_vars = exog_for_test[:, 1:]
                    if np.any(np.var(non_const_vars, axis=0) < 1e-9):
                        print("Warning (BP): Near-zero variance in exogenous variable. Skipping test.")
                        raise ValueError("One or more exogenous variables have near-zero variance.")
                
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, exog_for_test)
                diagnostics.append({ "name": "Heteroskedasticity (Breusch-Pagan)", "statistic": round(bp_stat, 4), "p_value": round(bp_pvalue, 4), "interpretation": "Pass (p>0.05)" if bp_pvalue > 0.05 else "Fail (p<=0.05) - Heteroskedasticity likely present" })
            else:
                diagnostics.append({"name": "Heteroskedasticity (Breusch-Pagan)", "interpretation": "Test requires more observations than regressors."})
        except Exception as e:
            print(f"Diagnostics Error (BP): {e}")
            diagnostics.append({"name": "Heteroskedasticity (Breusch-Pagan)", "interpretation": f"Test failed: {e}"})

        # 3. Serial Correlation (Breusch-Godfrey)
        try:
            if hasattr(model_results, 'model') and has_exog:
                nlags_bg = min(10, len(resid)//5) if len(resid) > 20 else max(1, len(resid)//5)
                if nlags_bg > 0 and len(resid) > nlags_bg and len(resid) > exog.shape[1]:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        bg_stat, bg_pvalue, _, _ = acorr_breusch_godfrey(model_results, nlags=nlags_bg)
                    diagnostics.append({ "name": f"Serial Correlation (Breusch-Godfrey, {nlags_bg} lags)", "statistic": round(bg_stat, 4), "p_value": round(bg_pvalue, 4), "interpretation": "Pass (p>0.05)" if bg_pvalue > 0.05 else "Fail (p<=0.05) - Serial correlation likely present" })
                else:
                    diagnostics.append({"name": "Serial Correlation (Breusch-Godfrey)", "interpretation": "Not enough data or lags for the test."})
            else:
                raise ValueError("BG test requires model results with exogenous variables.")
        except Exception as e:
            try:
                nlags_lb = min(10, len(resid)//5) if len(resid) > 20 else max(1, len(resid)//5)
                if nlags_lb > 0 and len(resid) > nlags_lb:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb_df = acorr_ljungbox(resid, lags=[nlags_lb], return_df=True, boxpierce=False)
                    lb_pval = lb_df['lb_pvalue'].iloc[0]
                    lb_stat = lb_df['lb_stat'].iloc[0]
                    # (!!!) (إصلاح: استخدام lb_pval بدلاً من bp_pvalue) (!!!)
                    diagnostics.append({ "name": f"Serial Correlation (Ljung-Box, {nlags_lb} lags)", "statistic": round(lb_stat, 4), "p_value": round(lb_pval, 4), "interpretation": "Pass (p>0.05)" if lb_pval > 0.05 else "Fail (p<=0.05) - Serial correlation likely present" })
                else:
                    diagnostics.append({"name": "Serial Correlation", "interpretation": "Not enough data for Ljung-Box test."})
            except Exception as lb_e:
                print(f"Diagnostics Error (BG/LB): {e}, {lb_e}")
                diagnostics.append({"name": "Serial Correlation", "interpretation": f"Test failed: {e}"})

    return diagnostics

def get_dataframe_from_cloud(file_path):
    """
    تحميل ملف من Firebase Storage وتحويله لـ DataFrame
    """
    try:
        print(f"Downloading file from cloud: {file_path}")
        bucket = storage.bucket()
        blob = bucket.blob(file_path)

        # تحميل الملف في الذاكرة (RAM)
        data_bytes = blob.download_as_string()

        # تحديد الامتداد من اسم الملف
        if file_path.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(data_bytes))
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(data_bytes))
        else:
            raise ValueError("Unsupported file format in cloud storage.")

        return df
    except Exception as e:
        print(f"Cloud Download Error: {e}")
        raise ValueError(f"Failed to load data from cloud. Please re-upload the file.")
    
# --- OLS Model ---
def run_ols_model(df, dependent_var, independent_vars, cov_type='nonrobust', cov_kwds=None):
    if not dependent_var or not independent_vars: raise ValueError("Dependent and Independent variables must be selected.")
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: raise ValueError("No common valid data points after dropping NaNs.")
    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    if pd.api.types.infer_dtype(Y) not in ('floating', 'integer'): raise ValueError(f"OLS dependent variable '{dependent_var}' must be numeric.")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty: raise ValueError(f"OLS independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")
    X = sm.add_constant(X, has_constant='raise')
    try:
        model = sm.OLS(Y, X)
        results = model.fit(cov_type=cov_type, cov_kwds=cov_kwds) 
        diagnostics_results = run_single_equation_diagnostics(results)
    except Exception as e:
        raise RuntimeError(f"Error fitting OLS model: {e}") from e

    first_regressor_pvalue = None
    try:
        if results.pvalues is not None and len(results.pvalues) > 1:
            first_regressor_pvalue = results.pvalues[1] # [0] is const, [1] is first X
    except Exception:
        pass 

    metrics = {
        "R-squared": getattr(results, 'rsquared', None),
        "Adj. R-squared": getattr(results, 'rsquared_adj', None),
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0)),
        "P-Value (X1)": first_regressor_pvalue # (إضافة P-Value)
    }

    return {
        "summary_html": results.summary().as_html(),
        "diagnostics": diagnostics_results,
        "metrics": metrics,
        "fitted_model_object": results 
    }


# --- ARDL Model ---

def run_ardl_model(df, endog_var, exog_vars, lags=None, exog_lags=0, order=None, trend='c', selection_method='fixed'):
    """
    ARDL Model: Supports both 'fixed' lags and 'auto' selection (AIC).
    """
    try:
        from statsmodels.tsa.ardl import ARDL, ardl_select_order
        
        # 1. تنظيف البيانات
        df.columns = df.columns.str.strip()
        endog_var = endog_var.strip()
        if isinstance(exog_vars, str): exog_vars = [exog_vars.strip()]
        else: exog_vars = [v.strip() for v in exog_vars]

        data_clean = df[[endog_var] + exog_vars].dropna()
        for col in data_clean.columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        data_clean = data_clean.dropna()
        
        endog = data_clean[endog_var]
        exog = data_clean[exog_vars] 

        # 2. تحديد القيم
        lags_y = int(lags) if lags is not None else 1
        lags_x = int(exog_lags) if exog_lags is not None else 0
        
        print(f"DEBUG ARDL: Method={selection_method}, Y_Val={lags_y}, X_Val={lags_x}")

        # 3. تشغيل النموذج حسب الطريقة
        if selection_method == 'auto':
            # الوضع التلقائي: نستخدم القيم كحد أقصى (Max Lags)
            # ic='aic' هو المعيار الأشهر
            sel_res = ardl_select_order(endog, maxlag=lags_y, exog=exog, maxorder=lags_x, trend=trend, ic='aic')
            model = sel_res.model
            results = model.fit()
            print(f"Auto Selected Order: {sel_res.model.ardl_order}")
        else:
            # الوضع اليدوي (Fixed): نطبق القيم كما هي
            model = ARDL(endog, lags=lags_y, exog=exog, order=lags_x, trend=trend)
            results = model.fit() 

        # 4. النتائج (مشتركة)
        r_squared = getattr(results, 'rsquared', 0.0)
        
        bounds_data = None
        try:
            if hasattr(results, 'bounds_test'):
                bt = results.bounds_test()
                bounds_data = {
                    "f_stat": round(bt.stat, 4),
                    "conclusion": "Likely Cointegration" if bt.stat > bt.crit_vals[1][1] else "No Cointegration",
                    "crit_vals": {"10%": bt.crit_vals[0].tolist(), "5%": bt.crit_vals[1].tolist(), "1%": bt.crit_vals[2].tolist()}
                }
        except: pass
            
        lr_params = []
        try:
            if hasattr(results, 'params'):
                 for name, val in results.params.items():
                     if name in exog_vars or any(x in name for x in exog_vars): 
                         lr_params.append({"variable": name, "coefficient": round(val, 5)})
        except: pass

        return {
            "summary_html": results.summary().as_html(),
            "diagnostics": run_single_equation_diagnostics(results),
            "metrics": {
                "aic": getattr(results, 'aic', 0),
                "bic": getattr(results, 'bic', 0),
                "r_squared": getattr(results, 'rsquared', 0.0)
            },
            "fitted_model_object": results,
            "ardl_extra": {"bounds_test": bounds_data, "long_run": lr_params} if bounds_data else None
        }

    except Exception as e:
        import traceback
        print(f"ARDL ERROR: {e}")
        traceback.print_exc()
        raise e
    

# --- VAR Model ---
def run_var_model(df, variables, maxlags=4):
    if not variables or len(variables) < 2: raise ValueError("At least two variables required for VAR.")
    if not all(v in df.columns for v in variables): raise ValueError("Variables not found.")
    model_data = df[variables].dropna()
    if model_data.empty: raise ValueError("No valid data points after dropping NaNs.")
    non_numeric = model_data.select_dtypes(exclude=np.number).columns
    if not non_numeric.empty: raise ValueError(f"VAR requires numeric variables. Non-numeric found: {', '.join(non_numeric)}")
    if len(model_data) < maxlags + 1: raise ValueError(f"Not enough valid data rows ({len(model_data)}) for VAR with maxlags={maxlags}. Need at least {maxlags + 1}.")

    try:
        model = VAR(model_data)
        results = model.fit(maxlags=maxlags, ic='aic')
        selected_lag_order = results.k_ar
    except Exception as e:
        raise RuntimeError(f"Error fitting VAR model: {e}") from e

    summary_html = f"<pre>{str(results.summary())}</pre>"
    interpretation = f"VAR model fitted using AIC criteria up to maxlags={maxlags}. Selected lag order (k_ar): {selected_lag_order}."

    metrics = {
        "R-squared": None,
        "Adj. R-squared": None,
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": summary_html,
        "diagnostics": [{"name": "VAR Info", "interpretation": interpretation},
                        {"name": "VAR Post-Tests", "interpretation": "Run specific post-tests for Stability, Residual Normality/Autocorrelation, IRF, and FEVD."}],
        "metrics": metrics,
        "fitted_model_object": results
    }

# --- VECM Model ---
def run_vecm_model(df, variables, lags=2, coint_rank=1):
    if not variables or len(variables) < 2: raise ValueError("At least two variables required for VECM.")
    if not all(v in df.columns for v in variables): raise ValueError("Variables not found.")
    if not isinstance(coint_rank, int) or coint_rank < 0 or coint_rank >= len(variables):
            raise ValueError(f"Invalid Cointegration Rank ({coint_rank}) specified for {len(variables)} variables.")
    model_data = df[variables].dropna()
    if model_data.empty: raise ValueError("No valid data points after dropping NaNs.")
    non_numeric = model_data.select_dtypes(exclude=np.number).columns
    if not non_numeric.empty: raise ValueError(f"VECM requires numeric variables. Non-numeric found: {', '.join(non_numeric)}")
    
    k_ar_diff = lags - 1
    if k_ar_diff < 0: raise ValueError("Lags (p) must be >= 1 for VECM (k_ar_diff = p-1 >= 0).")
    
    min_obs = k_ar_diff + len(variables) + 5
    if len(model_data) < min_obs:
            raise ValueError(f"Not enough valid data rows ({len(model_data)}) for VECM with lags(p)={lags}. Need at least {min_obs}.")

    try:
        model = VECM(model_data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic='ci')
        results = model.fit()
    except Exception as e:
        raise RuntimeError(f"Error fitting VECM model: {e}. Check data stationarity (I(1)), lags, and cointegration rank.") from e

    summary_html = f"<pre>{str(results.summary())}</pre>"

    metrics = {
        "R-squared": None,
        "Adj. R-squared": None,
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": summary_html,
        "diagnostics": [{"name": "VECM Post-Tests", "interpretation": "Run specific post-tests for Stability, Residual Normality/Autocorrelation, IRF, and FEVD."}],
        "metrics": metrics,
        "fitted_model_object": results
    }
# --- FEVD Function (Forecast Error Variance Decomposition) ---

        
# --- ARIMA / SARIMA Model ---
def run_arima_sarima(df, endog_var, exog_vars=None, order=(1,0,1), seasonal_order=(0,0,0,0)):
    if not endog_var: raise ValueError("Endogenous variable required for ARIMA.")
    if endog_var not in df.columns: raise ValueError(f"Variable '{endog_var}' not found.")
    if not isinstance(order, (list, tuple)) or len(order) != 3: raise ValueError("ARIMA 'order' must be a list/tuple of 3 integers (p,d,q).")
    if not isinstance(seasonal_order, (list, tuple)) or len(seasonal_order) != 4: raise ValueError("ARIMA 'seasonal_order' must be a list/tuple of 4 integers (P,D,Q,S).")

    endog = df[endog_var].dropna()
    exog = None
    if exog_vars:
        valid_exog = [v for v in exog_vars if v in df.columns]
        if valid_exog:
            exog = df[valid_exog].copy()
            common_index = endog.index.intersection(exog.index)
            endog = endog.loc[common_index]
            exog = exog.loc[common_index].dropna()
            common_index = endog.index.intersection(exog.index)
            endog = endog.loc[common_index]
            exog = exog.loc[common_index] 
        else: print("Warning: No valid exogenous variables found.")
    if endog.empty: raise ValueError("No valid data points for the endogenous variable after handling NaNs.")
    if exog is not None and exog.empty: exog = None
    if pd.api.types.infer_dtype(endog) not in ('floating', 'integer'): raise ValueError(f"ARIMA endog variable '{endog_var}' must be numeric.")
    if exog is not None:
        non_numeric_X = exog.select_dtypes(exclude=np.number).columns
        if not non_numeric_X.empty: raise ValueError(f"ARIMA exog variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")

    try:
        # التعديل: جعل الثابت موجوداً دائماً ('c') ليطابق EViews
        # EViews default behavior includes a constant even with differencing.
        trend = 'c' 
        
        model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order,
                        trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        try:
            diagnostics_results = run_single_equation_diagnostics(results)
        except Exception as diag_e:
            diagnostics_results = [{"name":"Diagnostics Note", "interpretation": f"Error during diagnostics: {diag_e}"}]
    except Exception as e:
        raise RuntimeError(f"Error fitting ARIMA/SARIMA model: {e}. Check orders, data stationarity, seasonality (S>=1 if P,D,Q > 0).") from e

    metrics = {
        "R-squared": None,
        "Adj. R-squared": None,
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": results.summary().as_html(),
        "diagnostics": diagnostics_results,
        "metrics": metrics,
        "fitted_model_object": results
    }

# --- GARCH Model ---
def run_garch_model(df, endog_var, p=1, q=1):
    if not endog_var: raise ValueError("Endogenous variable required for GARCH.")
    if endog_var not in df.columns: raise ValueError(f"Variable '{endog_var}' not found.")
    if not isinstance(p, int) or p < 0 or not isinstance(q, int) or q < 0: raise ValueError("GARCH orders p and q must be non-negative integers.")
    if p == 0 and q == 0: raise ValueError("GARCH orders p and q cannot both be zero.")
    series = df[endog_var].dropna()
    if series.empty: raise ValueError("No valid data points.")
    if pd.api.types.infer_dtype(series) not in ('floating', 'integer'): raise ValueError(f"GARCH variable '{endog_var}' must be numeric.")
    if series.var(skipna=True) < 1e-9: raise ValueError(f"Data for '{endog_var}' has near-zero variance. GARCH model cannot be fitted.")

    try:
        scaled_series = series * 100
        model = arch_model(scaled_series, vol='Garch', p=p, q=q, mean='Constant', dist='Normal')
        results = model.fit(disp='off')
        diagnostics_results = [{"name":"GARCH Diagnostics", "interpretation": "Run specific post-tests like Ljung-Box on Standardized Residuals and Squared Standardized Residuals."}]
    except Exception as e:
        raise RuntimeError(f"Error fitting GARCH({p},{q}) model: {e}. Check data (often needs returns), orders, and ensure variance exists.") from e

    metrics = {
        "R-squared": None,
        "Adj. R-squared": None,
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": results.summary().as_html(),
        "diagnostics": diagnostics_results,
        "metrics": metrics,
        "fitted_model_object": results
    }

# --- Panel Suite (Final Fix for Hausman) ---
def run_panel_suite(df, dependent_var, independent_vars, panel_id_var, panel_time_var):
    if PooledOLS is None: # Check if linearmodels imported
        raise ImportError("Panel models cannot run. The 'linearmodels' library is not installed.")
    
    if not dependent_var or not independent_vars: raise ValueError("Dependent and Independent variables required.")
    if not panel_id_var or not panel_time_var: raise ValueError("Panel ID and Time variables must be specified.")
    
    required_cols = [dependent_var] + independent_vars + [panel_id_var, panel_time_var]
    if not all(v in df.columns for v in required_cols): raise ValueError("One or more specified variables not found.")
    
    try:
        # Ensure correct types for index
        panel_data = df[required_cols].copy()
        panel_data[panel_time_var] = pd.to_datetime(panel_data[panel_time_var], errors='coerce').fillna(panel_data[panel_time_var])
        
        panel_data = panel_data.set_index([panel_id_var, panel_time_var])
        panel_data = panel_data.sort_index()
        panel_data = panel_data.dropna()
    except Exception as idx_e:
        raise ValueError(f"Error setting up panel index: {idx_e}.")
    
    if panel_data.empty: raise ValueError("No valid data points after setting index and dropping NaNs.")
    
    Y = panel_data[dependent_var]
    X_vars_only = panel_data[independent_vars]
    
    # Ensure numeric
    if not pd.api.types.is_numeric_dtype(Y): raise ValueError(f"Dependent variable '{dependent_var}' must be numeric.")
    
    X_with_const = sm.add_constant(X_vars_only, has_constant='raise')
    
    fitted_models = {}
    
    # 1. Pooled OLS
    try:
        model_pooled = PooledOLS(Y, X_with_const)
        fitted_models['Pooled'] = model_pooled.fit(cov_type='robust')
    except Exception as e: print(f"Pooled OLS failed: {e}")

    # 2. Fixed Effects
    try:
        model_fe = PanelOLS(Y, X_vars_only, entity_effects=True) 
        fitted_models['FE'] = model_fe.fit(cov_type='robust')
    except Exception as e: print(f"Fixed Effects failed: {e}")

    # 3. Random Effects
    try:
        model_re = RandomEffects(Y, X_with_const)
        fitted_models['RE'] = model_re.fit(cov_type='robust')
    except Exception as e: print(f"Random Effects failed: {e}")

    # --- Hausman Test (Manual Calculation) ---
    # This approach is robust across library versions
    hausman_summary_html = "Hausman test requires successful FE and RE models."
    hausman_results_list = []
    preferred_model_object = None
    
    # Generate comparison table HTML
    try:
        from linearmodels.panel import compare
        comp = compare(fitted_models)
        hausman_summary_html = comp.summary.as_html()
    except Exception:
        hausman_summary_html = "<p>Could not generate comparison table.</p>"

    if 'FE' in fitted_models and 'RE' in fitted_models:
        try:
            fe_params = fitted_models['FE'].params
            re_params = fitted_models['RE'].params
            fe_cov = fitted_models['FE'].cov
            re_cov = fitted_models['RE'].cov
            
            # Get common variables (Hausman compares coefficients of time-varying vars)
            common_vars = [v for v in fe_params.index if v in re_params.index and v != 'const']
            
            if not common_vars:
                raise ValueError("No common time-varying variables to compare.")

            b_diff = fe_params[common_vars] - re_params[common_vars]
            v_diff = fe_cov.loc[common_vars, common_vars] - re_cov.loc[common_vars, common_vars]
            
            # Calculate Hausman Statistic: (b_fe - b_re)' * inv(V_fe - V_re) * (b_fe - b_re)
            # Using pseudo-inverse to handle potential singularity
            hausman_stat = b_diff.T @ np.linalg.pinv(v_diff) @ b_diff
            df_hausman = len(common_vars)
            
            from scipy.stats import chi2
            hausman_pvalue = chi2.sf(hausman_stat, df_hausman)
            
            interpretation = "Hausman test failed."
            if not np.isnan(hausman_pvalue):
                if hausman_pvalue <= 0.05:
                    interpretation = "Reject H0 (p<=0.05): **Fixed Effects (FE)** is preferred (Consistent)."
                    preferred_model_object = fitted_models['FE']
                else:
                    interpretation = "Fail to Reject H0 (p>0.05): **Random Effects (RE)** is preferred (Efficient)."
                    preferred_model_object = fitted_models['RE']
            
            hausman_results_list.append({ 
                "name": "Model Comparison (Hausman: FE vs RE)", 
                "statistic": round(hausman_stat, 4), 
                "p_value": round(hausman_pvalue, 4), 
                "interpretation": interpretation 
            })
            
        except Exception as he:
            print(f"Hausman Error: {he}")
            # Fallback logic if calculation fails (e.g. matrices not aligned)
            hausman_results_list.append({"name": "Model Comparison (Hausman)", "interpretation": f"Manual Calculation Error: {he}"})
            preferred_model_object = fitted_models.get('FE') # Default to FE on error as it's safer
    else:
         hausman_results_list.append({"name": "Model Comparison (Hausman)", "interpretation": "Could not run FE and RE models successfully to compare."})

    if preferred_model_object is None:
        preferred_model_object = fitted_models.get('FE') or fitted_models.get('RE') or fitted_models.get('Pooled')

    metrics = {
        "R-squared": getattr(preferred_model_object, 'rsquared', None),
        "N. Observations": int(getattr(preferred_model_object, 'nobs', 0))
    }

    return {
        "comparison_html": hausman_summary_html,
        "diagnostics": hausman_results_list,
        "metrics": metrics,
        "fitted_model_object": preferred_model_object 
    }

# --- Logit Model ---
def run_logit_model(df, dependent_var, independent_vars):
    if not dependent_var or not independent_vars: raise ValueError("Dependent and Independent variables required.")
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: raise ValueError("No common valid data points.")
    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    unique_Y = Y.unique()
    is_binary = np.all(np.isin(unique_Y, [0, 1]))
    is_single_value_binary = len(unique_Y) == 1 and (unique_Y[0] == 0 or unique_Y[0] == 1)
    if not is_binary and not is_single_value_binary: raise ValueError(f"Dependent variable '{dependent_var}' must be binary (0/1) for Logit.")
    if is_single_value_binary: print(f"Warning: Dependent variable '{dependent_var}' contains only one value ({unique_Y[0]}).")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty: raise ValueError(f"Logit independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")
    X = sm.add_constant(X, has_constant='raise')
    try:
        model = Logit(Y, X)
        results = model.fit(disp=False)
        try: diagnostics_results = run_single_equation_diagnostics(results)
        except Exception as diag_e: diagnostics_results = [{"name":"Diagnostics Note", "interpretation": f"Standard diagnostics failed: {diag_e}"}]
        diagnostics_results.append({"name":"Logit Diagnostics", "interpretation": "Check Pseudo R-sq, LR test in summary. Run Classification Report as post-test."})
    except Exception as e:
        raise RuntimeError(f"Error fitting Logit model: {e}") from e

    metrics = {
        "R-squared": None,
        "Adj. R-squared": getattr(results, 'prsquared', None),
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": results.summary().as_html(),
        "diagnostics": diagnostics_results,
        "metrics": metrics,
        "fitted_model_object": results
    }

# --- Probit Model ---
def run_probit_model(df, dependent_var, independent_vars):
    if not dependent_var or not independent_vars: raise ValueError("Dependent and Independent variables required.")
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: raise ValueError("No common valid data points.")
    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    unique_Y = Y.unique()
    is_binary = np.all(np.isin(unique_Y, [0, 1]))
    is_single_value_binary = len(unique_Y) == 1 and (unique_Y[0] == 0 or unique_Y[0] == 1)
    if not is_binary and not is_single_value_binary: raise ValueError(f"Dependent variable '{dependent_var}' must be binary (0/1) for Probit.")
    if is_single_value_binary: print(f"Warning: Dependent variable '{dependent_var}' contains only one value ({unique_Y[0]}).")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty: raise ValueError(f"Probit independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")
    X = sm.add_constant(X, has_constant='raise')
    try:
        model = Probit(Y, X)
        results = model.fit(disp=False)
        try: diagnostics_results = run_single_equation_diagnostics(results)
        except Exception as diag_e: diagnostics_results = [{"name":"Diagnostics Note", "interpretation": f"Standard diagnostics failed: {diag_e}"}]
        diagnostics_results.append({"name":"Probit Diagnostics", "interpretation": "Check Pseudo R-sq, LR test in summary. Run Classification Report as post-test."})
    except Exception as e:
        raise RuntimeError(f"Error fitting Probit model: {e}") from e

    metrics = {
        "R-squared": None,
        "Adj. R-squared": getattr(results, 'prsquared', None),
        "AIC": getattr(results, 'aic', None),
        "BIC": getattr(results, 'bic', None),
        "N. Observations": int(getattr(results, 'nobs', 0))
    }

    return {
        "summary_html": results.summary().as_html(),
        "diagnostics": diagnostics_results,
        "metrics": metrics,
        "fitted_model_object": results
    }

# --- Machine Learning Models ---
def run_ml_model(df, model_type, dependent_var, independent_vars):
    if not dependent_var or not independent_vars:
        raise ValueError("Dependent and Independent variables must be selected.")
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: raise ValueError("No common valid data points after dropping NaNs.")

    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    
    if pd.api.types.infer_dtype(Y) not in ('floating', 'integer'):
            raise ValueError(f"ML models require numeric dependent variable. '{dependent_var}' is not numeric.")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty:
            raise ValueError(f"ML models require numeric independent variables. Non-numeric found: {', '.join(non_numeric_X)}")

    try:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        if X_train.empty or X_test.empty:
             raise ValueError("Not enough data to create a train/test split. Need more observations.")
    except Exception as split_e:
        raise ValueError(f"Error during data split: {split_e}. Ensure you have at least 5 data rows.")

    start_time = time.time()
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown ML model type: {model_type}")

    try:
        model.fit(X_train, Y_train)
    except Exception as fit_e:
        raise RuntimeError(f"Error fitting {model_type} model: {fit_e}")
    
    fit_time = time.time() - start_time
    
    Y_pred = model.predict(X_test)
    
    r2_test = r2_score(Y_test, Y_pred)
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_pred))
    
    importances = {}
    if hasattr(model, 'feature_importances_'):
        importances = {name: float(imp) for name, imp in zip(X.columns, model.feature_importances_)}
        importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    summary_html = f"<h3>{model_type.replace('_', ' ').title()} Results (on Test Set)</h3>"
    summary_html += "<p>Model trained on the first 80% of data (time-series split) and tested on the last 20%.</p>"
    
    summary_html += "<h4>Model Performance (Test Set):</h4>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 50%;'>"
    summary_html += f"<tr><td style='padding: 5px;'><b>R-squared (Test)</b></td><td style='padding: 5px;'><b>{r2_test:.4f}</b></td></tr>"
    summary_html += f"<tr><td style='padding: 5px;'>RMSE (Test)</td><td style='padding: 5px;'>{rmse_test:.4f}</td></tr>"
    summary_html += f"<tr><td style='padding: 5px;'>Fit Time (seconds)</td><td style='padding: 5px;'>{fit_time:.4f}</td></tr>"
    summary_html += "</table><br/>"

    if importances:
        summary_html += "<h4>Feature Importance:</h4>"
        summary_html += "<table border='1' style='border-collapse: collapse; width: 50%;'>"
        summary_html += "<tr><th style='padding: 5px;'>Feature (Variable)</th><th style='padding: 5px;'>Importance Score</th></tr>"
        for name, imp in importances.items():
            summary_html += f"<tr><td style='padding: 5px;'>{name}</td><td style='padding: 5px;'>{imp:.4f}</td></tr>"
        summary_html += "</table>"
    
    metrics = {
        "R-squared": None,
        "Adj. R-squared": r2_test,
        "AIC": None,
        "BIC": None,
        "N. Observations": int(len(Y))
    }

    diagnostics = [
        {"name": "Model Type", "interpretation": f"Machine Learning Regressor ({model_type})."},
        {"name": "Validation", "interpretation": f"Evaluated on 20% hold-out test set. R-squared (Test): {r2_test:.4f}", "statistic": r2_test, "p_value": None}
    ]
    if r2_test < 0:
        diagnostics.append({"name": "Warning", "interpretation": "Test R-squared is negative. The model performs worse than a simple horizontal line (the mean). It is not useful for prediction.", "p_value": 0.0})

    fitted_model_bundle = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "Y_train": Y_train,
        "Y_test": Y_test
    }

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "fitted_model_object": fitted_model_bundle 
    }

# --- LassoCV Model ---
def run_lasso_model(df, dependent_var, independent_vars):
    if not dependent_var or not independent_vars:
        raise ValueError("Dependent and Independent variables must be selected for Lasso.")
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: 
        raise ValueError("No common valid data points after dropping NaNs.")
    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    if pd.api.types.infer_dtype(Y) not in ('floating', 'integer'):
        raise ValueError(f"Lasso dependent variable '{dependent_var}' must be numeric.")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty:
        raise ValueError(f"Lasso independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000))
    ])

    try:
        pipeline.fit(X, Y)
    except Exception as e:
        raise RuntimeError(f"Error fitting LassoCV model: {e}")

    best_alpha = pipeline.named_steps['lasso'].alpha_
    lasso_coefs = pipeline.named_steps['lasso'].coef_
    intercept = pipeline.named_steps['lasso'].intercept_
    
    r2_score_val = pipeline.score(X, Y)
    n_obs = len(Y)
    
    # (!!!) (هذا هو الإصلاح لـ SyntaxWarning) (!!!)
    summary_html = f"<h3>Lasso Regression Results (with Cross-Validation)</h3>"
    summary_html += fr"<p>Model selected the optimal alpha (penalty) $\lambda = {best_alpha:.6f}$ using 5-fold Cross-Validation.</p>"
    
    summary_html += "<h4>Model Performance (In-Sample):</h4>"
    summary_html += f"<table border='1' style='border-collapse: collapse; width: 50%;'><tr><td style='padding: 5px;'><b>R-squared</b></td><td style='padding: 5px;'><b>{r2_score_val:.4f}</b></td></tr><tr><td style='padding: 5px;'>N. Observations</td><td style='padding: 5px;'>{n_obs}</td></tr></table><br/>"
    summary_html += "<h4>Selected Coefficients (Shrunk):</h4>"
    summary_html += "<p>Note: Coefficients are shrunk towards zero. Variables with a coefficient of 0.0000 were excluded by the model.</p>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 50%;'>"
    summary_html += "<tr><th style='padding: 5px;'>Feature (Variable)</th><th style='padding: 5px;'>Coefficient</th></tr>"
    summary_html += f"<tr><td style='padding: 5px;'><b>const (Intercept)</b></td><td style='padding: 5px;'><b>{intercept:.4f}</b></td></tr>"
    selected_vars = 0
    for coef, name in sorted(zip(lasso_coefs, independent_vars), key=lambda x: -abs(x[0])):
        is_zero = np.isclose(coef, 0)
        if not is_zero:
            selected_vars += 1
        summary_html += f"<tr><td style='padding: 5px;'>{name}</td><td style='padding: 5px; {'font-weight:bold; color:black;' if not is_zero else 'color: #999;'}'>{coef:.4f}</td></tr>"
    summary_html += "</table>"
    
    metrics = {
        "R-squared": r2_score_val,
        "Adj. R-squared": None,
        "AIC": None,
        "BIC": None,
        "N. Observations": n_obs
    }

    diagnostics = [
        {"name": "Model Type", "interpretation": f"Lasso Regression (L1 Penalty)."},
        {"name": "Optimal Alpha (λ)", "statistic": best_alpha, "p_value": None, "interpretation": f"Best penalty chosen by 5-fold CV: {best_alpha:.6f}"},
        {"name": "Variables Selected", "statistic": selected_vars, "p_value": None, "interpretation": f"Lasso selected {selected_vars} out of {len(independent_vars)} variables (non-zero coefficients)."}
    ]
    if selected_vars == 0:
        diagnostics.append({"name": "Warning", "interpretation": "Lasso shrunk all coefficients to zero. The model is just the intercept. This implies no X-variables were useful."})

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "fitted_model_object": pipeline
    }

# --- Causal ML (DML) Model ---
def run_double_ml_model(df, dependent_var, treatment_var, control_vars, ml_method='lasso'):
    """
    Executes Double Machine Learning (DML) and bundles residuals for post-tests.
    """
    # 1. Data Prep
    required_cols = [dependent_var, treatment_var] + control_vars
    model_data = df[required_cols].dropna()
    
    if model_data.empty: raise ValueError("No valid data points.")
    if len(model_data) < 10: raise ValueError("Not enough data for DML (Need 10+).")

    Y = model_data[dependent_var]
    D = model_data[treatment_var]
    X = model_data[control_vars]

    # 2. Select ML Model
    if ml_method == 'lasso':
        model_g = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1, max_iter=5000))])
        model_h = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1, max_iter=5000))])
    elif ml_method == 'random_forest':
        model_g = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
        model_h = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {ml_method}")

    # 3. Cross-Fitting
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    eta_y = np.zeros(len(model_data))
    eta_d = np.zeros(len(model_data))

    print(f"Running DML with {ml_method}...")
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]
        D_tr, D_te = D.iloc[train_idx], D.iloc[test_idx]

        model_g.fit(X_tr, Y_tr)
        eta_y[test_idx] = Y_te - model_g.predict(X_te)

        model_h.fit(X_tr, D_tr)
        eta_d[test_idx] = D_te - model_h.predict(X_te)

    # 4. Final Estimate
    eta_d_const = sm.add_constant(eta_d)
    final_ols = sm.OLS(eta_y, eta_d_const).fit()
    
    # Extract Stats
    alpha = final_ols.params[1]
    pval = final_ols.pvalues[1]
    conf = final_ols.conf_int()[1]

    # 5. Output Formatting
    summary_html = f"<h3>Causal ML (DML) Results</h3>"
    summary_html += f"<p><b>ML Method:</b> {ml_method.title()}</p>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 100%; text-align: center;'>"
    summary_html += "<tr style='background-color: #f0fdf4;'><th>Effect (Alpha)</th><th>Std. Err</th><th>P-Value</th><th>[0.025</th><th>0.975]</th></tr>"
    
    color = "green" if pval <= 0.05 else "gray"
    summary_html += f"<tr><td><b>{alpha:.4f}</b></td><td>{final_ols.bse[1]:.4f}</td>"
    summary_html += f"<td style='color:{color}; font-weight:bold;'>{pval:.4f}</td>"
    summary_html += f"<td>{conf[0]:.4f}</td><td>{conf[1]:.4f}</td></tr></table>"

    diagnostics = [
        {"name": "Causal Estimate", "statistic": alpha, "p_value": pval, "interpretation": f"One unit increase in {treatment_var} causes {alpha:.4f} increase in {dependent_var}."},
        {"name": "Robustness", "interpretation": "Run 'Sensitivity Analysis' to check stability."}
    ]

    # (!!!) الحزمة الكاملة للاختبارات البعدية (!!!)
    dml_result_bundle = {
        "final_ols": final_ols,        # كائن OLS للاستخدام في Sensitivity
        "residuals_y": eta_y,          # البواقي لاختبار Heterogeneity
        "residuals_d": eta_d,
        "original_X": X,               # المتغيرات الأصلية
        "treatment_var": treatment_var,
        "dependent_var": dependent_var
    }

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": {"N": len(model_data), "R2_Aux": final_ols.rsquared},
        "fitted_model_object": dml_result_bundle  # (!!!) إرجاع الحزمة بدلاً من OLS فقط
    }


# --- ElasticNetCV Model ---
def run_elasticnet_model(df, dependent_var, independent_vars):
    if not dependent_var or not independent_vars:
        raise ValueError("Dependent and Independent variables must be selected for ElasticNet.")
    
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: 
        raise ValueError("No common valid data points after dropping NaNs.")

    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    
    if pd.api.types.infer_dtype(Y) not in ('floating', 'integer'):
        raise ValueError(f"ElasticNet dependent variable '{dependent_var}' must be numeric.")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty:
        raise ValueError(f"ElasticNet independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0], 
                                    cv=5, random_state=42, n_jobs=-1, max_iter=2000))
    ])

    try:
        pipeline.fit(X, Y)
    except Exception as e:
        raise RuntimeError(f"Error fitting ElasticNetCV model: {e}")

    best_alpha = pipeline.named_steps['elasticnet'].alpha_
    best_l1_ratio = pipeline.named_steps['elasticnet'].l1_ratio_
    coefs = pipeline.named_steps['elasticnet'].coef_
    intercept = pipeline.named_steps['elasticnet'].intercept_
    
    r2_score_val = pipeline.score(X, Y)
    n_obs = len(Y)
    
    # (!!!) (هذا هو الإصلاح لـ SyntaxWarning) (!!!)
    summary_html = f"<h3>Elastic-Net Regression Results (with Cross-Validation)</h3>"
    summary_html += fr"<p>Model selected optimal alpha (penalty) $\lambda = {best_alpha:.6f}$ and l1_ratio = {best_l1_ratio:.2f} using 5-fold CV.</p>"

    summary_html += "<h4>Model Performance (In-Sample):</h4>"
    summary_html += f"<table border='1' style='border-collapse: collapse; width: 50%;'><tr><td style='padding: 5px;'><b>R-squared</b></td><td style='padding: 5px;'><b>{r2_score_val:.4f}</b></td></tr><tr><td style='padding: 5px;'>N. Observations</td><td style='padding: 5px;'>{n_obs}</td></tr></table><br/>"
    summary_html += "<h4>Selected Coefficients (Shrunk):</h4>"
    summary_html += "<p>Note: Coefficients are shrunk towards zero. Variables with a coefficient of 0.0000 were excluded by the model.</p>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 50%;'>"
    summary_html += "<tr><th style='padding: 5px;'>Feature (Variable)</th><th style='padding: 5px;'>Coefficient</th></tr>"
    summary_html += f"<tr><td style='padding: 5px;'><b>const (Intercept)</b></td><td style='padding: 5px;'><b>{intercept:.4f}</b></td></tr>"
    selected_vars = 0
    for coef, name in sorted(zip(coefs, independent_vars), key=lambda x: -abs(x[0])):
        is_zero = np.isclose(coef, 0)
        if not is_zero:
            selected_vars += 1
        summary_html += f"<tr><td style='padding: 5px;'>{name}</td><td style='padding: 5px; {'font-weight:bold; color:black;' if not is_zero else 'color: #999;'}'>{coef:.4f}</td></tr>"
    summary_html += "</table>"
    
    metrics = {
        "R-squared": r2_score_val,
        "Adj. R-squared": None, 
        "AIC": None,
        "BIC": None,
        "N. Observations": n_obs
    }

    diagnostics = [
        {"name": "Model Type", "interpretation": f"Elastic-Net Regression (L1+L2 Penalty)."},
        {"name": "Optimal Alpha (λ)", "statistic": best_alpha, "p_value": None, "interpretation": f"Best penalty chosen by 5-fold CV: {best_alpha:.6f}"},
        {"name": "Optimal L1 Ratio", "statistic": best_l1_ratio, "p_value": None, "interpretation": f"Best L1/L2 mix chosen by CV: {best_l1_ratio:.2f} (1=Lasso, 0=Ridge)"},
        {"name": "Variables Selected", "statistic": selected_vars, "p_value": None, "interpretation": f"Elastic-Net selected {selected_vars} out of {len(independent_vars)} variables (non-zero coefficients)."}
    ]
    if selected_vars == 0:
        diagnostics.append({"name": "Warning", "interpretation": "Elastic-Net shrunk all coefficients to zero. The model is just the intercept."})

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "fitted_model_object": pipeline
    }

# --- RidgeCV Model ---
def run_ridge_model(df, dependent_var, independent_vars):
    if not dependent_var or not independent_vars:
        raise ValueError("Dependent and Independent variables must be selected for Ridge.")
    
    required_cols = [dependent_var] + independent_vars
    model_data = df[required_cols].dropna()
    if model_data.empty: 
        raise ValueError("No common valid data points after dropping NaNs.")

    Y = model_data[dependent_var]
    X = model_data[independent_vars]
    
    if pd.api.types.infer_dtype(Y) not in ('floating', 'integer'):
        raise ValueError(f"Ridge dependent variable '{dependent_var}' must be numeric.")
    non_numeric_X = X.select_dtypes(exclude=np.number).columns
    if not non_numeric_X.empty:
        raise ValueError(f"Ridge independent variables must be numeric. Non-numeric found: {', '.join(non_numeric_X)}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=np.logspace(-6, 6, 100), cv=5))
    ])

    try:
        pipeline.fit(X, Y)
    except Exception as e:
        raise RuntimeError(f"Error fitting RidgeCV model: {e}")

    best_alpha = pipeline.named_steps['ridge'].alpha_
    coefs = pipeline.named_steps['ridge'].coef_
    intercept = pipeline.named_steps['ridge'].intercept_
    
    r2_score_val = pipeline.score(X, Y)
    n_obs = len(Y)
    
    # (!!!) (هذا هو الإصلاح لـ SyntaxWarning) (!!!)
    summary_html = f"<h3>Ridge Regression Results (with Cross-Validation)</h3>"
    summary_html += fr"<p>Model selected optimal alpha (penalty) $\lambda = {best_alpha:.6f}$ using 5-fold Cross-Validation.</p>"
    
    summary_html += "<h4>Model Performance (In-Sample):</h4>"
    summary_html += f"<table border='1' style='border-collapse: collapse; width: 50%;'><tr><td style='padding: 5px;'><b>R-squared</b></td><td style='padding: 5px;'><b>{r2_score_val:.4f}</b></td></tr><tr><td style='padding: 5px;'>N. Observations</td><td style='padding: 5px;'>{n_obs}</td></tr></table><br/>"
    summary_html += "<h4>Coefficients:</h4>"
    summary_html += "<p>Note: Ridge regression shrinks coefficients towards zero but does not force them to be exactly zero.</p>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 50%;'>"
    summary_html += "<tr><th style='padding: 5px;'>Feature (Variable)</th><th style='padding: 5px;'>Coefficient</th></tr>"
    summary_html += f"<tr><td style='padding: 5px;'><b>const (Intercept)</b></td><td style='padding: 5px;'><b>{intercept:.4f}</b></td></tr>"
    for coef, name in sorted(zip(coefs, independent_vars), key=lambda x: -abs(x[0])):
        summary_html += f"<tr><td style='padding: 5px;'>{name}</td><td style='padding: 5px; font-weight:bold; color:black;'>{coef:.4f}</td></tr>"
    summary_html += "</table>"
    
    metrics = {
        "R-squared": r2_score_val,
        "Adj. R-squared": None, 
        "AIC": None,
        "BIC": None,
        "N. Observations": n_obs
    }

    diagnostics = [
        {"name": "Model Type", "interpretation": f"Ridge Regression (L2 Penalty)."},
        {"name": "Optimal Alpha (λ)", "statistic": best_alpha, "p_value": None, "interpretation": f"Best penalty chosen by 5-fold CV: {best_alpha:.6f}"}
    ]

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": metrics,
        "fitted_model_object": pipeline
    }

# --- (دوال اقتراح النموذج - لا تغيير) ---
def _run_adf_test_internal(df, variable, test_level, regression_type):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise ValueError(f"Variable '{variable}' must be numeric for ADF test.")
    series = df[variable].dropna()
    if series.empty:
        raise ValueError(f"Variable '{variable}' contains no valid data after dropping NaNs.")
    if test_level == '1st_diff':
        series = series.diff().dropna()
    elif test_level == '2nd_diff':
        series = series.diff().diff().dropna()
    if len(series) < 4:
        raise ValueError(f"Too few observations ({len(series)}) to perform ADF test.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series, regression=regression_type, autolag='AIC')
    except Exception as e:
        raise RuntimeError(f"Error during ADF calculation: {e}") from e
    is_stationary = bool(result[1] <= 0.05)
    return { "is_stationary": is_stationary, "p_value": float(result[1]) }

def _run_kpss_test_internal(df, variable, test_level, regression_type):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[variable]):
        raise ValueError(f"Variable '{variable}' must be numeric for KPSS test.")
    series = df[variable].dropna()
    if series.empty:
        raise ValueError(f"Variable '{variable}' contains no valid data after dropping NaNs.")
    if test_level == '1st_diff':
        series = series.diff().dropna()
    elif test_level == '2nd_diff':
        series = series.diff().diff().dropna()
    if len(series) < 4:
        raise ValueError(f"Too few observations ({len(series)}) to perform KPSS test.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series, regression=regression_type, nlags='auto')
    except Exception as e:
        raise RuntimeError(f"Error during KPSS calculation: {e}") from e
    statistic, p_value, lags, crit_values = result
    crit_5pct = crit_values.get('5%')
    is_non_stationary = bool(crit_5pct is not None and statistic > crit_5pct)
    is_stationary = not is_non_stationary 
    return { "is_stationary": is_stationary, "p_value": float(p_value) }

# --- دالة مساعدة لاختبار السكون (Stationarity) ---
def _check_stationarity(df, var_name):
    """
    اختبار ذكي يجمع بين ADF و KPSS لتحديد حالة المتغير بدقة.
    Returns: 'I(0)', 'I(1)', or 'Uncertain'
    """
    series = df[var_name].dropna()
    
    # ADF Test (H0: Non-Stationary)
    adf_result = adfuller(series, autolag='AIC')
    adf_p = adf_result[1]
    is_adf_stationary = adf_p < 0.05

    # KPSS Test (H0: Stationary)
    # نستخدم 'c' (constant) كافتراضي
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_p = kpss_result[1]
    is_kpss_stationary = kpss_p >= 0.05

    if is_adf_stationary and is_kpss_stationary:
        return 'I(0)' # ساكنة في المستوى
    elif not is_adf_stationary and not is_kpss_stationary:
        return 'I(1)' # غير ساكنة (غالباً تصبح ساكنة بعد الفرق الأول)
    elif is_adf_stationary and not is_kpss_stationary:
        return 'I(0)' # Difference Stationary (نميل لنتائج ADF عادة)
    else:
        return 'I(1)' # Trend Stationary (تحتاج Detrending أو الفرق)

def get_model_suggestion(df, y_vars, x_vars, is_panel, panel_id, panel_time):
    """
    محرك اقتراح النماذج (Smart Decision Tree based on Stationarity).
    """
    # 1. تجميع المتغيرات وتنظيفها
    all_vars = y_vars + x_vars
    unique_vars = list(dict.fromkeys(all_vars))
    
    if not all_vars:
         return { "recommended_model": "Error", "justification": "Please select at least one variable." }

    # ==========================================
    # Phase 1: Structural Checks (Panel & Binary)
    # ==========================================
    
    # A. Panel Data Check
    if is_panel:
        if not panel_id or not panel_time:
             return { "recommended_model": "Setup Error", "justification": "Panel mode is ON, but Entity ID or Time ID is missing." }
        return { 
            "recommended_model": "Panel Data Models (FE/RE)", 
            "justification": "Data structure is identified as Panel Data. Standard Time-Series models (like VAR/ARDL) are not suitable. Use Fixed Effects or Random Effects." 
        }

    # B. Binary Outcome Check
    if len(y_vars) == 1:
        y_col = df[y_vars[0]].dropna()
        if len(y_col.unique()) == 2 and sorted(y_col.unique().tolist()) == [0, 1]:
            return {
                "recommended_model": "Logit / Probit",
                "justification": f"The dependent variable '{y_vars[0]}' is Binary (0/1). Linear models are invalid here."
            }

    # ==========================================
    # Phase 2: Stationarity Analysis (The Core)
    # ==========================================
    stationarity_results = {}
    for var in unique_vars:
        if not pd.api.types.is_numeric_dtype(df[var]): continue
        try:
            stationarity_results[var] = _check_stationarity(df, var)
        except:
            stationarity_results[var] = 'Uncertain'

    total_vars = len(stationarity_results)
    count_i0 = list(stationarity_results.values()).count('I(0)')
    count_i1 = list(stationarity_results.values()).count('I(1)')
    
    # ------------------------------------------
    # Scenario A: All Variables are Stationary I(0)
    # ------------------------------------------
    if count_i0 == total_vars:
        if len(y_vars) > 1:
            return { 
                "recommended_model": "Vector Autoregression (VAR)", 
                "justification": "All variables are Stationary (I(0)). A **VAR** model is ideal for analyzing the interdependencies between them without differencing." 
            }
        else:
            return { 
                "recommended_model": "OLS Regression", 
                "justification": "All variables are Stationary (I(0)). Standard **OLS** is efficient and unbiased. (Consider ARIMA if you suspect autocorrelation)." 
            }

    # ------------------------------------------
    # Scenario B: Mixed Integration (I(0) + I(1))
    # ------------------------------------------
    if count_i0 > 0 and count_i1 > 0:
        return { 
            "recommended_model": "Autoregressive Distributed Lag (ARDL)", 
            "justification": f"Data is Mixed: {count_i0} variable(s) are I(0) and {count_i1} are I(1). **ARDL** is the only standard methodology designed specifically for mixed orders of integration using the Bounds Test." 
        }

    # ------------------------------------------
    # Scenario C: All Variables are Non-Stationary I(1)
    # ------------------------------------------
    if count_i1 == total_vars:
        # هنا التعديل الذكي: لا نعتمد على جوهانسين فقط لتجنب التضارب
        # بل نقترح المسار الأكثر شيوعاً في الأدبيات
        
        suggestion = "VECM or ARDL"
        reason = f"All variables are Non-Stationary (I(1)). This usually implies a long-run relationship (Cointegration).\n\n" \
                 f"• **Recommendation 1 (VECM):** Use if you want to model the system as a whole (requires Cointegration).\n" \
                 f"• **Recommendation 2 (ARDL):** Use if you focus on one dependent variable (more robust for small samples)."
        
        # محاولة تشغيل Johansen كـ "تلميح" إضافي وليس كحكم نهائي
        try:
            model_data = df[unique_vars].dropna()
            # شرط بسيط: البيانات تكفي للاختبار
            if len(model_data) > len(unique_vars) * 5: 
                johansen_res = coint_johansen(model_data, det_order=0, k_ar_diff=1)
                rank = 0
                for i in range(len(unique_vars)):
                    if johansen_res.lr1[i] > johansen_res.cvt[i, 1]: rank += 1
                    else: break
                
                if rank > 0:
                    suggestion = "VECM"
                    reason += f"\n\n(Note: Automatic Johansen test detected Rank={rank}, supporting **VECM**)."
                else:
                    # إذا فشل التكام، نعود لـ VAR على الفروق
                    suggestion = "VAR (on Differences)"
                    reason = "All variables are I(1), and a quick Johansen test suggests **NO Cointegration**. You should likely difference your data and use **VAR**, or try **ARDL** to double-check."
        except:
            pass # لو فشل الاختبار، نكتفي بالاقتراح العام

        return { "recommended_model": suggestion, "justification": reason }

    # Default Fallback
    return { 
        "recommended_model": "Manual Investigation", 
        "justification": "Stationarity tests were inconclusive (variables might be I(2) or data is insufficient). Please review the 'Stability Tests' tab." 
    }

def run_double_ml_model(df, dependent_var, treatment_var, control_vars, ml_method='lasso'):
    """
    Executes Double Machine Learning (DML) and bundles residuals for post-tests.
    """
    # 1. Data Prep
    required_cols = [dependent_var, treatment_var] + control_vars
    model_data = df[required_cols].dropna()
    
    if model_data.empty: raise ValueError("No valid data points.")
    if len(model_data) < 10: raise ValueError("Not enough data for DML (Need 10+).")

    Y = model_data[dependent_var]
    D = model_data[treatment_var]
    X = model_data[control_vars]

    # 2. Select ML Model
    if ml_method == 'lasso':
        model_g = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1, max_iter=5000))])
        model_h = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1, max_iter=5000))])
    elif ml_method == 'random_forest':
        model_g = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
        model_h = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {ml_method}")

    # 3. Cross-Fitting
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    eta_y = np.zeros(len(model_data))
    eta_d = np.zeros(len(model_data))

    print(f"Running DML with {ml_method}...")
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        Y_tr, Y_te = Y.iloc[train_idx], Y.iloc[test_idx]
        D_tr, D_te = D.iloc[train_idx], D.iloc[test_idx]

        model_g.fit(X_tr, Y_tr)
        eta_y[test_idx] = Y_te - model_g.predict(X_te)

        model_h.fit(X_tr, D_tr)
        eta_d[test_idx] = D_te - model_h.predict(X_te)

    # 4. Final Estimate
    eta_d_const = sm.add_constant(eta_d)
    final_ols = sm.OLS(eta_y, eta_d_const).fit()
    
    # Extract Stats
    alpha = final_ols.params[1]
    pval = final_ols.pvalues[1]
    conf = final_ols.conf_int()[1]

    # 5. Output Formatting
    summary_html = f"<h3>Causal ML (DML) Results</h3>"
    summary_html += f"<p><b>ML Method:</b> {ml_method.title()}</p>"
    summary_html += "<table border='1' style='border-collapse: collapse; width: 100%; text-align: center;'>"
    summary_html += "<tr style='background-color: #f0fdf4;'><th>Effect (Alpha)</th><th>Std. Err</th><th>P-Value</th><th>[0.025</th><th>0.975]</th></tr>"
    
    color = "green" if pval <= 0.05 else "gray"
    summary_html += f"<tr><td><b>{alpha:.4f}</b></td><td>{final_ols.bse[1]:.4f}</td>"
    summary_html += f"<td style='color:{color}; font-weight:bold;'>{pval:.4f}</td>"
    summary_html += f"<td>{conf[0]:.4f}</td><td>{conf[1]:.4f}</td></tr></table>"

    diagnostics = [
        {"name": "Causal Estimate", "statistic": alpha, "p_value": pval, "interpretation": f"One unit increase in {treatment_var} causes {alpha:.4f} increase in {dependent_var}."},
        {"name": "Robustness", "interpretation": "Run 'Sensitivity Analysis' to check stability."}
    ]

    # (!!!) الحزمة الكاملة للاختبارات البعدية (!!!)
    dml_result_bundle = {
        "final_ols": final_ols,        # كائن OLS للاستخدام في Sensitivity
        "residuals_y": eta_y,          # البواقي لاختبار Heterogeneity
        "residuals_d": eta_d,
        "original_X": X,               # المتغيرات الأصلية
        "treatment_var": treatment_var,
        "dependent_var": dependent_var
    }

    return {
        "summary_html": summary_html,
        "diagnostics": diagnostics,
        "metrics": {"N": len(model_data), "R2_Aux": final_ols.rsquared},
        "fitted_model_object": dml_result_bundle  # (!!!) إرجاع الحزمة بدلاً من OLS فقط
    }

