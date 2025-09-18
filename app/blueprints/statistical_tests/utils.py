# app/blueprints/statistical_tests/utils.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ==========================================================
# 🧪 الاختبار 1: السكون (ADF Test)
# ==========================================================
def run_stationarity_tests(df):
    """Runs the ADF test on all numeric columns of a DataFrame."""
    stationarity_results = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10: continue
        
        result = adfuller(series)
        stationarity_results.append({
            "variable": col,
            "p_value": result[1],
            "is_stationary": bool(result[1] <= 0.05)
        })
    return stationarity_results

# ==========================================================
# 🧪 الاختبار 2: الارتباط الذاتي (ACF & PACF) لنموذج ARIMA
# ==========================================================
def run_autocorrelation_analysis(df):
    """Calculates ACF and PACF values for all numeric columns."""
    autocorrelation_results = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 20: continue

        acf_values = acf(series, nlags=20, fft=True).tolist()
        pacf_values = pacf(series, nlags=20).tolist()
        autocorrelation_results.append({
            "variable": col,
            "acf": acf_values,
            "pacf": pacf_values
        })
    return autocorrelation_results

# ==========================================================
# 🧪 الاختبار 3: الارتباط الخطي المتعدد (VIF)
# ==========================================================
def run_multicollinearity_test(df, independent_vars):
    """Calculates the Variance Inflation Factor (VIF) for independent variables."""
    if not independent_vars or len(independent_vars) < 2:
        return [] # VIF requires at least two independent variables
        
    X = df[independent_vars].dropna()
    # إضافة ثابت (intercept) للبيانات
    X_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["vif_factor"] = [variance_inflation_factor(X_const.values, i + 1) for i in range(len(X.columns))]
    
    return vif_data.to_dict(orient='records')

# ==========================================================
# 🧪 الاختبار 4: تحديد فترة الإبطاء المثلى (VAR Lag Order)
# ==========================================================
def run_optimal_lag_selection(df):
    """Selects the optimal lag order for a VAR model."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        return {}
        
    model = VAR(df[numeric_cols].dropna())
    # يمكنك زيادة maxlags إذا كانت بياناتك تسمح بذلك (ربع سنوية أو شهرية)
    selected_lags = model.select_order(maxlags=4)
    
    # تحويل ملخص النتائج إلى جدول HTML لسهولة عرضه
    return selected_lags.summary().as_html()

# ==========================================================
# 🧪 الاختبار 5: التكامل المشترك (Johansen Test)
# ==========================================================
def run_johansen_cointegration_test(df):
    """Performs the Johansen Cointegration test."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2 or len(df) < 20:
        return {}

    # det_order=0 يفترض وجود ثابت في علاقة التكامل المشترك
    # k_ar_diff=1 يفترض أن النموذج الأساسي هو VAR(2) على متغيرات المستوى
    result = coint_johansen(df[numeric_cols].dropna(), det_order=0, k_ar_diff=1)
    
    trace_stat = result.lr1
    trace_crit_vals = result.cvt
    
    # حساب عدد علاقات التكامل المشترك عند مستوى ثقة 95%
    num_cointegrating_relations = np.sum(trace_stat > trace_crit_vals[:, 1])
    
    return {
        "interpretation": f"The test suggests there are {num_cointegrating_relations} cointegrating relationships among the variables at the 95% significance level.",
        "details": f"Trace Statistic: {np.round(trace_stat, 2).tolist()}\nCritical Values (95%): {trace_crit_vals[:, 1].tolist()}"
    }
