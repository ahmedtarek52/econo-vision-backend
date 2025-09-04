# app/blueprints/modeling/utils.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA

def run_ols_model(df, endogenous_var, exogenous_vars):
    """
    Fits an Ordinary Least Squares (OLS) model.
    """
    if not exogenous_vars:
        raise ValueError("OLS model requires at least one exogenous variable.")
        
    Y = df[endogenous_var]
    X = df[exogenous_vars]
    X = sm.add_constant(X)  # Adds a constant term to the predictor
    
    model = sm.OLS(Y, X).fit()
    
    return {"model_summary": str(model.summary())}

def run_var_model(df, endogenous_vars):
    """
    Fits a Vector Autoregression (VAR) model.
    """
    model_data = df[endogenous_vars]
    model = VAR(model_data)
    results = model.fit()
    
    return {"model_summary": str(results.summary())}

def run_arima_model(df, endogenous_var):
    """
    Fits an ARIMA(p,d,q) model.
    For simplicity, we use a common order (1,1,1). A real app might auto-select.
    """
    series = df[endogenous_var]
    model = ARIMA(series, order=(1, 1, 1))
    results = model.fit()
    
    return {"model_summary": str(results.summary())}