import numpy as np
import statsmodels.api as sm
import traceback

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