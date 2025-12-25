# app/blueprints/ai_core/utils.py

import pandas as pd
import numpy as np
import traceback

# --- استيراد دوال الاختبارات بشكل آمن ---
try:
    # نحاول الاستيراد من المسار المفترض
    from app.blueprints.statistical_tests.utils import (
        run_adf_test, 
        run_kpss_test, 
        run_multicollinearity_test
    )
except ImportError:
    print("AI CORE WARNING: Could not import statistical_tests utils. AI Briefing functionality will be limited.")
    # دوال وهمية (Mock functions) لكي لا يتوقف الكود تماماً في حال خطأ الاستيراد
    def run_adf_test(df, v, **kwargs): return {"interpretation": "Error: ADF missing", "p_value": 1.0}
    def run_kpss_test(df, v, **kwargs): return {"interpretation": "Error: KPSS missing (non-stationary)"}
    def run_multicollinearity_test(df, v, **kwargs): return {"formatted_results": "Error: VIF missing"}


# --- (1) Stationarity Recommendation Logic ---
def get_stationarity_recommendation(adf_result=None, kpss_result=None):
    """
    Analyzes ADF and KPSS results to suggest stationarity treatment.
    """
    recommendation = "Recommendation: Check individual test interpretations."
    
    # Extract P-values and interpretations safely
    adf_p = adf_result.get('p_value') if adf_result else None
    kpss_interp = kpss_result.get('interpretation', '').lower() if kpss_result else ''
    
    # Determine status based on p-values (Alpha = 0.05)
    # ADF: Null = Non-Stationary. p > 0.05 -> Non-Stationary
    adf_nonstat = adf_p is not None and adf_p > 0.05
    
    # KPSS: Null = Stationary. 'non-stationary' in text indicates rejection of null
    kpss_nonstat = 'non-stationary' in kpss_interp 

    if adf_nonstat and kpss_nonstat:
        recommendation = "Recommendation (ADF & KPSS agree): Series is likely **Non-Stationary (I(1))**. Consider using the **1st Difference** for models requiring stationarity (like ARMA, VAR) or use models capable of handling I(1) data like ARIMA/VECM."
    
    elif adf_nonstat and not kpss_nonstat:
        recommendation = "Recommendation (Conflicting Results): **ADF suggests Non-Stationary, but KPSS suggests Stationary**. This typically indicates 'Difference Stationarity'. Double-check data plots. If differencing, try **d=1**."
    
    elif not adf_nonstat and kpss_nonstat:
         recommendation = "Recommendation (Conflicting Results): **ADF suggests Stationary, but KPSS suggests Non-Stationary**. This typically indicates 'Trend Stationarity'. Consider including a trend term in your model or proceed with caution (Try **d=0** initially)."
    
    elif not adf_nonstat and not kpss_nonstat:
         recommendation = "Recommendation (ADF & KPSS agree): Series is likely **Stationary (I(0))**. Use the original series (level) for modeling. Set **d=0** for ARIMA."
    
    return recommendation


# --- (2) Cointegration Recommendation Logic ---
def get_cointegration_recommendation(johansen_result=None, ardl_bounds_result=None):
    """
    Analyzes Johansen test results to suggest VECM or VAR.
    """
    recommendation = "Recommendation: Run relevant cointegration test first."
    
    if johansen_result:
        num_relations = johansen_result.get('cointegrating_relations', -1) 
        
        if num_relations > 0:
            recommendation = f"Recommendation (Johansen): **{num_relations} cointegrating relationship(s) found**. This strongly suggests a long-run relationship exists. Consider using a **VECM** model with rank = {num_relations}."
        elif num_relations == 0:
            recommendation = "Recommendation (Johansen): **No cointegration found**. Even if variables are I(1), they drift apart. Consider using **VAR on differenced data** to model short-run dynamics only."
        else:
            recommendation = "Recommendation (Johansen): Test results unclear or failed. Review test details manually."
            
    return recommendation


# --- (3) Panel Model Decision Logic ---
def get_panel_model_decision(hausman_result=None, panel_lm_result=None):
    """
    Suggests Pooled OLS vs RE vs FE based on Hausman and LM tests.
    """
    recommended_model = "Pooled OLS"
    decision_reason = ["Defaulting to Pooled OLS."]
    
    # 1. Check LM Test (Pooled OLS vs Random Effects)
    lm_pval = None
    if panel_lm_result and isinstance(panel_lm_result.get('p_value'), (int, float)):
        lm_pval = panel_lm_result['p_value']
        if lm_pval <= 0.05:
            recommended_model = "Random Effects"
            decision_reason = ["Breusch-Pagan LM test is significant (p<=0.05), suggesting random effects are present. Random Effects is better than Pooled OLS."]
        else:
            decision_reason = ["Breusch-Pagan LM test is not significant (p>0.05). No evidence of random effects; Pooled OLS is sufficient."]

    # 2. Check Hausman Test (Random Effects vs Fixed Effects) - Only if RE was preferred
    if recommended_model == "Random Effects" and hausman_result and isinstance(hausman_result.get('p_value'), (int, float)):
         hausman_pval = hausman_result['p_value']
         if hausman_pval <= 0.05:
             recommended_model = "Fixed Effects"
             decision_reason.append("However, the Hausman test is significant (p<=0.05), indicating that Random Effects assumptions (uncorrelated error term) are violated. **Fixed Effects** is the consistent estimator.")
         else:
             decision_reason.append("The Hausman test is not significant (p>0.05), meaning the Random Effects assumptions hold. **Random Effects** is more efficient.")

    final_recommendation = f"Recommended Panel Model: **{recommended_model}**.\nReasoning:\n- {' '.join(decision_reason)}"
    return final_recommendation


# --- (4) Model Diagnostics Recommendation Logic (UPDATED) ---
def get_model_diagnostics_recommendations(diagnostics_list=None):
    """
    Analyzes diagnostic tests (Heteroskedasticity, Autocorrelation, etc.) 
    and provides actionable recommendations as structured objects.
    """
    recommendations = []
    if not diagnostics_list: return recommendations

    has_heteroskedasticity = False
    has_autocorrelation = False
    has_normality_issue = False
    has_specification_issue = False

    for test in diagnostics_list:
        if not isinstance(test, dict): continue
        name = test.get('name', '').lower()
        interp = test.get('interpretation', '').lower()
        pval = test.get('p_value') 

        # Determine failure (Issue Detected)
        is_fail = 'fail' in interp or \
                  'reject' in interp or \
                  'present' in interp or \
                  'correlated' in interp or \
                  (pval is not None and isinstance(pval, (int, float)) and pval <= 0.05)
        
        if not is_fail: continue 

        if 'heteroskedasticity' in name or 'white' in name or 'breusch-pagan' in name:
            has_heteroskedasticity = True
        elif 'serial correlation' in name or 'autocorrelation' in name or 'durbin-watson' in name:
            has_autocorrelation = True
        elif 'normality' in name or 'jarque-bera' in name:
            has_normality_issue = True
        elif 'specification' in name or 'reset test' in name:
            has_specification_issue = True

    # --- Generate Recommendations based on issues ---
    
    if has_heteroskedasticity and has_autocorrelation:
        recommendations.append({
            "text": "Fix Suggestion (Autocorrelation & Heteroskedasticity): Your model suffers from both serial correlation and heteroskedasticity. Standard errors are unreliable.",
            "fix_code": "USE_HAC", 
            "fix_text": "Apply Robust HAC (Newey-West) Errors"
        })
    elif has_heteroskedasticity:
        recommendations.append({
            "text": "Fix Suggestion (Heteroskedasticity): Heteroskedasticity detected (variance of errors is not constant). Standard errors are unreliable.",
            "fix_code": "USE_ROBUST_HC3", 
            "fix_text": "Apply Robust Errors (HC3)"
        })
    elif has_autocorrelation:
        recommendations.append({
            "text": "Fix Suggestion (Serial Correlation): Serial correlation detected (errors are correlated with past errors). Standard errors are unreliable.",
            "fix_code": "USE_HAC", 
            "fix_text": "Apply Robust HAC (Newey-West) Errors"
        })

    if has_normality_issue:
        recommendations.append({
            "text": "Note (Normality): Residuals are not normally distributed. While OLS coefficients remain unbiased, inference (t-stats, p-values) might be less reliable in small samples. Consider transforming the dependent variable (e.g., Log transform)."
        })
        
    if has_specification_issue:
        recommendations.append({
            "text": "Note (Specification): Ramsey RESET test failed. The model might be misspecified (e.g., incorrect functional form). Consider adding non-linear terms (squared variables) or checking for omitted variables."
        })

    # --- (NEW) Success Case ---
    # If loop finished and no recommendations were added, it means no major issues were found.
    if not recommendations and diagnostics_list:
        recommendations.append({
            "text": "✅ Model Diagnostics: All automatic checks passed. No significant heteroskedasticity, autocorrelation, or specification errors detected."
        })

    return recommendations


# --- (5) Initial Data Assessment (AI Briefing Runner) ---
def run_initial_data_assessment(df):
    """
    Runs a full suite of pre-estimation tests (ADF, KPSS, VIF) 
    on all numeric columns to generate the AI Briefing.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Dataset is empty.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in the dataset for assessment.")

    stationarity_results = {}
    vif_results = {}
    
    print(f"[AI Briefing] Running stationarity tests on {len(numeric_cols)} columns...")
    for col in numeric_cols:
        adf_level, kpss_level, adf_diff = None, None, None
        try:
            # Run Level Tests
            adf_level = run_adf_test(df, col, test_level='level', regression_type='c')
            kpss_level = run_kpss_test(df, col, test_level='level', regression_type='c')
            
            # If ADF suggests Non-Stationary, try 1st Difference
            adf_p_level = adf_level.get('p_value', 1)
            if adf_p_level > 0.05:
                adf_diff = run_adf_test(df, col, test_level='1st_diff', regression_type='c')
            
            stationarity_results[col] = {'adf_level': adf_level, 'kpss_level': kpss_level, 'adf_diff': adf_diff}
        except Exception as e:
            print(f"[AI Briefing] Error testing stationarity for {col}: {e}")
            stationarity_results[col] = {'error': str(e)}

    print("[AI Briefing] Running VIF test...")
    if len(numeric_cols) >= 2:
        try:
            vif_results = run_multicollinearity_test(df, numeric_cols)
        except Exception as e:
            print(f"[AI Briefing] Error running VIF: {e}")
            vif_results = {'error': str(e)}
    else:
        vif_results = {'interpretation': 'Skipped (need at least 2 numeric variables).'}
    
    print("[AI Briefing] Assessment complete.")
    return {"stationarity": stationarity_results, "vif": vif_results}


# --- (6) Synthesize Briefing (Format Results for Frontend) ---
def synthesize_briefing_recommendations(assessment_results):
    """
    Analyzes the collected results from run_initial_data_assessment 
    and generates a final briefing and model recommendations in ENGLISH.
    """
    stationarity = assessment_results.get('stationarity', {})
    vif = assessment_results.get('vif', {})
    
    stationarity_results = []
    model_recommendations = set()
    vif_payload = {
        "status": "Skipped",
        "high_vif_vars": [],
        "message": "VIF Skipped (requires at least 2 numeric variables)."
    }
    
    has_i0 = False  # Stationary at level
    has_i1 = False  # Stationary at 1st diff
    has_i2_or_error = False # Higher order or error

    # Analyze Stationarity
    for var, tests in stationarity.items():
        if tests.get('error'):
            stationarity_results.append({"variable": var, "result_code": "Error", "interpretation": f"Error during testing: {tests['error']}"})
            has_i2_or_error = True
            continue
        
        adf_p_level = tests.get('adf_level', {}).get('p_value', 1)
        kpss_interp = tests.get('kpss_level', {}).get('interpretation', '').lower()
        
        is_adf_nonstat = adf_p_level > 0.05
        is_kpss_nonstat = 'non-stationary' in kpss_interp
        
        if not is_adf_nonstat and not is_kpss_nonstat:
            stationarity_results.append({"variable": var, "result_code": "I(0)", "interpretation": "Stationary at level (Both tests agree)."})
            has_i0 = True
        elif is_adf_nonstat and is_kpss_nonstat:
            # Both say non-stationary at level, check Diff
            adf_p_diff = tests.get('adf_diff', {}).get('p_value', 1)
            if adf_p_diff <= 0.05:
                stationarity_results.append({"variable": var, "result_code": "I(1)", "interpretation": "Non-stationary, but stationary at 1st Difference (I(1))."})
                has_i1 = True
            else:
                stationarity_results.append({"variable": var, "result_code": "I(2+)", "interpretation": "Warning: Non-stationary even at 1st Diff (Likely I(2) or higher)."})
                has_i2_or_error = True
        else:
            # Conflicting
            stationarity_results.append({"variable": var, "result_code": "Conflicting", "interpretation": "Conflicting results (ADF/KPSS). Manual review needed."})
            # Conservative assumption for mix suggestions
            if is_adf_nonstat: has_i1 = True 
            else: has_i0 = True

    # Analyze VIF
    if not vif.get('error') and vif.get('formatted_results'):
        high_vif_vars = []
        try:
            # Simple parsing of string output (assuming table format)
            lines = vif['formatted_results'].strip().split('\n')
            # Skip header
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 2:
                    vif_value_str = parts[-1] 
                    var_name = " ".join(parts[:-1]) 
                    if float(vif_value_str) > 10.0:
                        high_vif_vars.append(var_name)
            
            if high_vif_vars:
                vif_payload = {"status": "High", "high_vif_vars": high_vif_vars, "message": f"Warning: High multicollinearity (VIF > 10) detected in: {', '.join(high_vif_vars)}."}
            else:
                vif_payload = {"status": "OK", "high_vif_vars": [], "message": "Good: No significant multicollinearity detected (All VIF < 10)."}
        except Exception as e:
            vif_payload = {"status": "Error", "message": f"Error parsing VIF results: {e}"}
    elif vif.get('error'):
        vif_payload = {"status": "Error", "message": f"Error running VIF test: {vif['error']}"}

    # Generate Global Recommendations
    if not has_i0 and not has_i1 and not has_i2_or_error:
         model_recommendations.add("No valid numeric data found for analysis.")
    
    elif has_i2_or_error:
        model_recommendations.add("Warning: Some variables appear to be I(2) or caused calculation errors. Please clean data or transform variables before modeling.")
    
    elif has_i0 and not has_i1:
        model_recommendations.add("All variables appear Stationary I(0). **OLS** (Standard Regression) or **VAR** (at levels) are appropriate.")
    
    elif has_i1 and not has_i0:
        model_recommendations.add("All variables appear Non-Stationary I(1). Check for Cointegration (Johansen Test).")
        model_recommendations.add("If Cointegrated -> Use **VECM**.")
        model_recommendations.add("If Not Cointegrated -> Use **VAR on 1st Differences**.")
    
    elif has_i0 and has_i1:
        model_recommendations.add("Mixed Order of Integration detected (some I(0), some I(1)).")
        model_recommendations.add("**ARDL (Autoregressive Distributed Lag)** bounds test is the most robust method for mixed I(0)/I(1) variables.")
        model_recommendations.add("Alternatively, difference the I(1) variables to make them stationary before using OLS.")

    if vif_payload["status"] == "High":
        model_recommendations.add("Address VIF > 10 by removing one of the correlated variables, combining them, or using Ridge Regression.")

    final_briefing = {
        "title": "AI Initial Analysis Briefing",
        "stationarity": stationarity_results,
        "vif": vif_payload,
        "recommendations": list(model_recommendations)
    }
    
    return final_briefing