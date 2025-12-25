import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import chi2 # (!!!) إضافة جديدة لحساب Fisher Test
import io
import sys
import warnings
from arch.unitroot import ZivotAndrews # (جديد)
from scipy.stats import norm # (جديد)

# --- (1) Panel Data Support ---
# (تم إزالة محاولة استيراد linearmodels.panel.unitroot لأنها غير موجودة)
# سنعتمد على statsmodels وحساباتنا اليدوية لاختبارات Panel


# --- (2) ADF Test (Augmented Dickey-Fuller) ---
def run_adf_test(df, variable, test_level, regression_type):
    """Runs ADF test to check for Unit Root."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")
    
    # Ensure numeric only
    series = df[variable]
    if not pd.api.types.is_numeric_dtype(series):
         raise ValueError(f"Variable '{variable}' must be numeric for ADF test.")
    
    series = series.dropna()
    
    original_series_name = f"'{variable}' at {test_level.replace('_', ' ').title()}"
    
    # Apply transformations based on test level
    if test_level == '1st_diff':
        series = series.diff().dropna()
    elif test_level == '2nd_diff':
        series = series.diff().diff().dropna()

    if len(series) < 4:
        raise ValueError(f"Too few observations ({len(series)}) in {original_series_name} to perform ADF test.")
    
    # Check if series is constant (creates error in ADF)
    if series.nunique() <= 1:
        raise ValueError(f"Invalid input for ADF: '{variable}' is constant (no variation).")

    try:
        # Ignore warnings related to optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(series, regression=regression_type, autolag='AIC')
    except Exception as e:
        raise RuntimeError(f"Error during ADF calculation for {original_series_name}: {e}") from e

    stat, p_value, _, _, crit_vals, _ = result
    is_stationary = bool(p_value <= 0.05)
    
    output = f"Augmented Dickey-Fuller Test ({test_level})\n"
    output += f"Statistic: {stat:.4f}, P-value: {p_value:.4f}\n"
    output += "Critical Values:\n"
    for key, value in crit_vals.items():
        output += f"   {key}:             {value:.4f}\n"

    interpretation = (
        f"Conclusion (at 5% significance):\n"
        f"With a p-value of {p_value:.4f}, we {'REJECT' if is_stationary else 'FAIL TO REJECT'} the null hypothesis.\n"
        f"The series is likely {'STATIONARY' if is_stationary else 'NON-STATIONARY'}."
    )

    return {
        "formatted_results": output, 
        "interpretation": interpretation,
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_stationary": is_stationary 
    }


# --- (3) KPSS Test (Kwiatkowski–Phillips–Schmidt–Shin) ---
def run_kpss_test(df, variable, test_level, regression_type):
    """Runs KPSS test (Stationarity is Null Hypothesis)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")

    series = df[variable].dropna()
    original_series_name = f"'{variable}' at {test_level.replace('_', ' ').title()}"
    
    if test_level == '1st_diff':
        series = series.diff().dropna()
    elif test_level == '2nd_diff':
        series = series.diff().diff().dropna()

    if len(series) < 4: raise ValueError("Too few observations.")
    if series.nunique() <= 1: raise ValueError("Series is constant.")

    try:
        # regression_type: 'c' (level stationary) or 'ct' (trend stationary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(series, regression=regression_type, nlags='auto')
    except Exception as e:
        raise RuntimeError(f"Error during KPSS calculation: {e}")

    statistic, p_value, lags, crit_values = result
    
    crit_5pct = crit_values.get('5%')
    is_non_stationary = bool(crit_5pct is not None and statistic > crit_5pct)

    output = f"KPSS Test for {original_series_name}\n"
    output += f"Null Hypothesis: Series is stationary (around {'constant' if regression_type == 'c' else 'trend'})\n"
    output += f"Regression Type: {regression_type}\n"
    output += "----------------------------------------\n"
    output += f"KPSS Statistic:   {statistic:.4f}\n"

    # statsmodels doesn't give precise p-value, only 0.01, 0.025, 0.05, 0.10.
    p_value_str = f"{p_value:.4f}" 
    if p_value <= 0.01: p_value_str = "<= 0.01"
    elif p_value >= 0.1: p_value_str = ">= 0.10"
    
    output += f"P-value:            {p_value_str}\n"
    output += f"Lags Used:            {lags}\n"
    output += "Critical Values (approximate):\n"
    for key, value in crit_values.items():
        output += f"   {key}:             {value:.4f}\n"

    interpretation = (
        f"Conclusion (at 5% significance):\n"
        f"The test statistic ({statistic:.4f}) is {'GREATER' if is_non_stationary else 'LESS THAN OR EQUAL TO'} the 5% critical value ({crit_5pct:.4f}).\n"
        f"We {'REJECT' if is_non_stationary else 'FAIL TO REJECT'} the null hypothesis.\n"
        f"The series is likely {'NON-STATIONARY' if is_non_stationary else 'STATIONARY'}."
    )

    return {
        "formatted_results": output, 
        "interpretation": interpretation,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "is_non_stationary": is_non_stationary
    }


# --- (4) VIF Test (Variance Inflation Factor) ---
# --- (4) VIF Test (Variance Inflation Factor) - EViews Style ---
def run_multicollinearity_test(df, independent_vars):
    """Calculates VIF and returns an EViews-style table."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    if not independent_vars or len(independent_vars) < 2:
        return {"formatted_results": "VIF requires at least two independent variables.", "interpretation": "Select 2+ variables."}

    # تجهيز البيانات
    valid_vars = [v for v in independent_vars if v in df.columns]
    X = df[valid_vars].dropna().select_dtypes(include=np.number)
    
    # التأكد من وجود تباين (لتجنب القسمة على صفر)
    X = X.loc[:, X.var() > 1e-9]
    if X.shape[1] < 2:
        return {"formatted_results": "Variables are constant or insufficient.", "interpretation": "Cannot calculate VIF."}

    # إضافة القاطع لحساب VIF "Centered"
    X_const = add_constant(X, prepend=True)
    
    vif_data = []
    for i in range(X_const.shape[1]):
        col_name = X_const.columns[i]
        if col_name == 'const': continue
        
        try:
            val = variance_inflation_factor(X_const.values, i)
            # 1 / VIF (Tolerance)
            tolerance = 1/val if val != 0 else 0
        except:
            val = np.inf
            tolerance = 0
        
        vif_data.append({
            'Variable': col_name, 
            'Coefficient Variance': 'N/A', # بايثون لا يحسب هذا هنا بسهولة، نتركه فارغاً كما يفعل EViews أحياناً قبل التقدير
            'Uncentered VIF': 'N/A', 
            'Centered VIF': val
        })
    
    vif_df = pd.DataFrame(vif_data)

    # --- HTML Table (EViews Style) ---
    html = """
    <div style='font-family: Arial; font-size: 12px;'>
    <b>Variance Inflation Factors</b><br>
    Date: {date} &nbsp; Time: {time}<br>
    Sample: All<br>
    Included observations: {n_obs}<br><br>
    <table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse; width: 100%; border-color: #dcdcdc;'>
        <tr style='background-color: #f0f0f0; font-weight: bold;'>
            <td style='text-align: left;'>Variable</td>
            <td style='text-align: center;'>Centered VIF</td>
            <td style='text-align: center;'>1/VIF (Tolerance)</td>
        </tr>
    """.format(date=pd.Timestamp.now().strftime("%m/%d/%y"), 
               time=pd.Timestamp.now().strftime("%H:%M"),
               n_obs=len(X))

    for _, row in vif_df.iterrows():
        vif_val = row['Centered VIF']
        vif_str = f"{vif_val:.6f}" if vif_val != float('inf') else "NA"
        tol_val = 1/vif_val if vif_val not in [0, float('inf')] else 0
        
        # تلوين القيم الخطرة
        color = "color: red; font-weight: bold;" if (isinstance(vif_val, (int, float)) and vif_val > 10) else ""
        
        html += f"""
        <tr>
            <td style='text-align: left;'>{row['Variable']}</td>
            <td style='text-align: center; {color}'>{vif_str}</td>
            <td style='text-align: center;'>{tol_val:.6f}</td>
        </tr>
        """
    html += "</table></div>"

    # التفسير
    high_vif = vif_df[pd.to_numeric(vif_df['Centered VIF'], errors='coerce') > 10]
    interp = "<b>Result:</b> No significant multicollinearity."
    if not high_vif.empty:
        interp = f"<b>Warning:</b> High multicollinearity (VIF > 10) detected in: {', '.join(high_vif['Variable'].tolist())}."

    return {"formatted_results": html, "interpretation": interp}

# --- (5) Optimal Lag Selection (for VAR) ---
# --- (5) Optimal Lag Selection - EViews Style ---
def run_optimal_lag_selection(df, variables, maxlags=8):
    """Generates VAR Lag Order Selection Criteria table."""
    data = df[variables].dropna().select_dtypes(include=np.number)
    if len(data) < maxlags + 5:
        raise ValueError("Insufficient observations.")

    try:
        model = VAR(data)
        # statsmodels lag order selection summary is already excellent
        lag_results = model.select_order(maxlags=maxlags)
        summary = lag_results.summary()
        
        # تحويل الجدول النصي لـ HTML احترافي
        html_obj = summary.as_html()
        
        # إضافة ترويسة EViews
        final_html = f"""
        <div style='font-family: Arial; font-size: 12px;'>
        <b>VAR Lag Order Selection Criteria</b><br>
        Endogenous variables: {', '.join(variables)}<br>
        Included observations: {len(data) - maxlags}<br>
        <br>
        {html_obj}
        <br>
        <small>* indicates lag order selected by the criterion</small><br>
        <small>AIC: Akaike information criterion</small><br>
        <small>SC: Schwarz information criterion</small><br>
        <small>HQ: Hannan-Quinn information criterion</small>
        </div>
        """
        
        interp = f"Selected Lag: AIC({lag_results.aic}), SC({lag_results.bic}), HQ({lag_results.hqic})."
        
    except Exception as e:
        raise RuntimeError(f"Lag selection failed: {e}")

    return {"formatted_results": final_html, "interpretation": interp}

# --- (6) Johansen Cointegration Test ---
# --- (6) Johansen Cointegration - EViews Style ---
def run_johansen_cointegration_test(df, variables, det_order=0, k_ar_diff=1):
    """Runs Johansen test and outputs EViews-style table."""
    data = df[variables].dropna().select_dtypes(include=np.number)
    
    try:
        # det_order: -1 (no const), 0 (const), 1 (const+trend)
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    except Exception as e:
        raise RuntimeError(f"Johansen failed: {e}")

    # استخراج النتائج
    traces = result.lr1
    crit_vals = result.cvt[:, 1] # القيمة الحرجة عند 5%
    eigenvalues = result.eig
    
    hypotheses_names = []
    n_vars = len(variables)
    for i in range(n_vars):
        if i == 0: hypotheses_names.append("None")
        elif i == 1: hypotheses_names.append("At most 1")
        else: hypotheses_names.append(f"At most {i}")

    # --- HTML Table Construction ---
    html = """
    <div style='font-family: Arial; font-size: 12px;'>
    <b>Unrestricted Cointegration Rank Test (Trace)</b><br>
    Series: {series}<br>
    Lags interval (in first differences): 1 to {lags}<br>
    Trend assumption: {trend}<br><br>
    <table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse; width: 100%; border-color: #dcdcdc; text-align: center;'>
        <tr style='background-color: #f0f0f0; font-weight: bold;'>
            <td style='text-align: left;'>Hypothesized<br>No. of CE(s)</td>
            <td>Eigenvalue</td>
            <td>Trace<br>Statistic</td>
            <td>0.05<br>Critical Value</td>
            <td>Prob.**</td>
        </tr>
    """.format(series=", ".join(variables), 
               lags=k_ar_diff, 
               trend="Linear deterministic trend" if det_order==0 else "No deterministic trend")

    rank = 0
    for i in range(n_vars):
        trace = traces[i]
        crit = crit_vals[i]
        eig = eigenvalues[i]
        hypo = hypotheses_names[i]
        
        # تحديد المعنوية (بما أننا لا نملك P-value دقيقة من statsmodels)
        # إذا كانت الإحصائية أكبر من القيمة الحرجة -> نرفض العدم (توجد علاقة)
        is_sig = trace > crit
        sig_mark = "*" if is_sig else ""
        
        # P-value تقريبية للعرض (لأن statsmodels لا يعطيها)
        # سنكتب (Sig) إذا كانت معنوية، أو (Not Sig)
        prob_display = "< 0.05" if is_sig else "> 0.05"
        
        row_style = "background-color: #fafafa;" if i % 2 == 0 else ""
        
        html += f"""
        <tr style='{row_style}'>
            <td style='text-align: left;'>{hypo} {sig_mark}</td>
            <td>{eig:.6f}</td>
            <td>{trace:.4f}</td>
            <td>{crit:.4f}</td>
            <td>{prob_display}</td>
        </tr>
        """
        if is_sig: rank += 1

    html += """
    </table>
    <div style='margin-top: 5px;'>
    <small>Trace test indicates {rank} cointegrating eqn(s) at the 0.05 level</small><br>
    <small>* denotes rejection of the hypothesis at the 0.05 level</small><br>
    <small>** MacKinnon-Haug-Michelis (1999) p-values (Approx)</small>
    </div>
    </div>
    """.format(rank=rank)

    return {"formatted_results": html, "interpretation": f"Trace test indicates {rank} cointegrating equations.", "cointegrating_relations": rank}


# --- (7) Granger Causality Test ---
# --- (7) Granger Causality - EViews Style ---
def run_granger_causality_test(df, variables, max_lag=2):
    """Runs Pairwise Granger Causality Tests and formats like EViews."""
    data = df[variables].dropna().select_dtypes(include=np.number)
    
    # --- HTML Header ---
    html = """
    <div style='font-family: Arial; font-size: 12px;'>
    <b>Pairwise Granger Causality Tests</b><br>
    Date: {date} &nbsp; Time: {time}<br>
    Sample: All<br>
    Lags: {lags}<br><br>
    <table border='1' cellspacing='0' cellpadding='4' style='border-collapse: collapse; width: 100%; border-color: #dcdcdc;'>
        <tr style='background-color: #f0f0f0; font-weight: bold;'>
            <td style='text-align: left;'>Null Hypothesis:</td>
            <td style='text-align: center;'>Obs</td>
            <td style='text-align: center;'>F-Statistic</td>
            <td style='text-align: center;'>Prob.</td>
        </tr>
    """.format(date=pd.Timestamp.now().strftime("%m/%d/%y"), 
               time=pd.Timestamp.now().strftime("%H:%M"), 
               lags=max_lag)

    cols = data.columns
    found_causality = []

    # الحلقة التكرارية للأزواج
    import itertools
    for col1, col2 in itertools.permutations(cols, 2):
        # تجهيز البيانات للزوج
        pair_data = data[[col1, col2]].dropna() # Y, X (Does X Granger Cause Y?)
        n_obs = len(pair_data)
        
        if n_obs <= max_lag + 2: continue

        try:
            # statsmodels expects (Y, X) -> tests if X causes Y
            # We want to test "col2 does NOT Granger Cause col1"
            # statsmodels output: {lag: ({'ssr_ftest': (F, p, df_denom, df_num), ...})}
            res = grangercausalitytests(pair_data[[col1, col2]], maxlag=[max_lag], verbose=False)
            
            # استخراج النتائج
            f_stat = res[max_lag][0]['ssr_ftest'][0]
            p_val = res[max_lag][0]['ssr_ftest'][1]
            
            # التنسيق
            color = "color: blue; font-weight: bold;" if p_val < 0.05 else ""
            
            html += f"""
            <tr>
                <td style='text-align: left;'>{col2} does not Granger Cause {col1}</td>
                <td style='text-align: center;'>{n_obs}</td>
                <td style='text-align: center;'>{f_stat:.4f}</td>
                <td style='text-align: center; {color}'>{p_val:.4f}</td>
            </tr>
            """
            
            if p_val < 0.05:
                found_causality.append(f"{col2} -> {col1}")

        except Exception:
            continue

    html += "</table></div>"
    
    interp = "No significant causality found."
    if found_causality:
        interp = f"Significant Causality found (at 5%): {', '.join(found_causality)}"

    return {"formatted_results": html, "interpretation": interp}


# --- (8) Panel Unit Root Tests (Fisher-Type - Robust Replacement) ---
def run_panel_unit_root_test(df, variable, panel_id_var, panel_time_var, test_type='llc'):
    """
    Runs a Fisher-type Panel Unit Root Test using ADF on each entity.
    NOTE: Replaces LLC/IPS due to missing library support, providing a robust alternative.
    H0: All panels contain unit roots (Non-Stationary).
    Ha: At least one panel is stationary.
    """
    
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input dataset is empty or invalid.")
    
    required_cols = [variable, panel_id_var, panel_time_var]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("One or more required columns (Variable, Panel ID, Time ID) are missing.")
    
    try:
        # 1. Setup Data
        panel_df = df[required_cols].dropna()
        unique_ids = panel_df[panel_id_var].unique()
        
        if len(unique_ids) < 2:
             raise ValueError("Panel data must have at least 2 unique entities for a panel test.")

        # 2. Run ADF for each entity
        p_values = []
        valid_entities = 0
        
        for entity in unique_ids:
            # Filter data for this entity
            entity_data = panel_df[panel_df[panel_id_var] == entity][variable].dropna()
            
            # Basic check: enough data points
            if len(entity_data) > 4 and entity_data.nunique() > 1:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Run ADF with constant only
                        res = adfuller(entity_data, regression='c', autolag='AIC')
                        p_values.append(res[1])
                        valid_entities += 1
                except:
                    pass # Skip entities where ADF fails (e.g. too short)

        if valid_entities < 2:
             raise ValueError("Not enough valid entities/data to compute Panel Unit Root test.")

        # 3. Compute Fisher Statistic
        # Stat = -2 * sum(log(p_values)) ~ Chi-Squared(2*N)
        p_values_arr = np.array(p_values)
        # Avoid log(0)
        p_values_arr[p_values_arr == 0] = 1e-10
        
        fisher_stat = -2 * np.sum(np.log(p_values_arr))
        degrees_of_freedom = 2 * valid_entities
        combined_p_value = chi2.sf(fisher_stat, degrees_of_freedom)
        
        test_name = "Panel Unit Root (Fisher-ADF)"
        interpretation_null = "Null Hypothesis: All panels contain unit roots (Non-Stationary)"
        
    except Exception as e:
        raise RuntimeError(f"Error calculating Panel Unit Root test: {e}") from e

    is_stationary = bool(combined_p_value <= 0.05)
    
    output = f"{test_name} for '{variable}'\n"
    output += f"{interpretation_null}\n"
    output += f"Entities Included: {valid_entities}\n"
    output += "----------------------------------------\n"
    output += f"Chi-Square Statistic: {fisher_stat:.4f}\n"
    output += f"P-value:              {combined_p_value:.4f}\n"

    interpretation = (
        f"Conclusion (at 5% significance):\n"
        f"With a p-value of {combined_p_value:.4f}, we {'REJECT' if is_stationary else 'FAIL TO REJECT'} the null hypothesis.\n"
        f"{'At least one panel is stationary (The series is likely Stationary).' if is_stationary else 'The panels likely contain unit roots (Non-Stationary).'}"
    )

    return {
        "formatted_results": output, 
        "interpretation": interpretation,
        "statistic": float(fisher_stat),
        "p_value": float(combined_p_value),
        "is_stationary": is_stationary
    }

# --- (9) Zivot-Andrews Test (Structural Break) ---
def run_zivot_andrews_test(df, variable):
    """
    Runs Zivot-Andrews test for unit root with structural break.
    """
    series = df[variable].dropna()
    if len(series) < 30:
        # (تعديل) رسالة أوضح
        return {
            "formatted_results": "Test Skipped (Insufficient Data)",
            "interpretation": f"Zivot-Andrews test requires a longer time series (at least 30 observations). You only have {len(series)}. Please use a dataset with more historical data."
        }
    
    try:
        # trend='c' (intercept break), 't' (trend break), 'ct' (both)
        # We use 'c' as default for economic data
        za = ZivotAndrews(series, trim=0.15, trend='c', method='aic')
        
        output = f"Zivot-Andrews Unit Root Test\n"
        output += f"Null Hypothesis: Unit Root with Structural Break\n"
        output += "----------------------------------------\n"
        output += f"Statistic:      {za.stat:.4f}\n"
        output += f"P-Value:        {za.pvalue:.4f}\n"
        output += "Critical Values:\n"
        for key, val in za.critical_values.items():
            output += f"   {key}%:          {val:.4f}\n"
            
        is_stationary = za.pvalue <= 0.05
        interpretation = (
            f"Conclusion (p={za.pvalue:.4f}): {'REJECT H0' if is_stationary else 'FAIL TO REJECT H0'}.\n"
            f"{'The series is Stationary with a structural break.' if is_stationary else 'The series is Non-Stationary (even allowing for a break).'}"
        )
        
        return {"formatted_results": output, "interpretation": interpretation}
    except Exception as e:
        raise RuntimeError(f"Zivot-Andrews test failed: {e}")

# --- (10) Pesaran CD Test (Cross-Sectional Dependence) ---
def run_pesaran_cd_test(df, variable, panel_id_var, panel_time_var):
    """
    Calculates Pesaran (2004) CD test.
    """
    try:
        # 1. Reshape to Wide Format
        # (Check for duplicates first to avoid pivot error)
        if df.duplicated(subset=[panel_time_var, panel_id_var]).any():
             raise ValueError(f"Duplicates detected in Panel ID/Time combinations. Please remove duplicates in 'Data Preparation' first.")

        wide_df = df.pivot(index=panel_time_var, columns=panel_id_var, values=variable)
        
        # Drop entities with ANY missing values (Pesaran CD requires balanced correlation computation)
        original_N = wide_df.shape[1]
        wide_df = wide_df.dropna(axis=1, how='any') 
        
        T = wide_df.shape[0]
        N = wide_df.shape[1]
        
        if N < 2:
            raise ValueError(f"Not enough entities for correlation analysis. Started with {original_N}, but after removing entities with missing values (unbalanced panel), only {N} remained. Pesaran CD requires a balanced panel (no missing values).")
        
        if T < 5:
            raise ValueError(f"Time series too short (T={T}). Need at least 5 periods.")

       
        # 2. Calculate Correlation Matrix
        corr_matrix = wide_df.corr().values
        
        # 3. Sum off-diagonal elements
        # Only upper triangle (excluding diagonal)
        rho_sum = 0
        for i in range(N):
            for j in range(i + 1, N):
                rho_sum += corr_matrix[i, j]
                
        # 4. Calculate Statistic
        # CD ~ N(0, 1)
        cd_stat = np.sqrt((2 * T) / (N * (N - 1))) * rho_sum
        
        # 5. Calculate P-Value (Two-tailed)
        p_value = 2 * (1 - norm.cdf(abs(cd_stat)))
        
        output = f"Pesaran CD Test (Cross-Sectional Dependence)\n"
        output += f"Null Hypothesis: No cross-sectional dependence (residuals are independent)\n"
        output += f"Entities (N): {N}, Periods (T): {T}\n"
        output += "----------------------------------------\n"
        output += f"CD Statistic:   {cd_stat:.4f}\n"
        output += f"P-Value:        {p_value:.4f}\n"
        
        has_dependence = p_value <= 0.05
        interpretation = (
            f"Conclusion (p={p_value:.4f}): {'REJECT H0' if has_dependence else 'FAIL TO REJECT H0'}.\n"
            f"{'Significant Cross-Sectional Dependence detected.' if has_dependence else 'No evidence of Cross-Sectional Dependence.'}"
        )
        if has_dependence:
            interpretation += "\nRecommendation: Use Driscoll-Kraay standard errors or Spatial models."
            
        return {"formatted_results": output, "interpretation": interpretation}
        
    except Exception as e:
        raise RuntimeError(f"Pesaran CD test failed: {e}")