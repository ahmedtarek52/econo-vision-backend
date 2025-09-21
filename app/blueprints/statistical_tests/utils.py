# app/blueprints/statistical_tests/utils.py
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import numpy as np

def run_adf_test(df, variable):
    """
    Performs the Augmented Dickey-Fuller test and formats the output.
    """
    series = df[variable]
    result = adfuller(series, autolag='AIC')
    
    output = f"Augmented Dickey-Fuller Unit Root Test on {variable}\n"
    output += f"Null Hypothesis: {variable} has a unit root\n"
    output += "--------------------------------------------------------------\n"
    
    output += f"ADF Statistic:              {result[0]:.4f}\n"
    output += f"p-value:                    {result[1]:.4f}\n"
    output += f"Lags Used:                  {result[2]}\n"
    output += f"Number of Observations:     {result[3]}\n\n"
    
    output += "Critical Values:\n"
    for key, value in result[4].items():
        output += f"    {key}:                  {value:.4f}\n"
    
    output += "--------------------------------------------------------------\n"
    
    # Interpretation
    interpretation = ""
    if result[1] <= 0.05:
        interpretation = (
            f"Conclusion: The p-value ({result[1]:.4f}) is less than 0.05. "
            "We reject the null hypothesis.\n"
            f"The '{variable}' series is likely **stationary**."
        )
    else:
        interpretation = (
            f"Conclusion: The p-value ({result[1]:.4f}) is greater than 0.05. "
            "We fail to reject the null hypothesis.\n"
            f"The '{variable}' series is likely **non-stationary**.\n"
            "Recommendation: Consider differencing the series."
        )
        
    return {"formatted_results": output, "interpretation": interpretation}


def run_granger_causality_test(df, max_lags=2):
    """
    Performs pairwise Granger Causality tests on all numeric columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {
            "formatted_results": "Granger Causality test requires at least two numeric variables.",
            "interpretation": "Please provide more data."
        }
        
    output = f"Pairwise Granger Causality Tests\n"
    output += f"Sample: 1 to {len(df)}\n"
    output += f"Lags: {max_lags}\n"
    output += "=================================================================\n"
    output += "Null Hypothesis:                            F-Statistic    Prob.\n"
    output += "-----------------------------------------------------------------\n"
    
    all_results = []
    
    from itertools import permutations
    for col1, col2 in permutations(numeric_cols, 2):
        test_data = df[[col2, col1]]
        results = grangercausalitytests(test_data, [max_lags], verbose=False)
        
        # Extract results for the specified lag
        f_statistic = results[max_lags][0]['ssr_ftest'][0]
        p_value = results[max_lags][0]['ssr_ftest'][1]
        
        line = f"{col1} does not Granger Cause {col2}".ljust(40)
        line += f"{f_statistic:.4f}".rjust(12)
        line += f"{p_value:.4f}".rjust(9)
        output += line + "\n"
        all_results.append({'cause': col1, 'effect': col2, 'p_value': p_value})
        
    output += "=================================================================\n"
    
    # Interpretation
    significant_relations = [
        f"'{res['cause']}' may Granger-cause '{res['effect']}' (p={res['p_value']:.4f})"
        for res in all_results if res['p_value'] <= 0.05
    ]
    
    if significant_relations:
        interpretation = "Significant relationships found (at 5% level):\n- " + "\n- ".join(significant_relations)
    else:
        interpretation = "No significant Granger causality relationships were found between any pair of variables at the 5% significance level."

    return {"formatted_results": output, "interpretation": interpretation}