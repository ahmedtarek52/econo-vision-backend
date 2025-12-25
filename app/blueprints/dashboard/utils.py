# app/blueprints/dashboard/utils.py
import pandas as pd
import numpy as np # <-- إضافة numpy
from statsmodels.tsa.stattools import acf, pacf # <-- إضافة ACF/PACF
from statsmodels.tsa.stattools import acovf # Needed for confidence intervals

def calculate_summary_statistics(df):
    """
    Calculates key statistics for each numeric column in the DataFrame.
    """
    # --- الكود بتاعك هنا زي ما هو بدون تغيير ---
    summary_stats = {}
    # Check if df is empty or not a DataFrame
    if not isinstance(df, pd.DataFrame) or df.empty:
        return summary_stats # Return empty if no valid data

    numeric_cols = df.select_dtypes(include=np.number).columns # Use numpy.number for broader numeric check

    for col in numeric_cols:
        if df[col].isnull().all(): # Skip columns that are all NaN
            summary_stats[col] = { 'count': 0, 'mean': np.nan, 'std_dev': np.nan, 'min': np.nan, 'max': np.nan, 'median': np.nan, 'missing_values': len(df), 'outliers_found': 0 }
            continue

        desc = df[col].describe()
        missing_values = int(df[col].isnull().sum())

        # Detect outliers using the IQR method (handle potential NaNs in quantiles)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        outliers_found = 0 # Default to 0
        if pd.notna(Q1) and pd.notna(Q3): # Only calculate if quantiles are valid
            IQR = Q3 - Q1
            # Avoid IQR=0 issues or division by zero if needed later
            if IQR > 1e-9: # Check if IQR is reasonably large
                 lower_bound = Q1 - 1.5 * IQR
                 upper_bound = Q3 + 1.5 * IQR
                 outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
                 outliers_found = int(outlier_condition.sum())
            # else: # Handle case where IQR is zero or very small (e.g., constant data)
            #     # Outliers might be defined differently here, e.g., anything not equal to the constant value
            #     pass

        summary_stats[col] = {
            'count': int(desc.get('count', 0)),
            'mean': round(desc.get('mean', np.nan), 4),
            'std_dev': round(desc.get('std', np.nan), 4),
            'min': round(desc.get('min', np.nan), 4),
            'max': round(desc.get('max', np.nan), 4),
            'median': round(df[col].median(skipna=True), 4), # Ensure median calculation skips NaNs
            'missing_values': missing_values,
            'outliers_found': outliers_found
        }

    return summary_stats

# --- (إضافة جديدة) ---
def calculate_correlation_matrix(df):
    """
    Calculates the correlation matrix for numeric columns.
    Returns a dictionary suitable for JSON serialization.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    numeric_cols = df.select_dtypes(include=np.number)
    if numeric_cols.empty:
        return {}
    # Handle columns with zero variance which cause NaN in correlation
    numeric_cols = numeric_cols.loc[:, numeric_cols.std(skipna=True) > 1e-9]
    if numeric_cols.empty:
        return {}

    correlation = numeric_cols.corr().round(4)
    # Convert NaN to None for JSON compatibility
    correlation = correlation.where(pd.notnull(correlation), None)
    return correlation.to_dict() # Return as nested dictionary
# --- (نهاية الإضافة) ---


# --- (إضافة جديدة) ---
def calculate_histogram_data(df, variable, bins='auto'):
    """
    Calculates histogram data (bin edges and counts) for a specific variable.
    'bins' can be an integer or 'auto'.
    Returns a dictionary { 'bins': [edges...], 'counts': [counts...] }.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or variable not in df.columns or not pd.api.types.is_numeric_dtype(df[variable]):
        raise ValueError(f"Invalid input or variable '{variable}' not found/numeric for histogram.")

    series = df[variable].dropna()
    if series.empty:
         return {'bins': [], 'counts': []} # Return empty if no valid data after dropping NaNs

    # Use numpy.histogram for calculation
    # 'auto' tries to find a good number of bins (e.g., Freedman-Diaconis or Sturges rule)
    counts, bin_edges = np.histogram(series, bins=bins)

    return {
        'bins': bin_edges.tolist(), # Convert numpy array to list for JSON
        'counts': counts.tolist()   # Convert numpy array to list for JSON
    }
# --- (نهاية الإضافة) ---


# --- (إضافة جديدة) ---
def calculate_acf_pacf(df, variable, lags=20, alpha=0.05):
    """
    Calculates ACF and PACF values along with confidence intervals.
    Returns a dictionary containing acf_data and pacf_data.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or variable not in df.columns or not pd.api.types.is_numeric_dtype(df[variable]):
         raise ValueError(f"Invalid input or variable '{variable}' not found/numeric for ACF/PACF.")

    series = df[variable].dropna()
    if series.empty:
        raise ValueError(f"Variable '{variable}' contains no valid numeric data after dropping NaNs.")
    # Ensure sufficient data points for calculation
    n_obs = len(series)
    if n_obs <= lags:
        lags = max(1, n_obs - 1) # Adjust lags if too few observations
        print(f"Warning: Too few observations for requested lags. Reduced lags to {lags}.")
    if n_obs < 4: # Need at least a few points for variance calculation
         raise ValueError(f"Variable '{variable}' has too few non-NaN observations ({n_obs}) for ACF/PACF.")


    try:
        # Calculate ACF with confidence intervals
        # nlags includes lag 0, so request lags+1 if nlags excludes 0? Check docs.
        # statsmodels acf includes lag 0 by default.
        acf_values, confint_acf = acf(series, nlags=lags, alpha=alpha, fft=False) # fft=False for accurate CI with NaNs potentially handled earlier

        # Calculate PACF with confidence intervals
        pacf_values, confint_pacf = pacf(series, nlags=lags, alpha=alpha, method='ols') # OLS method is common

         # Confidence intervals from statsmodels often need reshaping or calculation
         # acf returns ci like [[lower0, upper0], [lower1, upper1], ...]
         # pacf also returns similar structure

        # Prepare lags array (excluding lag 0 for plotting)
        lag_numbers = list(range(1, lags + 1))

        # Format confidence intervals: Convert NaN to None for JSON
        ci_acf_formatted = [[round(l, 4) if pd.notna(l) else None, round(u, 4) if pd.notna(u) else None] for l, u in confint_acf[1:]] # Skip lag 0 CI
        ci_pacf_formatted = [[round(l, 4) if pd.notna(l) else None, round(u, 4) if pd.notna(u) else None] for l, u in confint_pacf[1:]] # Skip lag 0 CI


        acf_result = {
            'lags': lag_numbers,
            'acf': [round(x, 4) for x in acf_values[1:]], # Skip lag 0 value (which is always 1)
            'confint': ci_acf_formatted
        }
        pacf_result = {
            'lags': lag_numbers,
            'pacf': [round(x, 4) for x in pacf_values[1:]], # Skip lag 0 value
             'confint': ci_pacf_formatted
        }

    except Exception as e:
         # Catch potential errors during calculation (e.g., constant series)
         print(f"Error calculating ACF/PACF for {variable}: {e}")
         raise ValueError(f"Could not calculate ACF/PACF for '{variable}'. Check data variance.") from e


    return {
        'acf_data': acf_result,
        'pacf_data': pacf_result
    }
# --- (نهاية الإضافة) ---

# Pair Plot function would be more complex, involving plotting libraries (like seaborn/matplotlib)
# and saving the plot to a file or returning base64 data. Let's add it later.
# def generate_pair_plot_data(df):
#     # ... implementation using seaborn.pairplot ...
#     # ... save to BytesIO buffer ...
#     # ... convert buffer to base64 string ...
#     pass