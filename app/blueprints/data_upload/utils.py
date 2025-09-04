import pandas as pd

def get_file_extension(filename):
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def check_data_quality(df):
    issues = []
    if df.empty:
        issues.append("The uploaded file is empty.")
        return {"is_ok": False, "issues": issues}
    
    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        issues.append(f"Found {missing_values_count} missing value(s) in the dataset.")

    duplicate_rows_count = df.duplicated().sum()
    if duplicate_rows_count > 0:
        issues.append(f"Found {duplicate_rows_count} duplicate row(s).")
    
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        issues.append(f"The following columns are completely empty: {', '.join(empty_cols)}")
    
    if any('Unnamed:' in str(col) for col in df.columns):
        issues.append("Some columns may have missing headers (e.g., 'Unnamed: 0').")

    return {"is_ok": len(issues) == 0, "issues": issues}

def suggest_variable_types(df):
    suggestions = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 15:
                suggestions[col] = 'Categorical'
            else:
                suggestions[col] = 'Quantitative'
        else:
            suggestions[col] = 'Categorical'
    return suggestions