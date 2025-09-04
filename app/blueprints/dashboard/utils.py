# app/blueprints/dashboard/utils.py
import pandas as pd

def calculate_summary_statistics(df):
    """
    Calculates key statistics for each numeric column in the DataFrame.
    """
    summary_stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        # Basic descriptive stats
        desc = df[col].describe()
        
        # Count missing values
        missing_values = int(df[col].isnull().sum())
        
        # Detect outliers using the IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers_found = int(outlier_condition.sum())

        summary_stats[col] = {
            'count': int(desc.get('count', 0)),
            'mean': round(desc.get('mean', 0), 4),
            'std_dev': round(desc.get('std', 0), 4),
            'min': round(desc.get('min', 0), 4),
            'max': round(desc.get('max', 0), 4),
            'median': round(df[col].median(), 4),
            'missing_values': missing_values,
            'outliers_found': outliers_found
        }
        
    return summary_stats