# app/blueprints/data_preparation/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def remove_missing_values(df):
    """Removes all rows with any NaN values."""
    return df.dropna().reset_index(drop=True)

def impute_with_average(df):
    """
    NEW ALGORITHM: Imputes missing values for all numeric columns using their mean.
    This function finds all columns that are numeric, and for each one,
    it fills any missing values with the average (mean) of that specific column.
    """
    # Create a copy to avoid modifying the original DataFrame slice
    df_imputed = df.copy()
    
    # Select only the columns that are numeric (integers or floats)
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns
    
    # Initialize the imputer with the 'mean' strategy
    imputer = SimpleImputer(strategy='mean')
    
    # Apply the imputer to the numeric columns
    # This will calculate the mean for each column and fill NaNs accordingly
    if not numeric_cols.empty:
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
        
    return df_imputed

def handle_outliers(df, column, method='iqr'):
    """Removes outliers from a specific column using IQR method."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered.reset_index(drop=True)

def remove_duplicates(df):
    """
    EDITED ALGORITHM: This function already implements the requested logic.
    The default behavior of pandas' drop_duplicates() is to remove a row only if
    all of its values are identical to another row. No changes to the code are needed.
    """
    return df.drop_duplicates().reset_index(drop=True)

def normalize_data(df, column, method='standardize'):
    """Normalizes or standardizes a specific numeric column."""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return df

    scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df