# app/blueprints/data_preparation/utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# --- الدوال الحالية للتنظيف (بدون تغيير) ---
def remove_missing_values(df):
    """Removes all rows with any NaN values."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame() # Handle non-DataFrame input
    return df.dropna().reset_index(drop=True)

def impute_with_average(df):
    """Imputes missing values for all numeric columns using their mean."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns

    if not numeric_cols.empty:
        # Check for columns that are entirely NaN before imputing
        cols_to_impute = [col for col in numeric_cols if df_imputed[col].isnull().any() and not df_imputed[col].isnull().all()]
        if cols_to_impute:
            try:
                imputer = SimpleImputer(strategy='mean')
                df_imputed[cols_to_impute] = imputer.fit_transform(df_imputed[cols_to_impute])
            except Exception as e:
                # Handle potential errors during imputation (e.g., non-numeric data sneakily included)
                print(f"Error during imputation: {e}")
                raise ValueError("Could not impute missing values. Check column data types.") from e
        # Handle columns that were entirely NaN (SimpleImputer might drop them or raise error depending on version)
        all_nan_cols = [col for col in numeric_cols if df_imputed[col].isnull().all()]
        for col in all_nan_cols:
            df_imputed[col] = 0 # Or another placeholder, or keep as NaN if preferred
            print(f"Warning: Column '{col}' was entirely NaN, filled with 0.")

    return df_imputed

# --- (!!!) (هذه هي الدالة المُعدلة) (!!!) ---
def handle_outliers(df, column, method='iqr'):
    """
    (!!! مُعدل !!!)
    Handles outliers by 'capping' (Winsorizing) them at the 1.5*IQR bounds.
    """
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' not found or is not numeric for outlier handling.")

    df_capped = df.copy() # (!!!) نعمل على نسخة
    series = df_capped[column].dropna()
    
    if series.empty: return df_capped # Return copy if column is all NaN

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    if pd.isna(Q1) or pd.isna(Q3):
        print(f"Warning: Could not calculate IQR for '{column}'. Skipping outlier removal.")
        return df_capped # Return copy

    IQR = Q3 - Q1

    if IQR <= 1e-9: 
        print(f"Warning: IQR for '{column}' is zero or near zero. No outliers capped.")
        return df_capped # Return copy

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # (!!!) هذا هو المنطق الجديد (!!!)
    # نستخدم np.clip لتقييد القيم في مكانها
    df_capped[column] = df_capped[column].clip(lower_bound, upper_bound)
    
    return df_capped.reset_index(drop=True) # إرجاع النسخة المُعدلة
# --- (!!!) (نهاية الدالة المُعدلة) (!!!) ---

def remove_duplicates(df):
    """Removes duplicate rows (where all values in the row are identical)."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    return df.drop_duplicates().reset_index(drop=True)

def normalize_data(df, column, method='standardize'):
    """Normalizes or standardizes a specific numeric column."""
    if not isinstance(df, pd.DataFrame): return pd.DataFrame()
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' not found or is not numeric for normalization.")

    # Create a copy to avoid SettingWithCopyWarning
    df_normalized = df.copy()
    series = df_normalized[column].dropna()

    if series.empty:
        print(f"Warning: Column '{column}' is all NaN. Skipping normalization.")
        return df_normalized # Return original if column is all NaN

    scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()

    try:
        # Reshape data for scaler and fit/transform only non-NaN values
        scaled_values = scaler.fit_transform(series.values.reshape(-1, 1))
        # Place scaled values back into the copied DataFrame using the original index
        df_normalized.loc[series.index, column] = scaled_values.flatten()
    except Exception as e:
        print(f"Error during normalization of column '{column}': {e}")
        raise ValueError(f"Could not normalize column '{column}'. Check data.") from e

    return df_normalized

# --- (دالة هندسة المتغيرات - كما هي) ---
def apply_transformations(df, transformations):
    """
    Applies a list of transformations (log, diff, lag, arithmetic) to the DataFrame.
    Returns the transformed DataFrame and a list of new column names created.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(transformations, list):
        raise ValueError("'transformations' must be a list of dictionaries.")

    df_transformed = df.copy()
    new_column_names = []

    for t in transformations:
        if not isinstance(t, dict): continue # Skip invalid entries

        action = t.get('action')
        variable1 = t.get('variable1')
        variable2 = t.get('variable2') # For arithmetic actions

        if not action or not variable1:
            print(f"Warning: Skipping invalid transformation config: {t}")
            continue
        if variable1 not in df_transformed.columns:
            raise ValueError(f"Variable '{variable1}' not found in the dataset for action '{action}'.")

        new_col_name = ""

        try:
            if action == 'log':
                if not pd.api.types.is_numeric_dtype(df_transformed[variable1]):
                    raise ValueError(f"Logarithm requires a numeric column, but '{variable1}' is not.")
                new_col_name = f"log_{variable1}"
                series_to_log = df_transformed[variable1].astype(float)
                if (series_to_log <= 0).any():
                    print(f"Warning: Column '{variable1}' contains non-positive values. Adding a small constant (1e-9) before taking log.")
                    # Handle potential 0 or negative values by adding a small epsilon
                    series_to_log = series_to_log.apply(lambda x: x if x > 0 else 1e-9)
                df_transformed[new_col_name] = np.log(series_to_log)

            elif action == 'diff':
                order = t.get('order', 1)
                if not isinstance(order, int) or order < 1: order = 1 # Ensure valid order
                new_col_name = f"{variable1}_diff{order}"
                df_transformed[new_col_name] = df_transformed[variable1].diff(periods=order)

            elif action == 'lag':
                lags = t.get('lags', 1)
                if not isinstance(lags, int) or lags < 1: lags = 1 # Ensure valid lag
                new_col_name = f"{variable1}_lag{lags}"
                df_transformed[new_col_name] = df_transformed[variable1].shift(lags)

            elif action in ['add', 'subtract', 'multiply', 'divide']:
                if not variable2:
                    raise ValueError(f"Action '{action}' requires a second variable ('variable2').")
                if variable2 not in df_transformed.columns:
                    raise ValueError(f"Second variable '{variable2}' not found for action '{action}'.")
                if variable1 == variable2:
                    raise ValueError(f"Cannot perform '{action}' with the same variable ('{variable1}'). Select two different variables.")

                col1 = pd.to_numeric(df_transformed[variable1], errors='coerce')
                col2 = pd.to_numeric(df_transformed[variable2], errors='coerce')

                if col1.isnull().all() or col2.isnull().all():
                    raise ValueError(f"One or both variables ('{variable1}', '{variable2}') could not be converted to numeric for action '{action}'.")

                if action == 'add':
                    new_col_name = f"{variable1}_plus_{variable2}"
                    df_transformed[new_col_name] = col1 + col2
                elif action == 'subtract':
                    new_col_name = f"{variable1}_minus_{variable2}"
                    df_transformed[new_col_name] = col1 - col2
                elif action == 'multiply':
                    new_col_name = f"{variable1}_times_{variable2}"
                    df_transformed[new_col_name] = col1 * col2
                elif action == 'divide':
                    new_col_name = f"{variable1}_over_{variable2}"
                    col2_safe = col2.replace(0, np.nan)
                    df_transformed[new_col_name] = col1 / col2_safe

            elif action == 'square': # مثال: إضافة التربيع
                if not pd.api.types.is_numeric_dtype(df_transformed[variable1]):
                    raise ValueError(f"Square requires a numeric column, but '{variable1}' is not.")
                new_col_name = f"{variable1}_sq"
                df_transformed[new_col_name] = df_transformed[variable1].astype(float) ** 2

            elif action == 'drop': # Drop action doesn't create a new column
                if variable1 in df_transformed.columns:
                    df_transformed = df_transformed.drop(columns=[variable1])
                    print(f"Dropped column: {variable1}")
                else:
                    print(f"Warning: Column '{variable1}' not found for dropping.")
                continue # Skip adding to new_column_names for 'drop'

            else:
                print(f"Warning: Unknown transformation action '{action}'. Skipping.")
                continue # Skip unknown actions

            if new_col_name:
                if new_col_name in df.columns: # Check if overwriting existing column
                    print(f"Warning: Transformation '{action}' overwrote existing column '{new_col_name}'.")
                new_column_names.append(new_col_name)

        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Error applying action '{action}' on variable '{variable1}': {str(e)}") from e
        except Exception as e: # Catch any other unexpected errors
            raise RuntimeError(f"Unexpected error during action '{action}' on '{variable1}': {str(e)}") from e

    # (ملاحظة: لقد قمت بإزالة dropna() من هنا. من الأفضل أن يتحكم المستخدم بها من الفرونت إند)
    # original_rows = len(df_transformed)
    # df_transformed.dropna(inplace=True)
    # rows_dropped = original_rows - len(df_transformed)
    # if rows_dropped > 0:
    #     print(f"Note: {rows_dropped} rows containing NaN (likely due to lags/diffs) were dropped.")

    return df_transformed, list(set(new_column_names))
# --- (نهاية الإضافة) ---

# --- (إضافة جديدة) دالة حذف الأعمدة ---
def delete_columns(df, columns_to_delete):
    """
    Deletes a list of specified columns from the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(columns_to_delete, list):
        raise ValueError("'columns_to_delete' must be a list.")
    
    # (مهم) التأكد من أن الأعمدة موجودة قبل محاولة حذفها
    existing_cols_to_delete = [col for col in columns_to_delete if col in df.columns]
    
    if not existing_cols_to_delete:
        raise ValueError("None of the selected columns were found in the dataset.")

    try:
        df_deleted = df.drop(columns=existing_cols_to_delete, errors='ignore')
        
        # (مهم) التحقق إذا كانت كل الأعمدة ستحذف
        if df_deleted.empty or len(df_deleted.columns) == 0:
            raise ValueError("Cannot delete all columns. At least one column must remain.")
            
        return df_deleted, existing_cols_to_delete
        
    except Exception as e:
        raise RuntimeError(f"Error during column deletion: {e}") from e
# --- (نهاية الإضافة) ---


# --- (!!!) (إضافة جديدة) دالة لإنشاء المتغيرات الوهمية (!!!) ---
def create_dummy_variables(df, columns_to_dummify, drop_first=True):
    """
    Converts categorical columns into dummy (0/1) variables.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(columns_to_dummify, list) or not columns_to_dummify:
        raise ValueError("'columns_to_dummify' must be a non-empty list.")

    df_dummified = df.copy()
    new_column_names = []
    
    # (حماية) حد أقصى لعدد القيم الفريدة لمنع انفجار الذاكرة
    MAX_UNIQUE_VALUES = 100 
    
    for col in columns_to_dummify:
        if col not in df_dummified.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
        
        # التأكد أن العمود ليس رقمياً (رغم أن الفرونت إند سيرسل نصوصاً فقط)
        if pd.api.types.is_numeric_dtype(df_dummified[col]):
             print(f"Warning: '{col}' is numeric. Dummies may not be meaningful.")

        # التحقق من عدد القيم الفريدة
        unique_count = df_dummified[col].nunique()
        if unique_count > MAX_UNIQUE_VALUES:
            raise ValueError(f"Column '{col}' has {unique_count} unique values (max {MAX_UNIQUE_VALUES}). Cannot create dummies.")
        if unique_count <= 1:
            print(f"Warning: Column '{col}' has only one unique value. Skipping dummy creation.")
            continue

        # إنشاء المتغيرات الوهمية مع حذف أول متغير (لتجنب الدامي فاريابل تراب)
        dummies = pd.get_dummies(df_dummified[col], prefix=col, drop_first=drop_first, dtype=int) # (إضافة: dtype=int)
        
        # إضافة الأعمدة الجديدة للـ DataFrame
        df_dummified = pd.concat([df_dummified, dummies], axis=1)
        
        # إضافة أسماء الأعمدة الجديدة للقائمة
        new_column_names.extend(dummies.columns.tolist())
        
        # حذف العمود الأصلي (النصي)
        df_dummified = df_dummified.drop(columns=[col])

    return df_dummified, new_column_names
# --- (!!!) (نهاية الإضافة) (!!!) ---