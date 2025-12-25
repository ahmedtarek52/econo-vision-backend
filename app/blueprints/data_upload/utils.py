import pandas as pd
import numpy as np
import io 

def get_file_extension(filename):
    """Extracts the file extension from a filename."""
    if '.' not in filename:
        return None
    return filename.rsplit('.', 1)[1].lower()

def read_dataframe_from_file(file_storage, file_extension, sheet_name=None):
    """
    Reads an uploaded file (CSV, Excel, JSON) into a pandas DataFrame.
    Handles reading a specific sheet for Excel.
    """
    try:
        file_storage.seek(0) # Rewind the file stream
        if file_extension == 'csv':
            try:
                df = pd.read_csv(file_storage)
            except (pd.errors.ParserError, UnicodeDecodeError):
                file_storage.seek(0)
                print("CSV parsing failed, trying semicolon delimiter...")
                df = pd.read_csv(file_storage, sep=';')
                
        elif file_extension in ['xlsx', 'xls']:
            # sheet_name=None (default) reads the first sheet
            # sheet_name='SpecificSheet' reads that sheet
            df = pd.read_excel(file_storage, sheet_name=sheet_name, engine='openpyxl')
            
        elif file_extension == 'json':
            df = pd.read_json(file_storage, orient='records')
            
        else:
            raise ValueError(f"Unsupported file format: '.{file_extension}'.")
            
    except Exception as e:
        print(f"Error reading file stream: {e}")
        raise ValueError(f"Could not read the file. Ensure it is a valid, uncorrupted '{file_extension}' file. Error: {e}")
        
    if df.empty:
        raise ValueError("File is empty or could not be read.")

    return df

def post_process_dataframe(df, filename):
    """
    Takes a raw DataFrame and performs all necessary processing 
    (cleaning, quality checks, typing) and returns a JSON response dict.
    """
    
    # --- 1. Data Cleaning on Columns ---
    original_cols = df.columns.tolist()
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_', regex=False)
    df.columns = df.columns.str.replace('.', '_', regex=False)
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True) 
    
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    cleaned_col_map = dict(zip(original_cols, df.columns))
    print(f"Cleaned column names: {cleaned_col_map}")

    # --- 2. Infer Datetime (before type suggestion) ---
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # Check if it looks like a date before trying to parse
                # (Simple check to avoid parsing all object columns)
                sample_val = df[col].dropna().iloc[0]
                if isinstance(sample_val, str) and (len(sample_val) > 6 and ('/' in sample_val or '-' in sample_val)):
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    # If parsing is successful for > 50% of non-null values, convert
                    if parsed_dates.notnull().sum() > (df[col].notnull().sum() * 0.5):
                        print(f"Inferred column '{col}' as datetime.")
                        df[col] = parsed_dates
            except Exception:
                pass # Ignore errors on sampling or parsing

    # --- 3. Run Quality & Type Checks ---
    quality_report = check_data_quality(df)
    suggested_types = suggest_variable_types(df)
    num_rows, num_cols = df.shape

    # --- 4. Prepare for JSON serialization ---
    # Replace np.nan/pd.NaT with Python's None
    df_cleaned_for_json = df.replace({np.nan: None, pd.NaT: None, np.inf: None, -np.inf: None})

    # --- 5. Build Response Dictionary ---
    response_data = {
        "status": "success", # (جديد)
        "message": "File processed successfully!",
        "filename": filename,
        "columns": list(df.columns),
        "previewData": df_cleaned_for_json.head(20).to_dict(orient='records'),
        "suggestedTypes": suggested_types,
        "fullDataset": df_cleaned_for_json.to_dict(orient='records'),
        "qualityReport": quality_report,
        "numRows": num_rows,
        "numCols": num_cols 
    }
    return response_data

# --- (الدوال المساعدة كما هي) ---

def suggest_variable_types(df):
    """
    Infers the type of each column (Quantitative, Nominal, Datetime, Text)
    for frontend suggestions.
    """
    if not isinstance(df, pd.DataFrame): return {}
    
    suggested_types = {}
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            # Check if it looks like an ID (integer, high unique)
            if pd.api.types.is_integer_dtype(dtype) and df[col].nunique() > (len(df) * 0.9):
                suggested_types[col] = "ID (Integer)"
            else:
                suggested_types[col] = "Quantitative (Numeric)"
        elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            suggested_types[col] = "Datetime"
        elif pd.api.types.is_string_dtype(dtype) or dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
            unique_vals_count = df[col].nunique()
            total_rows = len(df)
            
            # Heuristic for categorical (nominal) vs. text
            if (unique_vals_count / total_rows < 0.5 and unique_vals_count < 100) or unique_vals_count == total_rows:
                suggested_types[col] = "Nominal (Categorical)"
            else:
                suggested_types[col] = "Text (String)"
        elif pd.api.types.is_bool_dtype(dtype):
            suggested_types[col] = "Binary (Boolean)"
        else:
            suggested_types[col] = str(dtype) # Fallback
            
    return suggested_types

def check_data_quality(df):
    """
    Performs a basic check for missing values and duplicates.
    Returns a dict {is_ok, issues}
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"is_ok": False, "issues": ["Data is empty or invalid."]}

    issues = []
    total_cells = df.size
    
    # 1. Check for missing values
    missing_cells = df.isnull().sum().sum()
    if missing_cells > 0:
        missing_percentage = (missing_cells / total_cells) * 100
        issues.append(f"Data contains **{missing_cells} missing values** ({missing_percentage:.1f}% of cells). Use 'Impute Missing' or 'Remove Missing' in Data Prep.")
        
    # 2. Check for duplicate rows
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        issues.append(f"Found **{duplicate_rows} duplicate rows**. Use 'Remove Duplicates' in Data Prep.")
        
    # 3. Check for non-numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    if non_numeric_cols:
        if len(numeric_cols) > 0:
            issues.append(f"Found **{len(non_numeric_cols)} non-numeric columns** (e.g., {', '.join(non_numeric_cols[:3])}{'...' if len(non_numeric_cols)>3 else ''}). These will be ignored by most models.")

    is_ok = len(issues) == 0
    if is_ok:
        issues.append("Data quality check passed. No immediate issues found. Ready for preparation.")
        
    return {"is_ok": is_ok, "issues": issues}