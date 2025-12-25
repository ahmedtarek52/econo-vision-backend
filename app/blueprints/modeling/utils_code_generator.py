# app/blueprints/modeling/utils_code_generator.py

import pandas as pd
import numpy as np
import re # (جديد) نحتاجه لتحليل المتغيرات المصدر

"""
(نسخة مكتملة ومصححة واحترافية)
هذا الموديول يقوم بإنشاء "أكواد توضيحية" (Code Snippets)
لأغراض التوثيق العلمي وإعادة الإنتاج (Reproducibility).
"""

# --- (!!!) (إضافة جديدة) دالة مساعدة ذكية (!!!) ---
def _extract_source_vars_from_history(transformation_history):
    """
    (جديد) يقرأ سجل التحويلات (المبني على النصوص)
    ويستخرج كل المتغيرات "المصدر" المستخدمة لإنشاء العينة.
    هذا يحل مشكلة KeyError (مثل 'nfa' not found).
    """
    source_vars = set()
    if not transformation_history:
        return source_vars
        
    # التحقق أولاً إذا كان التنسيق هو Objects (الاحترافي)
    if isinstance(transformation_history[0], dict):
        for step in transformation_history:
            if step.get("type") in ["transform", "clean", "delete_cols"]:
                source_vars.update(step.get("inputVars", []))
        return source_vars

    # إذا لم يكن Objects، استخدم التنسيق القديم (String parsing)
    # (هذا regex بسيط للبحث عن الكلمات داخل الأقواس)
    var_pattern = re.compile(r'\((.*?)\)')

    for step in transformation_history:
        try:
            if "Applied: " in step:
                # "Applied: log(nfa) -> log_nfa"
                # "Applied: add(var1, var2) -> new_var"
                action_full = step.split('Applied: ')[1].split(' -> ')[0]
                match = var_pattern.search(action_full)
                if match:
                    vars_part = match.group(1) # 'nfa' or 'var1, var2'
                    vars_list = [v.strip() for v in vars_part.split(',')]
                    source_vars.update(vars_list)
        except Exception as e:
            print(f"Code gen (source var extraction) error: {e} on step: {step}")
            
    return source_vars
# --- (!!!) (نهاية الدالة الجديدة) (!!!) ---


# --- (!!!) (هذا هو الكود الذي تم إصلاحه بالكامل) (!!!) ---
def _get_prep_steps_code(transformation_history):
    """
    (مُحدّث ومُحترف)
    يحول مصفوفة تاريخ التحويلات من الفرونت إند إلى كود Python.
    
    (جديد): يدعم طريقتين:
    1. (الأفضل) مصفوفة من Objects: [{"type": "transform", "operation": "log", ...}]
    2. (القديمة) مصفوفة من Strings: ["Applied: log(nfa) -> log_nfa"]
    """
    imports_needed = set()
    
    if not transformation_history:
        return "", "# --- 2. Data Preparation Steps ---\n# (No operations were applied)\n"
    
    code_lines = [
        "# --- 2. Data Preparation Steps ---",
        "# Applying operations from Data Prep screen..."
    ]

    lag_or_diff_applied = False # (جديد) نتتبع هذا بشكل عام
    
    # التحقق من نوع البيانات (جديد أم قديم)
    is_new_object_format = isinstance(transformation_history[0], dict)

    try:
        if is_new_object_format:
            # --- (المنطق الاحترافي الجديد: Object-based) ---
            code_lines.append("# (Processing new object-based history)")
            
            for step in transformation_history:
                op_type = step.get("type")
                op = step.get("operation")
                
                if op_type == "transform":
                    var_in = step.get("inputVars", [])
                    var_out = step.get("outputVar")
                    
                    if op == 'log':
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply Logarithm")
                        code_lines.append(f"data['{var_out}'] = np.log(data['{var_in[0]}'].apply(lambda x: x if x > 0 else 1e-9))")
                    
                    elif op == 'lag':
                        lag_order = step.get("params", {}).get("lags", 1)
                        code_lines.append(f"\n# Apply Lag")
                        code_lines.append(f"data['{var_out}'] = data['{var_in[0]}'].shift({int(lag_order)})")
                        lag_or_diff_applied = True
                    
                    elif op == 'diff':
                        diff_order = step.get("params", {}).get("order", 1)
                        code_lines.append(f"\n# Apply Difference")
                        code_lines.append(f"data['{var_out}'] = data['{var_in[0]}'].diff({int(diff_order)})")
                        lag_or_diff_applied = True
                    
                    elif op in ['add', 'subtract', 'multiply', 'divide']:
                        var1, var2 = var_in[0], var_in[1]
                        op_map = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
                        code_lines.append(f"\n# Apply Arithmetic: {op}")
                        if op == 'divide':
                            imports_needed.add("import numpy as np")
                            code_lines.append(f"data['{var_out}'] = data['{var1}'] / data['{var2}'].replace(0, np.nan)") 
                        else:
                            code_lines.append(f"data['{var_out}'] = data['{var1}'] {op_map[op]} data['{var2}']")
                
                elif op_type == "clean":
                    if op == "dropna":
                        code_lines.append(f"\n# Apply: Remove Missing Values (dropna)")
                        code_lines.append("data = data.dropna()")
                    
                    elif op == "impute_mean":
                        imports_needed.add("from sklearn.impute import SimpleImputer")
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Impute Missing Values (Mean)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("imputer = SimpleImputer(strategy='mean')")
                        code_lines.append("data[numeric_cols] = imputer.fit_transform(data[numeric_cols])")

                    elif op == "normalize":
                        imports_needed.add("from sklearn.preprocessing import StandardScaler")
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Normalize Data (StandardScaler)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("scaler = StandardScaler()")
                        code_lines.append("data[numeric_cols] = scaler.fit_transform(data[numeric_cols])")

                    elif op == "outliers_iqr":
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Handle Outliers (IQR Capping)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("for col in numeric_cols:")
                        code_lines.append("    Q1 = data[col].quantile(0.25)")
                        code_lines.append("    Q3 = data[col].quantile(0.75)")
                        code_lines.append("    IQR = Q3 - Q1")
                        code_lines.append("    if IQR > 0:")
                        code_lines.append("        lower_bound = Q1 - 1.5 * IQR")
                        code_lines.append("        upper_bound = Q3 + 1.5 * IQR")
                        code_lines.append("        data[col] = np.clip(data[col], lower_bound, upper_bound)")

                elif op_type == "delete_cols":
                    cols_list = step.get("inputVars", [])
                    code_lines.append(f"\n# Apply Column Deletion")
                    code_lines.append(f"data = data.drop(columns={cols_list}, errors='ignore')")

        else:
            # --- (المنطق القديم: String-based Fallback) ---
            code_lines.append("# (Processing legacy string-based history)")
            for step in transformation_history:
                # الحالة 1: التحويلات (التي تنشئ عموداً جديداً)
                if " -> " in step and "Applied: " in step:
                    action_desc, new_col = step.split(' -> ')
                    action_full = action_desc.split('Applied: ')[1]
                    action_type = action_full.split('(')[0]
                    vars_part = action_full[len(action_type)+1 : -1] # جلب ما بداخل الأقواس
                    
                    if action_type == 'log':
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply Logarithm")
                        code_lines.append(f"data['{new_col}'] = np.log(data['{vars_part}'].apply(lambda x: x if x > 0 else 1e-9))")
                    
                    elif action_type == 'lag':
                        var_name, lag_order = vars_part.split(', ')
                        code_lines.append(f"\n# Apply Lag")
                        code_lines.append(f"data['{new_col}'] = data['{var_name}'].shift({int(lag_order)})")
                        lag_or_diff_applied = True
                    
                    elif action_type == 'diff':
                        imports_needed.add("import numpy as np")
                        var_name, diff_order = vars_part.split(', ')
                        code_lines.append(f"\n# Apply Difference")
                        code_lines.append(f"data['{new_col}'] = data['{var_name}'].diff({int(diff_order)})")
                        lag_or_diff_applied = True

                    elif action_type in ['add', 'subtract', 'multiply', 'divide']:
                        var1, var2 = vars_part.split(', ')
                        op_map = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/'}
                        code_lines.append(f"\n# Apply Arithmetic: {action_type}")
                        if action_type == 'divide':
                            imports_needed.add("import numpy as np")
                            code_lines.append(f"data['{new_col}'] = data['{var1}'] / data['{var2}'].replace(0, np.nan)") 
                        else:
                            code_lines.append(f"data['{new_col}'] = data['{var1}'] {op_map[action_type]} data['{var2}']")
                    
                    else:
                        code_lines.append(f"\n# (Transformation '{action_type}' not implemented in code generator yet)")

                # الحالة 2: حذف الأعمدة
                elif step.startswith("Deleted columns: "):
                    cols_str = step.split("Deleted columns: ")[1]
                    cols_list = [col.strip() for col in cols_str.split(',')]
                    code_lines.append(f"\n# Apply Column Deletion")
                    code_lines.append(f"data = data.drop(columns={cols_list}, errors='ignore')")

                # الحالة 3: عمليات التنظيف
                elif step.startswith("Applied: "):
                    op_name = step.split("Applied: ")[1]
                    
                    if "Remove Duplicates" in op_name:
                        code_lines.append(f"\n# Apply: Remove Duplicates")
                        code_lines.append("data = data.drop_duplicates()")
                    
                    elif "Remove Missing Values" in op_name:
                        code_lines.append(f"\n# Apply: Remove Missing Values (dropna)")
                        code_lines.append("data = data.dropna()")
                    
                    elif "Impute Missing Values" in op_name:
                        imports_needed.add("from sklearn.impute import SimpleImputer")
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Impute Missing Values (Mean)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("imputer = SimpleImputer(strategy='mean')")
                        code_lines.append("data[numeric_cols] = imputer.fit_transform(data[numeric_cols])")

                    elif "Normalize Data" in op_name:
                        imports_needed.add("from sklearn.preprocessing import StandardScaler")
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Normalize Data (StandardScaler)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("scaler = StandardScaler()")
                        code_lines.append("data[numeric_cols] = scaler.fit_transform(data[numeric_cols])")

                    elif "Handle Outliers" in op_name:
                        imports_needed.add("import numpy as np")
                        code_lines.append(f"\n# Apply: Handle Outliers (IQR Capping)")
                        code_lines.append("numeric_cols = data.select_dtypes(include=np.number).columns")
                        code_lines.append("for col in numeric_cols:")
                        code_lines.append("    Q1 = data[col].quantile(0.25)")
                        code_lines.append("    Q3 = data[col].quantile(0.75)")
                        code_lines.append("    IQR = Q3 - Q1")
                        code_lines.append("    if IQR > 0:")
                        code_lines.append("        lower_bound = Q1 - 1.5 * IQR")
                        code_lines.append("        upper_bound = Q3 + 1.5 * IQR")
                        code_lines.append("        data[col] = np.clip(data[col], lower_bound, upper_bound)")
                    
                    else:
                        code_lines.append(f"# (Cleaning step '{op_name}' not implemented in code generator)")
    
    except Exception as e:
        print(f"Code gen prep error: {e} on step: {step}")
        code_lines.append(f"\n# (Error parsing prep step: {step})")

    # --- (!!!) هذا هو الإصلاح النهائي (!!!) ---
    # إذا تم استخدام lag أو diff في أي وقت (سواء بالطريقة الجديدة أو القديمة)
    # ولم يقم المستخدم بالضغط على "dropna" *بعدها*، يجب أن نضيفها.
    
    if lag_or_diff_applied:
        # (إضافة) تحقق ذكي: هل "dropna" هي آخر خطوة؟
        last_op_was_dropna = False
        try:
            if is_new_object_format:
                last_op_was_dropna = (transformation_history[-1].get("type") == "clean" and transformation_history[-1].get("operation") == "dropna")
            else:
                last_op_was_dropna = "Remove Missing Values" in transformation_history[-1]
        except Exception:
            pass # Keep it False if history is weird

        if not last_op_was_dropna:
            code_lines.append("\n# Dropping NaNs created by Lag/Diff operations")
            code_lines.append("data = data.dropna()")


    imports_header = "\n".join(list(imports_needed))
    code_str = "\n".join(code_lines)
    
    return imports_header, code_str
# --- (نهاية الدالة المُصلحة) ---


def _get_data_snippet(df, required_cols):
    """
    (Helper)
    يأخذ عينة من البيانات ويحولها إلى كود Python لـ DataFrame.
    """
    imports_needed = set() 
    try:
        # (!!!) (إصلاح) تأكد من أننا لا نطلب أعمدة غير موجودة (!!!)
        valid_cols = [col for col in required_cols if col in df.columns]
        # (إصلاح) تأكد من عدم وجود تكرار
        valid_cols = list(dict.fromkeys(valid_cols)) 
        
        if not valid_cols:
            return "", "# (Could not find required columns in dataset for snippet)\n"

        sample_df = df[valid_cols].head(5)
        data_dict = sample_df.to_dict('list')
        
        data_str_lines = ["{"]
        for key, values in data_dict.items():
            formatted_values = []
            for v in values:
                if isinstance(v, str):
                    formatted_values.append(f"'{v}'")
                elif pd.isna(v):
                    imports_needed.add("import numpy as np") 
                    formatted_values.append("np.nan") 
                else:
                    formatted_values.append(str(v))
                    
            data_str_lines.append(f"    '{key}': [{', '.join(formatted_values)}],")
        data_str_lines.append("}")
        data_str = "\n".join(data_str_lines)

        header = "# --- 1. Load Data ---\n"
        header += "# We recommend loading your full dataset (e.g., from CSV)\n"
        header += "# data = pd.read_csv('your_file.csv')\n\n"
        header += "# Using a 5-row sample for demonstration:\n"
        header += f"data_dict = {data_str}\n"
        header += "data = pd.DataFrame(data_dict)\n"
        
        imports_header = "\n".join(list(imports_needed))
        
        return imports_header, header 
    except Exception as e:
        print(f"Error creating data snippet: {e}")
        return "", f"# Error: Could not generate data snippet.\n# {e}\n"


# --- دوال إنشاء الكود (مع إصلاح OLS و ARDL) ---

def _get_ols_code(params):
    y_var = params.get('dependent_var')
    x_vars = params.get('independent_vars', [])
    
    imports = "import pandas as pd\nimport statsmodels.api as sm\n"
    
    cov_type = params.get('cov_type', 'nonrobust')
    fit_note = f"Note: Model fitted with '{cov_type}' standard errors."
    
    fit_command = ""
    if cov_type == 'HAC':
        fit_command = "results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})"
    elif cov_type in ['HC0', 'HC1', 'HC2', 'HC3']:
        fit_command = f"results = model.fit(cov_type='{cov_type}')"
    else:
        fit_command = "results = model.fit()"
        fit_note = "" 

    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
X = data[{x_vars}]

# --- 4. Add Constant ---
X = sm.add_constant(X, has_constant='raise')

# --- 5. Fit OLS Model ---
model = sm.OLS(Y, X)

# Fit the model (with robust errors if specified)
{fit_command}

# --- 6. View Results ---
print(results.summary())
"""
    
    if fit_note:
        model_logic += f"\nprint(f\"\\n{fit_note}\")\n"
        
    return imports, model_logic

def _get_penalized_code(model_id, params):
    y_var = params.get('dependent_var')
    x_vars = params.get('independent_vars', [])
    
    model_map = {
        'lasso': 'LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)',
        'ridge': 'RidgeCV(alphas=np.logspace(-6, 6, 100), cv=5)',
        'elastic_net': 'ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .99, 1.0], cv=5, random_state=42, n_jobs=-1, max_iter=2000)'
    }
    import_map = {
        'lasso': 'from sklearn.linear_model import LassoCV\n',
        'ridge': 'from sklearn.linear_model import RidgeCV\nimport numpy as np\n',
        'elastic_net': 'from sklearn.linear_model import ElasticNetCV\n'
    }
    
    imports = "import pandas as pd\nfrom sklearn.pipeline import Pipeline\n"
    imports += "from sklearn.preprocessing import StandardScaler\n"
    imports += import_map[model_id]
    
    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
X = data[{str(x_vars)}]

# --- 4. Create Pipeline ---
# {model_id.title()} requires data to be standardized (scaled).
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('{model_id}', {model_map[model_id]})
])

# --- 5. Fit Model ---
pipeline.fit(X, Y)

# --- 6. View Results ---
model_fit = pipeline.named_steps['{model_id}']
best_alpha = model_fit.alpha_
coefficients = model_fit.coef_
intercept = model_fit.intercept_

print(f"--- {model_id.title()}CV Results ---")
print(f"Optimal Alpha (λ) found by CV: {{best_alpha}}")
if hasattr(model_fit, 'l1_ratio_'):
    print(f"Optimal L1 Ratio: {{model_fit.l1_ratio_}} (1.0=Lasso, 0.0=Ridge)")
print(f"Intercept: {{intercept}}")
print("\\nCoefficients:")
for coef, name in zip(coefficients, {str(x_vars)}):
    print(f"  {{name}}: {{coef:.6f}}")
"""
    return imports, model_logic

def _get_logit_probit_code(model_id, params):
    y_var = params.get('dependent_var')
    x_vars = params.get('independent_vars', [])
    model_func = "sm.Logit" if model_id == 'logit' else "sm.Probit"
    
    imports = "import pandas as pd\nimport statsmodels.api as sm\n"
    
    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
X = data[{x_vars}]
X = sm.add_constant(X, has_constant='raise')

# --- 4. Fit {model_id.title()} Model ---
model = {model_func}(Y, X)
results = model.fit(disp=False)

# --- 5. View Results ---
print(results.summary())
"""
    return imports, model_logic

def _get_arima_code(params):
    y_var = params.get('endog_var')
    x_vars = params.get('exog_vars') # قد تكون None
    order = params.get('order', (1,0,0))
    seasonal_order = params.get('seasonal_order', (0,0,0,0))
    
    imports = "import pandas as pd\nfrom statsmodels.tsa.statespace.sarimax import SARIMAX\n"
    
    model_logic = f"""
# --- 3. Define Variables ---
# (Note: Ensure data is time-indexed if applicable)
endog = data['{y_var}']
exog = None
"""
    
    if x_vars:
        model_logic += f"exog = data[{x_vars}]\n"
    
    model_logic += f"""
# --- 4. Fit Model ---
# Model: SARIMAX(p,d,q)(P,D,Q,S)
model = SARIMAX(endog, 
                exog=exog, 
                order={tuple(order)}, 
                seasonal_order={tuple(seasonal_order)},
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=False)

# --- 5. View Results ---
print(results.summary())
"""
    return imports, model_logic

def _get_var_code(params):
    variables = params.get('variables', [])
    maxlags = params.get('maxlags', 4)
    
    imports = "import pandas as pd\nfrom statsmodels.tsa.api import VAR\n"
    
    model_logic = f"""
# --- 3. Define Variables ---
# (Note: VAR assumes variables are stationary I(0))
model_data = data[{variables}]

# --- 4. Fit VAR Model ---
model = VAR(model_data)

# Select lags automatically based on AIC (up to maxlags)
results = model.fit(maxlags={maxlags}, ic='aic')

# --- 5. View Results ---
print(results.summary())
"""
    return imports, model_logic

def _get_vecm_code(params):
    variables = params.get('variables', [])
    lags_p = params.get('lags', 2)
    rank = params.get('coint_rank', 1)
    k_ar_diff = lags_p - 1
    
    imports = "import pandas as pd\nfrom statsmodels.tsa.api import VECM\n"
    
    model_logic = f"""
# --- 3. Define Variables ---
# (Note: VECM assumes variables are I(1) but cointegrated)
model_data = data[{variables}]

# --- 4. Fit VECM Model ---
# Lags (p) = {lags_p}, so k_ar_diff = p-1 = {k_ar_diff}
model = VECM(model_data, 
             k_ar_diff={k_ar_diff}, 
             coint_rank={rank}, 
             deterministic='ci')

results = model.fit()

# --- 5. View Results ---
print(results.summary())
"""
    return imports, model_logic

def _get_ml_code(model_id, params):
    y_var = params.get('dependent_var')
    x_vars = params.get('independent_vars', [])
    
    imports = "import pandas as pd\nfrom sklearn.model_selection import train_test_split\n"
    imports += "from sklearn.metrics import r2_score\n"
    
    if model_id == 'random_forest':
        imports += "from sklearn.ensemble import RandomForestRegressor\n"
        model_init = "model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)"
    else: # xgboost
        imports += "from xgboost import XGBRegressor\n"
        model_init = "model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)"
        
    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
X = data[{str(x_vars)}]

# --- 4. Split Data (Time Series Aware) ---
# shuffle=False is crucial for time-series data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# --- 5. Initialize & Fit Model ---
{model_init}
model.fit(X_train, Y_train)

# --- 6. Evaluate on Test Set ---
Y_pred = model.predict(X_test)
r2_test = r2_score(Y_test, Y_pred)
print(f"--- {model_id.title()} Results ---")
print(f"Test Set R-squared: {{r2_test:.4f}}")

if hasattr(model, 'feature_importances_'):
    print("\\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index={str(x_vars)})
    print(importances.sort_values(ascending=False))
"""
    return imports, model_logic

def _get_double_ml_code(params):
    y_var = params.get('dependent_var')
    d_var = params.get('treatment_var')
    x_vars = params.get('control_vars', [])
    ml_method = params.get('ml_method', 'lasso')
    
    imports = "import pandas as pd\nimport numpy as np\nimport statsmodels.api as sm\n"
    imports += "from sklearn.model_selection import KFold\n"
    imports += "from sklearn.preprocessing import StandardScaler\n"
    imports += "from sklearn.pipeline import Pipeline\n"
    
    if ml_method == 'lasso':
        imports += "from sklearn.linear_model import Lasso\n"
    else:
        imports += "from sklearn.ensemble import RandomForestRegressor\n"
        
    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
D = data['{d_var}']
X = data[{x_vars}]

# --- 4. Select ML Method ---
if '{ml_method}' == 'lasso':
    model_g = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1))])
    model_h = Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1))])
else:
    model_g = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    model_h = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)

# --- 5. Implement Cross-fitting (Partialling-Out) ---
kf = KFold(n_splits=2, shuffle=True, random_state=42)
eta_y_residuals = np.zeros(len(data))
eta_d_residuals = np.zeros(len(data))

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    D_train, D_test = D.iloc[train_index], D.iloc[test_index]

    # Fit g(X) to predict Y
    model_g.fit(X_train, Y_train)
    Y_pred = model_g.predict(X_test)
    eta_y_residuals[test_index] = Y_test - Y_pred

    # Fit h(X) to predict D
    model_h.fit(X_train, D_train)
    D_pred = model_h.predict(X_test)
    eta_d_residuals[test_index] = D_test - D_pred

print("Cross-fitting complete.")

# --- 6. Final OLS on Residuals ---
# This estimates the pure causal effect 'alpha'
eta_d_reshaped = sm.add_constant(eta_d_residuals)
final_ols_model = sm.OLS(eta_y_residuals, eta_d_reshaped)
final_results = final_ols_model.fit()

print("\\n--- Causal ML (DML) Results ---")
print(f"Causal Effect of '{d_var}' on '{y_var}':")
print(final_results.summary().tables[1])
"""
    return imports, model_logic

def _get_garch_code(params):
    y_var = params.get('endog_var')
    p = params.get('p', 1)
    q = params.get('q', 1)
    
    imports = "import pandas as pd\nfrom arch import arch_model\n"
    
    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']

# --- 4. Scale Data (Common practice for GARCH) ---
# Scaling by 100 often helps the optimizer converge
scaled_Y = Y * 100

# --- 5. Fit GARCH Model ---
model = arch_model(scaled_Y, 
                   vol='Garch', 
                   p={p}, 
                   q={q}, 
                   mean='Constant', 
                   dist='Normal')

results = model.fit(disp='off')

# --- 6. View Results ---
print(results.summary())
"""
    return imports, model_logic

def _get_ardl_code(params):
    y_var = params.get('endog_var')
    x_vars = params.get('exog_vars', [])
    lags = params.get('lags', 1)
    
    cov_type = params.get('cov_type', 'nonrobust')
    fit_note = f"Note: Model fitted with '{cov_type}' standard errors."
    
    fit_command = ""
    if cov_type == 'HAC':
        fit_command = "results = model.fit(cov_type='HAC')"
    elif cov_type in ['HC0', 'HC1', 'HC2', 'HC3']:
        fit_command = f"results = model.fit(cov_type='{cov_type}')"
    else:
        fit_command = "results = model.fit()"
        fit_note = ""
    
    imports = "import pandas as pd\n"
    imports += "# Note: ARDL import path may vary by statsmodels version\n"
    imports += "try:\n    from statsmodels.tsa.ardl import ARDL\n"
    imports += "except ImportError:\n    from statsmodels.tsa.api import ARDL\n"

    model_logic = f"""
# --- 3. Define Variables ---
Y = data['{y_var}']
X = data[{x_vars}]

# --- 4. Fit ARDL Model ---
# Using lags={lags} for endogenous var and order={lags} for all exogenous
model = ARDL(Y, 
             lags={lags}, 
             exog=X, 
             order={lags}, 
             trend='c')

# Fit the model (with robust errors if specified)
{fit_command}

# --- 5. View Results ---
print(results.summary())
"""
    if fit_note:
        model_logic += f"\nprint(f\"\\n{fit_note}\")\n"
        
    return imports, model_logic

def _get_panel_code(params):
    y_var = params.get('dependent_var')
    x_vars = params.get('independent_vars', [])
    id_var = params.get('panel_id_var')
    time_var = params.get('panel_time_var')
    
    imports = "import pandas as pd\nimport numpy as np\nimport statsmodels.api as sm\n"
    imports += "from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects\n"
    imports += "from scipy.stats import chi2\n"

    model_logic = f"""
# --- 3. Setup Panel Data ---
# Ensure MultiIndex is set correctly
data = data.set_index(['{id_var}', '{time_var}'])
Y = data['{y_var}']
X_vars_only = data[{x_vars}]
X_with_const = sm.add_constant(X_vars_only, has_constant='raise')

# --- 4. Fit Models ---

# A. Pooled OLS
model_pooled = PooledOLS(Y, X_with_const)
results_pooled = model_pooled.fit(cov_type='robust')
print("--- Pooled OLS Results ---")
print(results_pooled.summary)

# B. Fixed Effects (FE)
model_fe = PanelOLS(Y, X_vars_only, entity_effects=True)
results_fe = model_fe.fit(cov_type='robust')
print("\\n--- Fixed Effects (FE) Results ---")
print(results_fe.summary)

# C. Random Effects (RE)
model_re = RandomEffects(Y, X_with_const)
results_re = model_re.fit(cov_type='robust')
print("\\n--- Random Effects (RE) Results ---")
print(results_re.summary)

# --- 5. Hausman Test (Manual Calculation) ---
# Note: We calculate this manually to be robust across library versions.
print("\\n--- Hausman Test (FE vs RE) ---")

try:
    # 1. Get Coefficients and Covariance for common variables
    common_vars = [v for v in results_fe.params.index if v in results_re.params.index]
    
    b_fe = results_fe.params[common_vars]
    b_re = results_re.params[common_vars]
    v_fe = results_fe.cov.loc[common_vars, common_vars]
    v_re = results_re.cov.loc[common_vars, common_vars]
    
    # 2. Calculate Difference
    b_diff = b_fe - b_re
    v_diff = v_fe - v_re
    
    # 3. Calculate Statistic: (b_diff)' * inv(v_diff) * (b_diff)
    # We use pseudo-inverse (pinv) for stability
    hausman_stat = b_diff.T @ np.linalg.pinv(v_diff) @ b_diff
    
    # 4. Calculate P-Value
    df = len(common_vars)
    p_value = chi2.sf(hausman_stat, df)
    
    print(f"Hausman Statistic: {{hausman_stat:.4f}}")
    print(f"P-Value:           {{p_value:.4f}}")
    
    if p_value <= 0.05:
        print("Result: Reject H0. Fixed Effects (FE) is consistent and preferred.")
    else:
        print("Result: Fail to Reject H0. Random Effects (RE) is efficient and preferred.")

except Exception as e:
    print(f"Could not calculate Hausman test: {{e}}")
"""
    return imports, model_logic


def _get_default_code(model_id, params):
    """(Fallback)"""
    imports = "import pandas as pd\n"
    model_logic = f"""
# --- Model: {model_id} ---
print("Code generation for {model_id} is not yet implemented.")
print("This model is available in the DataNomics platform.")
print("Original parameters used:")
print({params})
"""
    return imports, model_logic

def _get_plotting_code(model_id, params):
    """
    يولد كود الرسم البياني (Visualization) بناءً على نوع النموذج.
    """
    # نحاول معرفة اسم المتغير لطباعته في العنوان
    # نستخدم Y.name في الكود المولد ليكون ديناميكياً
    
    plot_logic = f"""
# --- 7. Visualization (Actual vs Fitted) ---
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
"""

    # 1. حالة النماذج القياسية (OLS, ARDL, ARIMA, Logit, etc.)
    if model_id in ['ols', 'ardl', 'logit', 'probit', 'arima']:
        plot_logic += "plt.plot(Y.values, label='Actual', color='blue')\n"
        plot_logic += "plt.plot(results.fittedvalues.values, label='Fitted', color='red', linestyle='--')\n"
        
        # (!!!) هنا التريك: نستخدم f داخلية و f خارجية، ونضاعف الأقواس {{ }} للمتغير الذي لا نريد تعويضه الآن
        plot_logic += f"plt.title(f'Actual vs Fitted for {{Y.name}} ({model_id.upper()})')\n"

    # 2. حالة GARCH (نرسم التقلبات Volatility)
    elif model_id == 'garch':
        plot_logic += "plt.plot(results.conditional_volatility, label='Conditional Volatility', color='red', linestyle='--')\n"
        plot_logic += f"plt.title(f'GARCH Volatility for {{Y.name}}')\n"

    # 3. حالة نماذج الـ ML (Random Forest, XGBoost, Lasso...)
    elif model_id in ['random_forest', 'xgboost', 'lasso', 'ridge', 'elastic_net']:
        plot_logic += """
# For ML models, we plot the predictions on the Test Set
plt.plot(Y_test.values, label='Actual (Test)', color='blue')
plt.plot(Y_pred, label='Predicted (Test)', color='green', linestyle='--')
"""
        plot_logic += f"plt.title(f'Actual vs Predicted ({model_id.upper()})')\n"

    # 4. حالة VAR/VECM (رسم جاهز من المكتبة)
    elif model_id in ['var', 'vecm']:
        plot_logic = """
# --- 7. Visualization ---
import matplotlib.pyplot as plt
# VAR/VECM models have built-in plotting functions
results.plot()
plt.show()
"""
        return plot_logic # خروج مبكر لأننا لا نحتاج كود البواقي المخصص بالأسفل

    # تكملة الكود للأنواع 1, 2, 3
    plot_logic += """
plt.legend()
plt.grid(True)
plt.show()

# --- 8. Residuals Plot ---
# (Visual Check for Homoscedasticity)
if hasattr(results, 'resid'):
    plt.figure(figsize=(10, 4))
    plt.plot(results.resid, label='Residuals', color='grey')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residuals over Time')
    plt.legend()
    plt.show()
"""
    return plot_logic
# --- الدالة الرئيسية (معدلة ومصححة) ---
def generate_code_snippet(model_id, params, df, transformation_history=None):
    """
    (معدلة)
    Generates a runnable Python code snippet for the specified model,
    including data preparation steps.
    """
    
    # --- (!!!) (هذا هو الإصلاح الذكي) (!!!) ---
    
    # 1. تحديد المتغيرات المطلوبة للنموذج (من البارامترات)
    required_cols = set()
    if model_id == 'double_ml':
        required_cols.add(params.get('dependent_var'))
        required_cols.add(params.get('treatment_var'))
        required_cols.update(params.get('control_vars', []))
            
    elif model_id in ['var', 'vecm']:
        required_cols.update(params.get('variables', []))
        
    elif model_id in ['ols', 'lasso', 'ridge', 'elastic_net', 'logit', 'probit', 'random_forest', 'xgboost', 'panel']:
        required_cols.add(params.get('dependent_var'))
        required_cols.update(params.get('independent_vars', []))
        if model_id == 'panel':
             required_cols.add(params.get('panel_id_var'))
             required_cols.add(params.get('panel_time_var'))

    elif model_id in ['arima', 'garch', 'ardl']:
        required_cols.add(params.get('endog_var'))
        
        # --- (NEW FIX V2) ---
        # Explicitly check for None before updating
        exog_list = params.get('exog_vars')
        if exog_list:  # This is False if exog_list is None or []
            required_cols.update(exog_list)
        # --- (END NEW FIX V2) ---
    
    # 2. (جديد) استخراج المتغيرات "المصدر" من سجل التحويلات
    # هذا يحل مشكلة KeyError (مثل 'nfa' not found)
    source_vars = _extract_source_vars_from_history(transformation_history)
    required_cols.update(source_vars)
    
    # إزالة أي قيم None
    required_cols = [col for col in required_cols if col]
    
    # 3. إنشاء Data Snippet
    data_imports, data_snippet = _get_data_snippet(df, required_cols)
    # --- (!!!) (نهاية الإصلاح الذكي) (!!!) ---
    
    
    # 4. إنشاء كود تجهيز البيانات (الآن يدعم الوضع المزدوج)
    prep_imports, prep_code = _get_prep_steps_code(transformation_history)
    
    # 5. اختيار دالة الكود الصحيحة (مكتملة الآن)
    if model_id == 'ols':
        imports, model_logic = _get_ols_code(params)
    elif model_id in ['lasso', 'ridge', 'elastic_net']:
        imports, model_logic = _get_penalized_code(model_id, params)
    elif model_id in ['logit', 'probit']:
        imports, model_logic = _get_logit_probit_code(model_id, params)
    elif model_id == 'arima':
        imports, model_logic = _get_arima_code(params)
    elif model_id == 'var':
        imports, model_logic = _get_var_code(params)
    elif model_id == 'vecm':
        imports, model_logic = _get_vecm_code(params)
    elif model_id in ['random_forest', 'xgboost']:
        imports, model_logic = _get_ml_code(model_id, params)
    elif model_id == 'double_ml':
        imports, model_logic = _get_double_ml_code(params)
    elif model_id == 'garch':
        imports, model_logic = _get_garch_code(params)
    elif model_id == 'ardl':
        imports, model_logic = _get_ardl_code(params)
    elif model_id == 'panel':
        imports, model_logic = _get_panel_code(params)
    else:
        # Fallback (in case a model ID is missed)
        imports, model_logic = _get_default_code(model_id, params)
        
    # 6. تجميع الكود النهائي
    all_imports = set(imports.split('\n'))
    all_imports.update(data_imports.split('\n'))
    all_imports.update(prep_imports.split('\n'))
    all_imports = {imp for imp in all_imports if imp.strip()} 
    
    final_imports = "\n".join(sorted(list(all_imports)))
    
    final_code = final_imports + "\n\n" + data_snippet + "\n" + prep_code + "\n" + model_logic
    # ... (في نهاية ملف utils_code_generator.py)

    # 5. اختيار دالة الكود الصحيحة
    # ... (الكود الموجود عندك حالياً) ...

    # ... (بعد اختيار model_logic)

    # إضافة كود الرسم
    plotting_code = _get_plotting_code(model_id, params)
    
    # تجميع الـ imports
    all_imports = set(imports.split('\n'))
    all_imports.update(data_imports.split('\n'))
    all_imports.update(prep_imports.split('\n'))
    
    if "matplotlib" in plotting_code:
        all_imports.add("import matplotlib.pyplot as plt")

    all_imports = {imp for imp in all_imports if imp.strip()} 
    final_imports = "\n".join(sorted(list(all_imports)))
    
    # التجميع النهائي
    final_code = final_imports + "\n\n" + data_snippet + "\n" + prep_code + "\n" + model_logic + "\n" + plotting_code
    
    return final_code