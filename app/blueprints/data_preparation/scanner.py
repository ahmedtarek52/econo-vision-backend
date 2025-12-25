import pandas as pd
import numpy as np
import re
import warnings

def scan_dataset_health(df, manual_date_col=None):
    """
    فحص ذكي بتسلسل صارم للأولويات لمنع تداخل الأنواع.
    """
    health_report = {}
    total_rows = len(df)
    
    # 1. الحسابات الأولية (القيم المفقودة)
    for col in df.columns:
        col_data = df[col]
        report = {
            'detected_type': 'unknown',
            'quality_score': 100,
            'issues': [],
            'sample_errors': [], 
            'recommendation': 'none',
            'null_count': int(col_data.isnull().sum()),
            'total_count': total_rows
        }
        
        if report['null_count'] > 0:
            report['issues'].append('missing_values')
            report['quality_score'] -= (report['null_count'] / total_rows) * 100
            
            if col_data.dtype in [np.dtype('float64'), np.dtype('int64')] or col_data.astype(str).str.isnumeric().sum() / total_rows > 0.5:
                report['recommendation'] = 'fill_mean'
            else:
                report['recommendation'] = 'drop_missing_rows'
        
        health_report[col] = report

    # 2. التشخيص العميق (التسلسل الصارم)
    for col in df.columns:
        report = health_report[col]
        
        # تحويل البيانات لنصوص للفحص
        str_data = df[col].astype(str)
        
        # المؤشرات الحاسمة
        # نستخدم الهروب \ للرموز الخاصة لضمان عمل الـ Regex بشكل صحيح
        has_currency = str_data.str.contains(r'[\$\€\£]', regex=True).any()
        
        # (!!!) التعديل: كتم تحذير الـ Regex الخاص بـ Groups لتجنب UserWarning في الكونسول (!!!)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_dirty_text = str_data.str.contains(r'(Error|N/A|High|Low|Unknown|null)', flags=re.IGNORECASE, regex=True).any()
        
        # تنظيف مبدئي لمحاولة التحويل الرقمي
        temp_clean = str_data.str.replace(r'[\$\,\%]', '', regex=True)
        numeric_conv = pd.to_numeric(temp_clean, errors='coerce')
        num_nans = numeric_conv.isnull().sum()
        numeric_ratio = (total_rows - num_nans) / total_rows
        is_pure_numeric = (num_nans == report['null_count'])

        # --- [المرحلة 1]: الأولوية القصوى لعمود التاريخ اليدوي ---
        if col == manual_date_col:
            report['detected_type'] = 'datetime' # نفترضه تاريخاً لأنه تحدد يدوياً
            # نفحص التنسيق
            date_conv = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            date_nans = date_conv.isnull().sum()
            bad_dates = date_nans - report['null_count']
            
            if bad_dates > 0:
                report['detected_type'] = 'datetime_mixed'
                report['issues'].append('mixed_date_formats')
                report['recommendation'] = 'standardize_date'
                report['quality_score'] -= (bad_dates / total_rows) * 100
                
                bad_indices = date_conv.isnull() & df[col].notnull()
                report['sample_errors'] = df.loc[bad_indices, col].head(5).astype(str).tolist()
            
            # **توقف هنا لهذا العمود**
            continue

        # --- [المرحلة 2]: الأولوية للأرقام الملوثة (تمنع تشخيصها كتواريخ) ---
        # إذا كان يحتوي على عملة، أو نصوص خطأ، أو أنه رقمي بنسبة كبيرة (>70%) ولكنه ليس رقمياً صافياً
        elif has_currency or has_dirty_text or (numeric_ratio > 0.7 and not is_pure_numeric):
            report['detected_type'] = 'numeric_mixed'
            report['issues'].append('mixed_types')
            
            if has_currency:
                report['recommendation'] = 'clean_currency_and_coerce'
            else:
                report['recommendation'] = 'coerce_to_nan'
            
            report['quality_score'] -= ((num_nans - report['null_count']) / total_rows) * 100
            
            # استخراج الأخطاء
            bad_indices = numeric_conv.isnull() & df[col].notnull()
            report['sample_errors'] = df.loc[bad_indices, col].head(5).astype(str).tolist()
            
            # فحص القيم المتطرفة (Outliers) للأرقام التي نجح تحويلها
            series = numeric_conv.dropna()
            if len(series) > 5:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((series < (Q1 - 2.5*IQR)) | (series > (Q3 + 2.5*IQR)))
                if outliers.sum() > 0:
                    if 'outliers_detected' not in report['issues']: report['issues'].append('outliers_detected')
                    # نضيف عينة من القيم المتطرفة
                    outlier_vals = df.loc[series[outliers].index, col].head(3).astype(str).tolist()
                    for v in outlier_vals:
                        if v not in report['sample_errors']: report['sample_errors'].append(v)
                    
                    if report['recommendation'] == 'none': report['recommendation'] = 'handle_outliers'

            # **توقف هنا لهذا العمود**
            continue

        # --- [المرحلة 3]: الأرقام الصافية ---
        elif is_pure_numeric:
            report['detected_type'] = 'numeric'
            # فحص القيم المتطرفة فقط
            series = numeric_conv.dropna()
            if len(series) > 5:
                Q1 = series.quantile(0.25); Q3 = series.quantile(0.75); IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((series < (Q1 - 2.5*IQR)) | (series > (Q3 + 2.5*IQR)))
                    if outliers.sum() > 0:
                        report['issues'].append('outliers_detected')
                        report['sample_errors'] = df.loc[series[outliers].index, col].head(3).astype(str).tolist()
                        report['recommendation'] = 'handle_outliers'
            continue

        # --- [المرحلة 4]: التواريخ (فقط إذا لم يكن ما سبق) ---
        else:
            date_conv = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            date_nans = date_conv.isnull().sum()
            valid_date_ratio = (total_rows - date_nans) / total_rows
            
            if valid_date_ratio > 0.5: # إذا كان أكثر من نصفه تواريخ
                report['detected_type'] = 'datetime' if date_nans == report['null_count'] else 'datetime_mixed'
                if report['detected_type'] == 'datetime_mixed':
                    report['issues'].append('mixed_date_formats')
                    report['recommendation'] = 'standardize_date'
                    
                    bad_indices = date_conv.isnull() & df[col].notnull()
                    report['sample_errors'] = df.loc[bad_indices, col].head(5).astype(str).tolist()
            
            # --- [المرحلة 5]: نصوص / فئات ---
            else:
                unique_count = df[col].nunique()
                if unique_count < 20:
                    report['detected_type'] = 'categorical'
                    report['recommendation'] = 'convert_to_dummy' if unique_count < 10 else 'none'
                else:
                    report['detected_type'] = 'text'

        # تنظيف نهائي
        report['quality_score'] = round(max(0, report['quality_score']), 1)
        report['sample_errors'] = list(set(report['sample_errors']))

    return health_report


def apply_cleaning_fixes(df, cleaning_plan):
    """تطبيق الإصلاحات"""
    df_clean = df.copy()
    for col, strategy in cleaning_plan.items():
        if col not in df_clean.columns: continue
        
        if strategy == 'coerce_to_nan':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        elif strategy == 'extract_numbers':
            extracted = df_clean[col].astype(str).str.extract(r'(\-?\d+\.?\d*)')[0]
            df_clean[col] = pd.to_numeric(extracted, errors='coerce')
        elif strategy == 'clean_currency_and_coerce':
            cleaned = df_clean[col].astype(str).str.replace(r'[\$\,\%]', '', regex=True)
            df_clean[col] = pd.to_numeric(cleaned, errors='coerce')
        elif strategy == 'drop_rows':
            # نفحص النوع الحالي لتحديد طريقة كشف الخطأ
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                is_bad = df_clean[col].isna()
            else:
                # محاولة تحويل لمعرفة الخطأ
                is_bad = pd.to_numeric(df_clean[col], errors='coerce').isna() & df_clean[col].notnull()
            df_clean = df_clean[~is_bad]
        elif strategy == 'standardize_date':
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', dayfirst=True)
        elif strategy == 'fill_mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'fill_zero':
            df_clean[col] = df_clean[col].fillna(0)
        elif strategy == 'drop_missing_rows':
            df_clean = df_clean.dropna(subset=[col])
            
    return df_clean.reset_index(drop=True)