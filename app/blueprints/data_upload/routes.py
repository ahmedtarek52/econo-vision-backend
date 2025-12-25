from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import pandas as pd
import io

# --- (!!!) (1. الإصلاح الأمني: استيراد الحارس) (!!!) ---
# (المسار الصحيح الذي تأكدنا منه)
from app.decorators import login_required
# --- (!!!) (نهاية الإضافة) (!!!) ---

# (جديد) استيراد الدوال المحدثة
from .utils import (
    get_file_extension,
    read_dataframe_from_file,
    post_process_dataframe
)

# --- (!!!) (2. إصلاح 404: تم حذف url_prefix من هنا) (!!!) ---
# (لأننا نعرفه في app/__init__.py)
data_upload_bp = Blueprint('data_upload_bp', __name__)
# --- (!!!) (نهاية التعديل) (!!!) ---


def handle_error(e, status_code=500):
    """Helper function to return a standardized error JSON."""
    error_message = str(e)
    if isinstance(e, ValueError):
        status_code = 400 # Bad Request for validation errors
    
    print(f"Error: {error_message}")
    import traceback
    traceback.print_exc()
    
    return jsonify({"error": error_message, "status": "error"}), status_code

# --- (!!!) (3. الإصلاح الأمني: تأمين المسار) (!!!) ---
@data_upload_bp.route('/upload', methods=['POST']) 
@login_required # <-- تم إضافة الحارس
def upload_file():
    if 'file' not in request.files:
        return handle_error(ValueError("No file part in the request"))
    
    file = request.files['file']
    if file.filename == '':
        return handle_error(ValueError("No file selected"))

    try:
        filename = secure_filename(file.filename)
        file_ext = get_file_extension(filename)
        
        if file_ext not in ['csv', 'xlsx', 'xls', 'json']:
                raise ValueError(f"Unsupported file format: '.{file_ext}'. Please use CSV, Excel, or JSON.")

        sheet_name_to_read = None

        # (!!!) هذا هو المنطق الخاص بـ Excel (!!!)
        if file_ext in ['xlsx', 'xls']:
            file.seek(0) # Rewind stream before reading
            try:
                # استخدم pd.ExcelFile لفتح الملف دون قراءته بالكامل
                xls = pd.ExcelFile(file)
                sheet_names = xls.sheet_names
                
                if len(sheet_names) > 1:
                    # إذا كان هناك عدة "شيتات"، اطلب من المستخدم الاختيار
                    print(f"File has multiple sheets: {sheet_names}")
                    return jsonify({
                        "status": "sheet_selection_required",
                        "sheet_names": sheet_names
                    }), 200
                elif len(sheet_names) == 1:
                    # إذا كان هناك "شيت" واحد، قم بتعيينه للقراءة
                    sheet_name_to_read = sheet_names[0]
                    
            except Exception as e:
                # (معالجة الملفات التالفة)
                return handle_error(ValueError(f"Could not read Excel file. It may be corrupted or password-protected. Error: {e}"))
            finally:
                file.seek(0) # (مهم) أعد الملف إلى البداية للقراءة الفعلية
        
        # --- القراءة والمعالجة ---
        # (CSV/JSON/Excel-with-one-sheet سيصلون إلى هنا)
        
        # 1. قراءة البيانات
        df = read_dataframe_from_file(file, file_ext, sheet_name=sheet_name_to_read)
        
        # 2. معالجة البيانات وإعداد الرد
        response_data = post_process_dataframe(df, filename)
        
        return jsonify(response_data), 200

    except Exception as e:
        return handle_error(e)

# --- (!!!) (4. الإصلاح الأمني: تأمين المسار) (!!!) ---
@data_upload_bp.route('/upload-sheet', methods=['POST'])
@login_required # <-- تم إضافة الحارس
def upload_sheet():
    if 'file' not in request.files or 'sheetName' not in request.form:
        return handle_error(ValueError("Missing 'file' or 'sheetName' in sheet upload request."))

    file = request.files['file']
    sheet_name = request.form['sheetName']
    
    try:
        filename = secure_filename(file.filename)
        file_ext = get_file_extension(filename)

        if file_ext not in ['xlsx', 'xls']:
                raise ValueError("Sheet selection is only for Excel files.")
        
        # 1. قراءة "الشيت" المحدد
        df = read_dataframe_from_file(file, file_ext, sheet_name=sheet_name)
        
        # 2. معالجة البيانات وإعداد الرد (نفس الدالة المستخدمة سابقاً)
        response_data = post_process_dataframe(df, filename)
        
        # أضف اسم الشيت للرسالة لمزيد من الوضوح
        response_data["message"] = f"Sheet '{sheet_name}' processed successfully!"
        
        return jsonify(response_data), 200

    except Exception as e:
        return handle_error(e)