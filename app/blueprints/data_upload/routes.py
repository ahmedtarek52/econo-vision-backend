from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np  # <-- Make sure to import numpy
from .utils import get_file_extension, suggest_variable_types, check_data_quality

data_upload_bp = Blueprint('data_upload_bp', __name__)

@data_upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = get_file_extension(filename)
            if ext == 'csv':
                df = pd.read_csv(filepath)
            elif ext in ['xls', 'xlsx']:
                df = pd.read_excel(filepath)
            elif ext == 'json':
                df = pd.read_json(filepath)
            else:
                os.remove(filepath)
                return jsonify({"error": f"Unsupported file format: .{ext}"}), 400
            
            quality_report = check_data_quality(df)
            suggested_types = suggest_variable_types(df)
            
            # --- FIX: Replace numpy.NaN with Python's None ---
            # This is a robust way to ensure all NaN values are converted before creating the JSON.
            df_cleaned_for_json = df.replace({np.nan: None})

            response = {
                "message": "File processed successfully!",
                "filename": filename,
                "columns": list(df.columns),
                "previewData": df_cleaned_for_json.head().to_dict(orient='records'),
                "suggestedTypes": suggested_types,
                "fullDataset": df_cleaned_for_json.to_dict(orient='records'),
                "qualityReport": quality_report
            }
            return jsonify(response), 200

        except Exception as e:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500