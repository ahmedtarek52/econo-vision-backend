# app/blueprints/dashboard/routes.py
from flask import Blueprint, request, jsonify, current_app # <-- Import current_app if needed for logging/config
import pandas as pd
import numpy as np # <-- Import numpy

# --- (!!!) (إضافة جديدة: استيراد الحارس) (!!!) ---
# (استخدمنا ".." مرتين للرجوع من 'dashboard' إلى 'blueprints' ثم إلى 'app')
from ...decorators import login_required
# --- (!!!) (نهاية الإضافة) (!!!) ---

# --- (تعديل) استيراد الدوال الجديدة ---
from .utils import (
    calculate_summary_statistics,
    calculate_correlation_matrix,
    calculate_histogram_data,
    calculate_acf_pacf
    # Import pairplot function later when implemented
    # from .utils import generate_pair_plot_data
)
# --- (نهاية التعديل) ---

# --- (تعديل) تغيير اسم الـ Blueprint ليكون أوضح ---
# dashboard_bp = Blueprint('dashboard_bp', __name__)
dashboard_bp = Blueprint('dashboard_api', __name__, url_prefix='/api/dashboard') # Added url_prefix
# --- (نهاية التعديل) ---


# --- Helper Function for Error Handling ---
def handle_error(e, default_message="An error occurred", status_code=500):
    """Logs the error and returns a JSON response."""
    error_message = str(e)
    # Log the full error for debugging (using current_app.logger if configured)
    try:
        current_app.logger.error(f"Dashboard API Error: {error_message}", exc_info=True)
    except: # Fallback if logger isn't set up
        print(f"Dashboard API Error: {error_message}")
        import traceback
        traceback.print_exc()

    # Return a user-friendly message
    user_message = default_message
    if isinstance(e, ValueError): # Pass specific validation errors
        user_message = error_message
        status_code = 400 # Bad request for validation errors

    return jsonify({"error": user_message}), status_code
# --- End Helper Function ---


@dashboard_bp.route('/summary', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_summary():
    """Endpoint to get summary statistics for numeric columns."""
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload:
            return jsonify({"error": "Missing 'dataset' in request payload"}), 400

        dataset = payload['dataset']
        # Basic validation: check if dataset is a list of objects (rows)
        if not isinstance(dataset, list) or (len(dataset) > 0 and not isinstance(dataset[0], dict)):
             return jsonify({"error": "'dataset' must be a list of objects (rows)."}), 400
        if not dataset: # Handle empty dataset
             return jsonify({"summary": {}, "message": "Dataset is empty."}), 200


        df = pd.DataFrame(dataset)

        # Generate summary stats
        summary = calculate_summary_statistics(df)

        return jsonify({
            "message": "Summary statistics generated successfully.",
            "summary": summary
        }), 200

    except Exception as e:
        return handle_error(e, "Failed to generate summary statistics")


# --- (إضافة جديدة) Endpoint لمصفوفة الارتباط ---
@dashboard_bp.route('/correlation', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_correlation_matrix():
    """Endpoint to calculate and return the correlation matrix."""
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload:
            return jsonify({"error": "Missing 'dataset' in request payload"}), 400

        dataset = payload['dataset']
        if not isinstance(dataset, list):
            return jsonify({"error": "'dataset' must be a list."}), 400
        if not dataset:
             return jsonify({"correlation_matrix": {}, "message": "Dataset is empty."}), 200


        df = pd.DataFrame(dataset)

        # Calculate correlation matrix using the utility function
        correlation_matrix = calculate_correlation_matrix(df)

        return jsonify({
            "message": "Correlation matrix calculated successfully.",
            "correlation_matrix": correlation_matrix
        }), 200

    except Exception as e:
         return handle_error(e, "Failed to calculate correlation matrix")
# --- (نهاية الإضافة) ---


# --- (إضافة جديدة) Endpoint لبيانات الهيستوجرام ---
@dashboard_bp.route('/histogram', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_histogram_data():
    """Endpoint to get data for plotting a histogram."""
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'variable' not in payload:
            return jsonify({"error": "Missing 'dataset' or 'variable' in request payload"}), 400

        dataset = payload['dataset']
        variable = payload['variable']
        bins = payload.get('bins', 'auto') # Get bins from request, default to 'auto'

        if not isinstance(dataset, list):
            return jsonify({"error": "'dataset' must be a list."}), 400
        if not dataset:
             return jsonify({"histogram_data": {'bins':[], 'counts':[]}, "message": "Dataset is empty."}), 200

        df = pd.DataFrame(dataset)

        # Calculate histogram data using the utility function
        histogram_data = calculate_histogram_data(df, variable, bins)

        return jsonify({
            "message": f"Histogram data for '{variable}' generated successfully.",
            "histogram_data": histogram_data
        }), 200

    except ValueError as ve: # Catch specific errors from the util function
        return handle_error(ve) # Returns 400 status code
    except Exception as e:
        return handle_error(e, f"Failed to generate histogram data for '{payload.get('variable', 'N/A')}'")
# --- (نهاية الإضافة) ---


# --- (إضافة جديدة) Endpoint لبيانات ACF/PACF ---
@dashboard_bp.route('/acf', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_acf_pacf_data():
    """Endpoint to get ACF and PACF data for a time series variable."""
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload or 'variable' not in payload:
            return jsonify({"error": "Missing 'dataset' or 'variable' in request payload"}), 400

        dataset = payload['dataset']
        variable = payload['variable']
        lags = payload.get('lags', 20) # Default to 20 lags
        alpha = payload.get('alpha', 0.05) # Default confidence level

        if not isinstance(dataset, list):
             return jsonify({"error": "'dataset' must be a list."}), 400
        if not dataset:
             # Need to decide what to return for empty dataset ACF/PACF
             return jsonify({"acf_data": None, "pacf_data": None, "message": "Dataset is empty."}), 200


        df = pd.DataFrame(dataset)

        # Calculate ACF/PACF using the utility function
        acf_pacf_results = calculate_acf_pacf(df, variable, lags=lags, alpha=alpha)

        return jsonify({
            "message": f"ACF/PACF data for '{variable}' generated successfully.",
            "acf_data": acf_pacf_results.get('acf_data'),
            "pacf_data": acf_pacf_results.get('pacf_data')
        }), 200

    except ValueError as ve:
        return handle_error(ve)
    except Exception as e:
         return handle_error(e, f"Failed to generate ACF/PACF data for '{payload.get('variable', 'N/A')}'")
# --- (نهاية الإضافة) ---


# --- (إضافة جديدة) Endpoint للـ Pair Plot (مبدئي) ---
@dashboard_bp.route('/pairplot', methods=['POST'])
@login_required # (!!!) (تمت إضافة الحارس) (!!!)
def get_pair_plot():
    """
    Endpoint to generate and return a pair plot image URL.
    NOTE: This is a placeholder. Actual implementation requires
    a plotting library (seaborn, matplotlib), saving the image
    (e.g., to cloud storage or static folder), and returning the URL.
    """
    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload:
            return jsonify({"error": "Missing 'dataset' in request payload"}), 400

        dataset = payload['dataset']
        if not isinstance(dataset, list) or not dataset:
            return jsonify({"error": "'dataset' must be a non-empty list."}), 400

        df = pd.DataFrame(dataset)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
             return jsonify({"error": "Pair plot requires at least two numeric variables."}), 400
        if len(numeric_cols) > 10: # Limit complexity
             return jsonify({"error": "Pair plot generation is limited to a maximum of 810 numeric variables for performance."}), 400


        # --- Placeholder Logic ---
        # 1. Use seaborn.pairplot(df[numeric_cols])
        # 2. Save the plot to a BytesIO buffer: fig = plot.get_figure(); buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0);
        # 3. Upload buffer to cloud storage (like AWS S3, Google Cloud Storage, or Fly.io volumes) OR save to a static folder accessible by URL.
        # 4. Get the public URL of the saved image.
        # For now, we'll return a placeholder image URL.
        placeholder_url = f"https://placehold.co/600x400/EEE/31343C?text=Pair+Plot+{len(numeric_cols)}vars"
        # In a real scenario, replace placeholder_url with the actual generated image URL

        return jsonify({
            "message": "Pair plot generated (placeholder).",
            "pair_plot_url": placeholder_url # Return the URL
        }), 200
        # --- End Placeholder ---

    except Exception as e:
        return handle_error(e, "Failed to generate pair plot")
# --- (نهاية الإضافة) ---