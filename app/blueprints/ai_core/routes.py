import os
import google.generativeai as genai
from flask import Blueprint, request, jsonify, current_app
import pandas as pd 
from app.decorators import login_required 

# محاولة استيراد دوال المساعدة بشكل آمن لتجنب توقف التطبيق
try:
    from .utils import (
        run_initial_data_assessment,
        synthesize_briefing_recommendations,
        get_stationarity_recommendation,
        get_cointegration_recommendation,
        get_model_diagnostics_recommendations,
        get_panel_model_decision
    )
except ImportError as e:
    print(f"Warning: Could not import AI utils: {e}")
    # دوال وهمية في حال الفشل
    def run_initial_data_assessment(*args): return {}
    def synthesize_briefing_recommendations(*args): return {"error": "AI Utils Failed"}

# (!!!) تعديل: إزالة url_prefix من هنا لتركه للملف الرئيسي (!!!)
ai_core_bp = Blueprint('ai_core_api', __name__)

# --- إعداد Gemini API ---
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set.")
        model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        print("Gemini AI Core (gemini-1.5-flash-latest) initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini AI Core: {e}")
    model = None

def handle_error(e, default_message="An error occurred", status_code=500):
    """Logs the error and returns a JSON response."""
    error_message = str(e)
    try:
        current_app.logger.error(f"AI Core Error: {error_message}", exc_info=True)
    except Exception:
        print(f"AI Core Error: {error_message}")
    
    user_message = default_message
    if isinstance(e, (ValueError, KeyError, TypeError)):
        user_message = error_message
        status_code = 400
    elif "API_KEY" in error_message:
        user_message = "AI service is not configured (API Key error)."
        status_code = 503
    
    return jsonify({"error": user_message}), status_code

SYSTEM_PROMPT = """
أنت "مساعد DataNomics"، وهو مرشد ذكاء اصطناعي خبير في منصة "DataNomics".
هدفك هو مساعدة المستخدم على فهم المفاهيم، وتفسير النتائج.
كن مختصراً (3-4 جمل) واستخدم لغة عربية بسيطة.
"""

@ai_core_bp.route('/chat', methods=['POST', 'OPTIONS'])
@login_required 
def handle_chat():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if model is None:
        return handle_error(ValueError("Gemini AI model is not initialized. Check API Key."), 503)

    try:
        data = request.get_json()
        if not data or 'userMessage' not in data:
            raise ValueError("Missing 'userMessage'.")

        user_message = data.get('userMessage')
        context = data.get('context', {}) 
        
        context_text = "Current User Context:\n"
        if context.get('currentLocation'):
            context_text += f"- Page: {context['currentLocation'].get('pathname')}\n"
        
        full_prompt = f"{SYSTEM_PROMPT}\n\n{context_text}\n\nUser Question: {user_message}"
        
        try:
            response = model.generate_content(full_prompt)
            bot_text = response.text
        except Exception as api_e:
            print(f"Gemini API Error: {api_e}")
            bot_text = "حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي."

        return jsonify({"text": bot_text})

    except Exception as e:
        return handle_error(e)

@ai_core_bp.route('/generate-briefing', methods=['POST', 'OPTIONS'])
@login_required 
def generate_initial_briefing():
    """
    Runs pre-tests (ADF, KPSS, VIF) and synthesizes a recommendation.
    """
    # السماح لطلبات Preflight
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        payload = request.get_json()
        if not payload or 'dataset' not in payload:
            raise ValueError("Missing 'dataset' in request payload.")
        
        dataset = payload['dataset']
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("'dataset' must be a non-empty list.")
        
        df = pd.DataFrame(dataset)
        # تحويل البيانات الرقمية
        df = df.apply(pd.to_numeric, errors='ignore')
        
        assessment_results = run_initial_data_assessment(df)
        final_briefing = synthesize_briefing_recommendations(assessment_results)
        
        return jsonify(final_briefing), 200

    except Exception as e:
        return handle_error(e, "An unexpected server error occurred during AI briefing")