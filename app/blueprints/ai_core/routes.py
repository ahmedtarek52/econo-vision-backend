import os
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google.generativeai package not installed. AI chat features are disabled.")

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
model = None
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if genai is None:
        print("Warning: google.generativeai is not available, Gemini AI Core disabled.")
    elif not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY environment variable not set.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("Gemini AI Core (gemini-2.0-flash) initialized successfully.")
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


def validate_dataset_payload(payload):
    """Validate and normalize the incoming dataset payload."""
    if not payload or 'dataset' not in payload:
        raise ValueError("Missing 'dataset' in request payload.")

    dataset = payload['dataset']

    if isinstance(dataset, dict):
        if not dataset:
            raise ValueError("'dataset' must be a non-empty list or dict.")
        return pd.DataFrame(dataset)

    if isinstance(dataset, list):
        if not dataset:
            raise ValueError("'dataset' must be a non-empty list.")

        # Accept both list-of-objects and list-of-values.
        return pd.DataFrame(dataset)

    raise ValueError("'dataset' must be a list of row objects or a dictionary of columns.")

SYSTEM_PROMPT = """
أنت "مساعد DataNomics"، الخبير والجاهز لمساعدة المستخدمين في منصة DataNomics للتحليل الاقتصادي القياسي.
إليك إرشاداتك ومعلومات المنصة التي يجب أن تستخدمها لإجابة المستخدم بذكاء واحترافية:

1. أسلوب الرد:
- أجب بلغة عربية بسيطة، مهنية، ومشجعة.
- اجعل ردودك مختصرة ومباشرة (في حدود 3-5 جمل) ما لم يطلب المستخدم تفاصيل إضافية.
- ركز على تقديم خطوات عملية واضحة.

2. هيكل المنصة وأقسامها لمساعدة المستخدم في التنقل:
- صفحة Upload (الرئيسية): لرفع ملفات البيانات (Excel أو CSV).
- صفحة Dashboard: لعرض الإحصاءات الوصفية والرسوم البيانية للبيانات المرفوعة.
- صفحة Data Preparation: لتنظيف البيانات، التعامل مع القيم المفقودة، وعمل التحويلات (مثل اللوغاريتم والفروق الأولى).
- صفحة Stability Tests: لإجراء اختبارات الاستقرارية وجذور الوحدة (ADF, KPSS) واختبارات التكامل المشترك (Cointegration).
- صفحة Models & Analysis: لتقدير النماذج الاقتصادية (مثل الانحدار الخطي OLS، نماذج البيانات الطولية Panel Data كالأثر الثابت والعشوائي، إلخ).
- صفحة AI Reports: لإنشاء وتنزيل تقارير ذكاء اصطناعي شاملة وملخصة للتحليلات والاختبارات التي تمت.
- صفحة User Guide: دليل شامل يشرح كيفية استخدام كل أداة بالتفصيل.

استخدم هذا السياق دائماً لتوجيه المستخدمين خطوة بخطوة عند سؤالهم عن كيفية القيام بشيء ما داخل المنصة.
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
        df = validate_dataset_payload(payload)

        # تحويل البيانات الرقمية بشكل آمن لكل عمود على حدة لتجنب مشاكل المعاملات في apply
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except Exception:
                pass

        if df.empty:
            raise ValueError("Dataset is empty after parsing. Please provide a non-empty dataset.")

        assessment_results = run_initial_data_assessment(df)
        final_briefing = synthesize_briefing_recommendations(assessment_results)

        return jsonify(final_briefing), 200

    except Exception as e:
        return handle_error(e, "An unexpected server error occurred during AI briefing")

@ai_core_bp.route('/describe-upload', methods=['POST', 'OPTIONS'])
@login_required
def describe_upload():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    if model is None:
        return handle_error(ValueError("Gemini AI model is not initialized. Check API Key."), 503)
        
    try:
        data = request.get_json() or {}
        columns = data.get('columns', [])
        num_rows = data.get('numRows')
        num_cols = data.get('numCols')
        suggested_types = data.get('suggestedTypes', {})
        
        prompt = f"""
You are an expert economic data analyst. A user has just uploaded a dataset with the following properties:
- Dimensions: {num_rows} rows x {num_cols} columns
- Columns: {", ".join(columns)}
- Suggested Column Types: {suggested_types}

Write a professional, welcoming initial assessment of this dataset.
Identify:
1. What type of economic data it appears to be (Time Series, Cross-Sectional, Panel, etc.).
2. The key target variable and potential policy/exogenous variables.
3. Recommendations for what pre-estimation diagnostics the user should perform (e.g. Stationarity/Unit Root tests, Multicollinearity check).

Keep it professional, highly motivating, and structured using clean markdown. Keep it concise (around 150-200 words).
You can reply in Arabic (preferred) or English.
"""
        response = model.generate_content(prompt)
        return jsonify({"description": response.text}), 200
    except Exception as e:
        return handle_error(e, "Failed to generate dataset description")

@ai_core_bp.route('/interpret-test', methods=['POST', 'OPTIONS'])
@login_required
def interpret_test():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    if model is None:
        return handle_error(ValueError("Gemini AI model is not initialized. Check API Key."), 503)
        
    try:
        data = request.get_json() or {}
        test_id = data.get('testId')
        params = data.get('params', {})
        results = data.get('results', {})
        
        prompt = f"""
You are an expert econometrics professor. A student has run a pre-estimation diagnostic test:
- Test Name: {test_id}
- Configuration Parameters: {params}
- Raw Test Results: {results}

Provide a professional, clear interpretation of these test results.
Explain:
1. The null hypothesis of the test.
2. Whether the null hypothesis is rejected or failed to be rejected based on the results/p-values.
3. What this means in plain Arabic (and English context if needed) for the economic data (e.g., is the variable stationary, is there cointegration, is there Granger causality, or is there multicollinearity).
4. Direct, actionable next steps for their econometric model.

Keep it highly authoritative and educational, formatting it nicely in markdown. Do not exceed 250 words.
"""
        response = model.generate_content(prompt)
        return jsonify({"interpretation": response.text}), 200
    except Exception as e:
        return handle_error(e, "Failed to interpret test results")