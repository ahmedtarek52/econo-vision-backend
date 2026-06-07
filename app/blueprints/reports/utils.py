import os
import requests
import json
import traceback 
import pandas as pd 
import numpy as np 

# app/blueprints/reports/utils.py

def generate_english_report(model_summary):
    """Generates a structured, presentation-ready English framework report."""
    summary_str = str(model_summary) if model_summary else "No model summary provided."
    report = f"""
## Econometric Analysis Report (Statistical Framework)

### 1. Executive Summary
This document provides a structured quantitative breakdown of the estimated econometric specification. The empirical framework below details the directional vectors, magnitude of impacts, and the underlying covariance structures parsed from the core dataset.

### 2. Model Specification and Results
The empirical estimation matrix across the configured statistical dimensions is formatted below:

{summary_str}

### 3. Comprehensive Framework Interpretation Guidance
- **Goodness-of-Fit Evaluation:** Review the reported R-squared parameters above. In high-dimensional panel setups or non-linear binarychoice models (McFadden Pseudo R2), values quantify the exact proportion of cross-sectional and temporal variance successfully captured by the exogenous regressors.
- **Statistical Significance Matrix:** Evaluate the individual coefficient t-statistics and corresponding p-values. Parameters exhibiting p-values below the critical threshold (typically 0.05 or 0.01) indicate a statistically significant structural relationship, rejecting the null hypothesis of zero impact.
- **Post-Estimation Diagnostic Controls:** Inspect the automatic diagnostics table. Tests for residual autocorrelation (e.g., Wooldridge or Durbin-Watson) and heteroskedasticity validate whether the standard errors are consistent and reliable for structural policy inference.

### 4. Methodological Framework & Structural Context
The generated estimations provide a rigorous quantitative baseline. All parsed coefficients must be interpreted in alignment with established economic theories (e.g., scale economies, structural gravitational gravity models) to ensure empirical regularities reflect logical institutional policies.

---
*Report framework compiled dynamically by DataNomics Platform.*
"""
    return report.strip()


def generate_arabic_report(model_summary):
    """Generates a structured, presentation-ready Arabic framework report."""
    summary_str = str(model_summary) if model_summary else "لم يتم تقديم ملخص للنموذج."
    report = f"""
## تقرير التحليل الاقتصادي القياسي (الإطار الهيكلي المستند إلى البيانات)

### ١. الملخص التنفيذي
يقدم هذا التقرير تفصيلاً كمياً منظماً للمواصفات القياسية المقدرة. يعرض الإطار التجريبي أدناه متجهات الاتجاه، وحجم التأثيرات الهيكلية، وهياكل التباين المشترك المستخرجة مباشرة من قاعدة البيانات.

### ٢. مواصفات النموذج والنتائج الإحصائية
مصفوفة نتائج التقدير الإحصائي عبر الأبعاد التي تم تكوينها موضحة بالكامل في الجدول التالي:

{summary_str}

### ٣. القواعد الأكاديمية لتفسير المخرجات الرقمية
- **تقييم جودة التوفيق (Goodness-of-Fit):** يرجى مراجعة معاملات الـ R-squared المذكورة في الجدول أعلاه. في نماذج البانل أو النماذج غير الخطية (McFadden Pseudo R2)، تقيس هذه القيم النسبة المئوية الدقيقة لتباين البيانات الذي نجحت المتغيرات المستقلة في تفسيره.
- **مصفوفة الدلالة الإحصائية (Statistical Significance):** يتم تقييم معنوية المتغيرات عبر النظر إلى قيم t-statistics والـ P-values المصاحبة لها. المعاملات التي تسجل قيم P-value أقل من مستويات المعنوية الحرجة (0.05 أو 0.01) تعني رفض فرضية العدم وثبوت الأثر الهيكلي للمتغير في سياق الاقتصاد الكلي.
- **صلاحية الاختبارات التشخيصية البعدية:** يجب فحص جدول الاختبارات التلقائية؛ حيث تضمن سلامة مخرجات اختبارات الارتباط الذاتي (Wooldridge) واختلاف التباين أن الأخطاء المعيارية وقيم الدلالة المقدرة غير متحيزة وصالحة تماماً للاستدلال الاستراتيجي.

### ٤. الآثار المنهجية وسياق السياسات العامة
توفر هذه التقديرات أساساً كمياً راسخاً لمتخذي القرار. يجب دائمًا ربط الإشارات الرياضية للمتغيرات (طردية أو عكسية) بنظريات الاقتصاد القياسي المستقرة لضمان تحويل الأرقام الصماء إلى سياسات ومبادرات مؤسسية قابلة للتطبيق.

---
*تم إنشاء هيكل التقرير ديناميكياً بواسطة منصة DataNomics.*
"""
    return report.strip()


def format_diagnostics_for_prompt(diagnostics_results=None, post_test_result=None):
    """Formats diagnostic test results into a readable string for the AI prompt."""
    formatted_string = ""
    found_diagnostics = False

    # Format automatic diagnostics (list of dicts)
    if isinstance(diagnostics_results, list) and diagnostics_results:
        formatted_string += "\n\n**Automatic Diagnostic Test Results:**\n"
        count = 0
        for test in diagnostics_results:
            if not isinstance(test, dict): continue

            name = test.get('name', 'Unnamed Test')
            stat = test.get('statistic', 'N/A')
            pval = test.get('p_value', 'N/A')
            interp = test.get('interpretation', '')

            try:
                pval_str = f"{float(pval):.4f}" if pval not in [None, 'N/A', ''] else 'N/A'
            except (ValueError, TypeError):
                pval_str = str(pval) 

            try:
                stat_str = f"{float(stat):.4f}" if stat not in [None, 'N/A', ''] else 'N/A'
            except (ValueError, TypeError):
                stat_str = str(stat) 

            formatted_string += f"- **{name}:** Stat={stat_str}, P-Value={pval_str}. *Interpretation:* {interp}\n"
            count += 1
        if count > 0: found_diagnostics = True

    # 🟢 تم التصحيح: فك التشابك وسحب الـ P-Value من كائن الـ post_test_result الصحيح 🟢
    if isinstance(post_test_result, dict) and post_test_result:
        formatted_string += "\n**Specific Post-Estimation Test Result:**\n"
        name = post_test_result.get('name', post_test_result.get('test_name', 'Specific Test'))
        stat = post_test_result.get('statistic', post_test_result.get('stat', 'N/A'))
        pval = post_test_result.get('p_value', 'N/A') # 👈 صلحنا البج هنا
        interp = post_test_result.get('interpretation', '')
        details = post_test_result.get('formatted_results', post_test_result.get('details', ''))
        html = post_test_result.get('html_table', '')

        try:
            pval_str = f"{float(pval):.4f}" if pval not in [None, 'N/A', ''] else 'N/A'
        except (ValueError, TypeError):
            pval_str = str(pval)

        try:
            stat_str = f"{float(stat):.4f}" if stat not in [None, 'N/A', ''] else 'N/A'
        except (ValueError, TypeError):
            stat_str = str(stat)

        formatted_string += f"- **{name}:**"
        if stat_str != 'N/A': formatted_string += f" Stat={stat_str}"
        if pval_str != 'N/A': formatted_string += f", P-Value={pval_str}"
        formatted_string += f". *Interpretation:* {interp}\n"

        if details and isinstance(details, str) and not html:
            cleaned_details = '\n'.join(line.strip() for line in details.strip().splitlines() if line.strip())
            formatted_string += f"   Details:\n```\n{cleaned_details}\n```\n"
        elif html:
            formatted_string += f"   (Detailed results were provided in an HTML table format).\n"
        found_diagnostics = True

    if not found_diagnostics:
        formatted_string = "\n\n**Diagnostic Tests:** No specific diagnostic results provided or tests were not applicable.\n"

    return formatted_string


def call_gemini_api(prompt, model_summary, diagnostics_results=None, post_test_result=None):
    """Calls the Gemini API with model summary and formatted diagnostic results."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
         print("CRITICAL ERROR: GEMINI_API_KEY not found in environment variables.")
         raise ValueError("AI configuration error: API key for the generative model is missing.")

    model_name = "gemini-1.5-pro-latest" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    formatted_diagnostics = format_diagnostics_for_prompt(diagnostics_results, post_test_result)
    summary_str = str(model_summary) if model_summary else "No model summary provided."

    full_prompt = f"{prompt}\n\n"
    full_prompt += f"**Model Summary Output:**\n```\n{summary_str}\n```\n"
    full_prompt += formatted_diagnostics

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "temperature": 0.5, # تقليل حرارة التوليد قليلاً لثبات الهيكل الأكاديمي
            "maxOutputTokens": 4096,
            "topP": 0.95,
            "topK": 40
        },
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=120) 
        response.raise_for_status() 
        result = response.json()

        if 'candidates' not in result or not result['candidates']:
             prompt_feedback = result.get('promptFeedback', {})
             block_reason = prompt_feedback.get('blockReason', 'Unknown')
             return f"Error: AI model did not return candidates. Reason: {block_reason}."

        candidate = result['candidates'][0]
        content = candidate.get('content', {})
        parts = content.get('parts', [{}])
        generated_text = parts[0].get('text', '') if parts else ''

        return generated_text.strip()

    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 'N/A'
        response_text = e.response.text if e.response is not None else 'N/A'
        print(f"Error: Gemini API request failed: Status={status_code}, Response={response_text[:300]}")
        try:
            error_json = json.loads(response_text)
            return f"Error: {error_json.get('error', {}).get('message', 'AI Generation Failed')}"
        except:
             return f"Error: AI report generation failed (Status: {status_code})."
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"


def generate_local_smart_report_en(model_summary, diagnostics_results=None, post_test_result=None, tone='academic'):
    issues = []
    passed = []
    
    all_tests = []
    if isinstance(diagnostics_results, list):
        all_tests.extend(diagnostics_results)
    if isinstance(post_test_result, dict) and post_test_result:
        all_tests.append(post_test_result)
        
    for test in all_tests:
        if not isinstance(test, dict): continue
        name = test.get('name', test.get('test_name', 'Diagnostic Test'))
        pval = test.get('p_value')
        stat = test.get('statistic', test.get('stat', 'N/A'))
        interp = test.get('interpretation', '')
        
        try:
            pval_val = float(pval) if pval not in (None, 'N/A', '') else None
        except:
            pval_val = None
            
        is_fail = 'fail' in interp.lower() or 'reject' in interp.lower() or (pval_val is not None and pval_val <= 0.05)
        
        test_info = {
            "name": name,
            "stat": stat,
            "pval": pval,
            "interp": interp,
            "failed": is_fail
        }
        if is_fail:
            issues.append(test_info)
        else:
            passed.append(test_info)

    if not all_tests:
        verdict = "Verdict Undetermined: No post-estimation diagnostic test outputs were provided. Statistical verification remains incomplete."
        recs = "Run primary diagnostic checks (e.g., Heteroskedasticity, Serial Correlation, Normality) to assess reliability."
    elif issues:
        verdict = "Conditionally Reliable / Unreliable: Several critical assumptions are violated. Standard errors are potentially biased, affecting significance tests."
        recs = "Apply Heteroskedasticity/Autocorrelation Consistent (HAC) standard errors or transform variables to restore model consistency."
    else:
        verdict = "Completely Reliable: All diagnostic tests passed. No evidence of heteroskedasticity, autocorrelation, or model misspecification was detected."
        recs = "Model can be safely deployed for policy forecasting and structural analysis."

    title_tone = "Academic" if tone == 'academic' else ("Policy Advisory" if tone == 'advisory' else "Executive Summary")
    report = []
    report.append(f"## Local AI-Synthesized {title_tone} Report (English)")
    report.append("\n### 1. Executive Summary")
    report.append("This empirical analysis report has been generated using local rule-based AI diagnostics. The framework evaluates the estimated model outputs, diagnostic metrics, and potential specification anomalies.")
    if issues:
        report.append(f"**Key Findings:** The model has failed {len(issues)} critical diagnostic checks. The coefficients are unbiased, but hypothesis tests (p-values) may be invalid due to standard error distortion.")
    else:
        report.append("**Key Findings:** The model successfully passed all diagnostic checks. Parameter estimates and covariance structures appear robust.")

    report.append("\n### 2. Model Interpretation")
    report.append("The estimated relationship represents the empirical response of the dependent variable to Exogenous regressors.")
    report.append("Refer to the statistical estimation summary below for the calculated parameter coefficients, standard errors, and goodness-of-fit metrics:")
    report.append(f"\n{model_summary}\n")

    report.append("\n### 3. Diagnostic Assessment")
    if not all_tests:
        report.append("No diagnostic test results were found. It is highly recommended to run normality, serial correlation, and heteroskedasticity tests to validate the model's assumptions.")
    else:
        report.append("| Diagnostic Test | Statistic | P-Value | Status | Interpretation |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for test in all_tests:
            name = test.get('name', 'Test')
            stat = test.get('statistic', test.get('stat', 'N/A'))
            pval = test.get('p_value', 'N/A')
            interp = test.get('interpretation', '')
            status = "❌ Failed" if test in [x for x in issues] else "✅ Passed"
            report.append(f"| {name} | {stat} | {pval} | {status} | {interp} |")

    report.append("\n### 4. Model Reliability Verdict")
    report.append(f"**Verdict:** {verdict}")

    report.append("\n### 5. Policy Implications")
    report.append("1. **Evidence-Based Policy formulation:** Significant coefficients should guide policy design. Positive parameters show target variables rise in response to adjustments.")
    report.append("2. **Robustness Safeguard:** Policies drafted using this framework should incorporate a safety margin to account for residual variance.")

    report.append("\n### 6. Limitations & Technical Recommendations")
    report.append(f"- **Core Recommendation:** {recs}")
    if issues:
        report.append("- **Remedial Actions:** Apply robust covariance standard errors (HC3 or HAC/Newey-West) to adjust p-values for heteroskedasticity or autocorrelation.")
    report.append("- **Data Sufficiency:** Ensure there are enough observations and no outliers distorting regression lines.")

    return "\n".join(report)


def generate_local_smart_report_ar(model_summary, diagnostics_results=None, post_test_result=None, tone='academic'):
    issues = []
    passed = []
    
    all_tests = []
    if isinstance(diagnostics_results, list):
        all_tests.extend(diagnostics_results)
    if isinstance(post_test_result, dict) and post_test_result:
        all_tests.append(post_test_result)
        
    for test in all_tests:
        if not isinstance(test, dict): continue
        name = test.get('name', test.get('test_name', 'الاختبار التشخيصي'))
        pval = test.get('p_value')
        stat = test.get('statistic', test.get('stat', 'N/A'))
        interp = test.get('interpretation', '')
        
        try:
            pval_val = float(pval) if pval not in (None, 'N/A', '') else None
        except:
            pval_val = None
            
        is_fail = 'fail' in interp.lower() or 'reject' in interp.lower() or (pval_val is not None and pval_val <= 0.05)
        
        test_info = {
            "name": name,
            "stat": stat,
            "pval": pval,
            "interp": interp,
            "failed": is_fail
        }
        if is_fail:
            issues.append(test_info)
        else:
            passed.append(test_info)

    if not all_tests:
        verdict = "الموثوقية غير محددة: لم يتم توفير نتائج الاختبارات التشخيصية للنموذج المقدر."
        recs = "يرجى تشغيل الاختبارات التشخيصية الأساسية (مثل اختلاف التباين، الارتباط الذاتي، والتوزيع الطبيعي) لتقييم جودة النموذج."
    elif issues:
        verdict = "موثوق بشرط / غير موثوق: تم اكتشاف انتهاك لبعض الفرضيات القياسية الأساسية، مما يجعل الأخطاء المعيارية وقيم الدلالة (p-values) غير متسقة."
        recs = "يُنصح بشدة باستخدام الأخطاء المعيارية القوية (HAC/Newey-West) لتعديل قيم الدلالة الإحصائية أو تحويل المتغيرات لإعادة النموذج للاتساق."
    else:
        verdict = "موثوق بالكامل: النموذج يجتاز جميع الاختبارات التشخيصية بنجاح، مما يثبت خلو البواقي من مشاكل الارتباط الذاتي أو عدم ثبات التباين."
        recs = "يمكن استخدام التقديرات الحالية بأمان في صياغة السياسات الهيكلية والتنبؤات الاقتصادية."

    title_tone = "أكاديمي" if tone == 'academic' else ("استشاري للسياسات" if tone == 'advisory' else "ملخص تنفيذي")
    report = []
    report.append(f"## تقرير إحصائي قياسي ({title_tone}) مصاغ محلياً (عربي)")
    report.append("\n### ١. الملخص التنفيذي")
    report.append("تم إعداد هذا التقرير الإحصائي ديناميكياً عبر محرك التحليل الإحصائي القياسي لمنصة DataNomics. يقوم النظام بتقييم هيكلية النموذج واختبار الفرضيات الأساسية وصياغة مؤشرات الموثوقية الاستشارية.")
    if issues:
        report.append(f"**أبرز النتائج:** أخفق النموذج في اجتياز {len(issues)} من الاختبارات التشخيصية الحرجة. المعاملات تظل غير متحيزة ولكن اختبارات الفرضيات قد تكون مضللة بسبب تشوه الأخطاء المعيارية.")
    else:
        report.append("**أبرز النتائج:** اجتاز النموذج كافة الاختبارات الإحصائية بنجاح، مما يثبت قوة وجودة التوفيق القياسي للنموذج المقدر.")

    report.append("\n### ٢. تفسير نتائج التقدير")
    report.append("توضح المعاملات المقدرة سلوك المتغير التابع استجابةً للتغيرات في المتغيرات المستقلة المفسرة.")
    report.append("يرجى مراجعة مصفوفة النتائج وجداول التقدير التالية للحصول على القيم الرقمية الدقيقة ومؤشرات جودة التوفيق:")
    report.append(f"\n{model_summary}\n")

    report.append("\n### ٣. التقييم التشخيصي البعدي")
    if not all_tests:
        report.append("لم يتم العثور على أي نتائج لاختبارات تشخيصية. نوصي بشدة بإجراء اختبارات التوزيع الطبيعي والارتباط الذاتي وتجانس التباين لضمان سلامة التقدير.")
    else:
        report.append("| الاختبار الإحصائي | قيمة الإحصاء | القيمة الاحتمالية P-Value | حالة الاختبار | التفسير الإحصائي |")
        report.append("| :--- | :--- | :--- | :--- | :--- |")
        for test in all_tests:
            name = test.get('name', 'الاختبار')
            stat = test.get('statistic', test.get('stat', 'N/A'))
            pval = test.get('p_value', 'N/A')
            interp = test.get('interpretation', '')
            status = "❌ فشل" if test in [x for x in issues] else "✅ نجح"
            report.append(f"| {name} | {stat} | {pval} | {status} | {interp} |")

    report.append("\n### ٤. حكم موثوقية النموذج وجدارة الثقة")
    report.append(f"**القرار النهائي:** {verdict}")

    report.append("\n### ٥. الآثار المترتبة على السياسات العامة")
    report.append("١. **صناعة السياسات القائمة على الأدلة:** يجب الاعتماد على المعاملات ذات الدلالة الإحصائية الحقيقية لرسم الخطط والسياسات.")
    report.append("٢. **مبدأ الحيطة القياسية:** نوصي بترك هامش أمان عند التنبؤ استناداً إلى نتائج هذا النموذج لتعويض تباينات الأخطاء غير المقاسة.")

    report.append("\n### ٦. القيود والتوصيات الفنية المقترحة")
    report.append(f"- **التوصية الفنية الرئيسية:** {recs}")
    if issues:
        report.append("- **الإجراءات العلاجية:** يُنصح بتطبيق تصحيح الأخطاء المعيارية المتسقة (HC3 أو Newey-West HAC) لحل مشكلة عدم ثبات التباين والارتباط الذاتي للبواقي.")
    report.append("- **حجم البيانات:** يفضل دائماً التأكد من كفاية حجم العينة وخلوها من القيم المتطرفة التي قد تحرف خط الانحدار.")

    return "\n".join(report)


def generate_gemini_english_report(model_summary, diagnostics_results=None, post_test_result=None, tone='academic'):
    """Generates a professional English report using the Gemini API, including diagnostics."""
    tone_instruction = ""
    if tone == 'advisory':
        tone_instruction = "Write this report in a Policy Advisory style, emphasizing practical economic guidance, policy implications, and institutional actions."
    elif tone == 'executive':
        tone_instruction = "Write this report in an Executive Summary style, keeping the analysis high-level, concise, and focused on key findings."
    else:
        tone_instruction = "Write this report in a rigorous Academic style, using precise econometric terminology and detailed diagnostic verification."

    prompt = f"""
    As an expert econometrician and Senior Economic Researcher, write a comprehensive and professional academic report based on the provided **Model Summary Output** AND **Diagnostic Test Results**. 
    
    {tone_instruction}
    
    CRITICAL STRUCTURE RULES:
    - Do NOT use any HTML tags (such as <b>, <i>, <span>, or <br>).
    - Use strict Markdown formatting only (**bold** for emphasis, lists, and standard markdown tables).
    
    Structure the report into these exact numbered sections:

    1. **Executive Summary:** Summarize core empirical findings regarding the variable relationships and pass/fail diagnostics framework.
    2. **Model Interpretation:** Analyze estimated coefficients (magnitude, sign, and economic logic at p<0.05) and goodness-of-fit parameters (R-squared / McFadden Pseudo R2).
    3. **Diagnostic Assessment:** Evaluate the 'Automatic Diagnostic Test Results' and any 'Specific Post-Estimation Test Result'. If tests failed (e.g. Heteroskedasticity or Serial Correlation), highlight the reliability breakdown.
    4. **Model Reliability Verdict:** Issue a clear econometric statement on whether the model is completely reliable, conditionally reliable, or unreliable for structural policy inference.
    5. **Policy Implications:** Formulate 2-3 highly actionable institutional policy recommendations based strictly on the significant coefficients.
    6. **Limitations & Technical Recommendations:** Outline model limits and prescribe exact remedial steps (e.g., HAC Newey-West standard errors, structural break controls, lag extensions, or switching to systems like VECM).
    """
    try:
        return call_gemini_api(prompt, model_summary, diagnostics_results, post_test_result)
    except ValueError as ve:
        if "API key" in str(ve):
            return generate_local_smart_report_en(model_summary, diagnostics_results, post_test_result, tone=tone)
        raise ve


def generate_gemini_arabic_report(model_summary, diagnostics_results=None, post_test_result=None, tone='academic'):
    """Generates a professional Arabic report using the Gemini API, including diagnostics."""
    tone_instruction = ""
    if tone == 'advisory':
        tone_instruction = "صغ هذا التقرير بأسلوب استشاري للسياسات، مع التركيز على التوجيهات الاقتصادية العملية والآثار المترتبة على السياسات العامة والإجراءات المؤسسية."
    elif tone == 'executive':
        tone_instruction = "صغ هذا التقرير كملخص تنفيذي، مع الحفاظ على مستوى عالٍ وموجز من التحليل والتركيز على النتائج والاستخلاصات الرئيسية."
    else:
        tone_instruction = "صغ هذا التقرير بأسلوب أكاديمي رصين ودقيق، مع استخدام المصطلحات الاقتصادية القياسية المتخصصة والتحقق التفصيلي من الاختبارات التشخيصية."

    prompt = f"""
    بصفتك خبيراً أول في الاقتصاد القياسي ومستشاراً اقتصادياً، قم بإعداد تقرير تحليلي واحترافي شامل باللغة العربية الفصحى الرصينة، مستنداً إلى **ملخص النموذج الإحصائي** و **نتائج الاختبارات التشخيصية** المرفقة.
    
    {tone_instruction}
    
    شروط التنسيق الحتمية:
    - يمنع منعاً باتاً استخدام أي وسوم أو تاجز HTML (مثل <b> أو <br> أو <span>).
    - يجب الاعتماد الكلي على تنسيق Markdown النظيف فقط (مثل استخدام النجمتين ** للنصوص العريضة، والجداول القياسية).
    
    قم بصياغة التقرير بدقة وفقاً للأقسام المرقمة التالية:

    ١. **الملخص التنفيذي:** تلخيص موجز لأبرز العلاقات السببية المكتشفة ومدى صلاحية النموذج الهيكلية بناءً على بيئة الاختبارات.
    ٢. **تفسير النموذج الاقتصادي:** تحليل دقيق للمتغيرات والمعاملات المقدرة (حجم الأثر، الإشارة الطردية/العكسية، والدلالة الإحصائية عند p<0.05) وتقييم القوة التفسيرية (R-squared أو McFadden Pseudo R2 للوجيت/البروبيت).
    ٣. **التقييم النظري للاختبارات التشخيصية:** نقد وتقييم شامل لكل من 'نتائج الاختبارات التلقائية' و 'الاختبار البعدي المحدد'. اشرح أثر الاختبارات الفاشلة (مثل عدم ثبات التباين أو الارتباط الذاتي) على كفاءة تقديرات الخطأ المعياري.
    ٤. **حكم موثوقية النموذج وجدارة الثقة:** تقديم قرار اقتصادي قياسي قاطع حول جدارة النموذج للاستدلال ورسم السياسات (موثوق بالكامل، موثوق مشروطاً مع حذر، غير موثوق).
    ٥. **الآثار المترتبة على السياسات العامة:** صياغة 2-3 توصيات استراتيجية قابلة للتطبيق الفوري ومستمدة مباشرة من المؤشرات ذات الدلالة الإحصاسية، مع ربط ديباجتها بحالة الموثوقية الصادرة في القسم ٤.
    ٦. **القيود والتوصيات الفنية المقترحة:** رصد محدد للقيود (مثل حذف متغيرات هامة) وتقديم حلول قياسية للعلاج (مثل: استخدام أخطاء Newey-West HAC، ضبط الانقطاعات الهيكلية، أو التحول لأنظمة متكاملة مثل VECM).
    """
    try:
        return call_gemini_api(prompt, model_summary, diagnostics_results, post_test_result)
    except ValueError as ve:
        if "API key" in str(ve):
            return generate_local_smart_report_ar(model_summary, diagnostics_results, post_test_result, tone=tone)
        raise ve