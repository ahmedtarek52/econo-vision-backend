from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from functools import reduce
from app.decorators import login_required
from app.services.external_data_service import GlobalEconomicDataHub

external_data_bp = Blueprint('external_data_api', __name__, url_prefix='/api/external-data')

def sanitize_for_json(df):
    df_clean = df.copy()
    df_clean = df_clean.replace([np.inf, -np.inf], None)
    df_clean = df_clean.where(pd.notnull(df_clean), None)
    return df_clean.to_dict(orient='records')

@external_data_bp.route('/search-indicator', methods=['GET'])
@login_required
def search_indicator_route():
    query = request.args.get('q', '')
    # استدعاء الدالة من داخل الكلاس مباشرة
    results = GlobalEconomicDataHub.search_indicator_list(query)
    return jsonify(results), 200

@external_data_bp.route('/pull-world-bank', methods=['POST'])
@login_required
def pull_world_bank():
    """
    سحب ودمج بيانات المؤشرات من البنك الدولي
    """
    try:
        # 1. استلام البيانات من الفرونت إند
        payload = request.get_json()
        indicators = payload.get('indicators', [])
        countries = payload.get('countries', ['EGY'])
        start_year = int(payload.get('startYear', 2010))
        end_year = int(payload.get('endYear', 2025))

        if not indicators:
            return jsonify({"error": "يرجى اختيار مؤشر واحد على الأقل"}), 400

        # 2. سحب البيانات لكل مؤشر
        data_frames = []
        for ind_id in indicators:
            print(f"DEBUG: Fetching indicator: {ind_id}") # للمتابعة في الـ Terminal
            
            df = GlobalEconomicDataHub.fetch_world_bank_data(
                indicator=ind_id, 
                countries=countries, 
                start_year=start_year, 
                end_year=end_year
            )
            
            # التأكد من أن الـ DataFrame يحتوي على بيانات
            if not df.empty:
                data_frames.append(df)
            else:
                print(f"DEBUG: WARNING - No data found for: {ind_id}")

        # 3. التحقق من وجود بيانات بعد السحب
        if not data_frames:
            # هنا التعديل: إرجاع 200 بدلاً من 404 لمنع توقف الفرونت إند
            return jsonify({
                "status": "warning", 
                "message": "لا توجد بيانات متاحة للمؤشرات المختارة في هذا النطاق الزمني."
            }), 200

        # 4. دمج الجداول (Merge)
        # نقوم بدمج كل الجداول بناءً على الأعمدة المشتركة
        final_df = reduce(lambda left, right: pd.merge(
            left, right, on=['Entity', 'Country_Code', 'Year'], how='outer'
        ), data_frames)

        # 5. تنظيف البيانات وإرسالها
        cleaned_dataset = sanitize_for_json(final_df)
        
        return jsonify({
            "status": "success", 
            "dataset": cleaned_dataset, 
            "columns": final_df.columns.tolist()
        }), 200
        
    except Exception as e:
        # تسجيل الخطأ في الـ Terminal لمعرفته لاحقاً
        print(f"DEBUG: CRITICAL ERROR in pull-world-bank: {str(e)}")
        return jsonify({"error": f"حدث خطأ داخلي: {str(e)}"}), 500