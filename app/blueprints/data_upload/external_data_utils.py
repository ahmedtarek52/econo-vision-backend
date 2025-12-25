import pandas as pd
import numpy as np
import wbgapi as wb 
import requests 
import io 
import warnings 

warnings.filterwarnings("ignore", category=FutureWarning)

# --- بيانات ثابتة (مُعدة للـ Frontend) ---
WORLD_BANK_COUNTRIES = [
    {'id': 'EG', 'name': 'Egypt, Arab Rep.'},
    {'id': 'SA', 'name': 'Saudi Arabia'},
    {'id': 'AE', 'name': 'United Arab Emirates'},
    {'id': 'TR', 'name': 'Türkiye'},
    {'id': 'US', 'name': 'United States'},
    {'id': 'WLD', 'name': 'World (Global)'}, 
]
WORLD_BANK_INDICATORS = [
    {'id': 'NY.GDP.MKTP.CD', 'name': 'GDP (Current US$)'},
    {'id': 'SP.POP.TOTL', 'name': 'Population, total'},
    {'id': 'FP.CPI.TOTL.ZG', 'name': 'Inflation, consumer prices (annual %)'},
    {'id': 'SL.UEM.TOTL.ZS', 'name': 'Unemployment, total (% of labor force)'},
    {'id': 'GC.BAL.CASH.GD.ZS', 'name': 'Cash surplus/deficit (% of GDP)'},
]

# --- دالة سحب بيانات البنك الدولي (World Bank) ---
def fetch_world_bank_data(indicator_codes, countries=None, date_range=None):
    if not indicator_codes:
        raise ValueError("Indicator codes list cannot be empty.")
        
    time_period = None
    if date_range and len(date_range) == 2:
        start, end = date_range
        time_period = range(start, end + 1)
        
    all_data_frames = []
    
    # قائمة الدول يجب أن يتم تنظيفها هنا (نحذف WLD لو كانت موجودة لتجنب بطء الـ API)
    countries_list = countries if countries and 'all' not in countries else [c['id'] for c in WORLD_BANK_COUNTRIES if c['id'] != 'WLD']
    
    if not countries_list:
        raise ValueError("No valid countries selected for fetching.")
        
    for country_code in countries_list:
        try:
            print(f"Fetching data for country: {country_code}")
            df = wb.data.DataFrame(
                series=indicator_codes,
                economy=country_code, 
                time=time_period,
                labels=True
            )
            if not df.empty:
                all_data_frames.append(df)
        except Exception as e:
            print(f"Skipping country {country_code} due to API error: {e}")
            continue

    if not all_data_frames:
        raise RuntimeError("Failed to fetch data for all selected countries.")

    df_combined = pd.concat(all_data_frames)
    df = df_combined.reset_index()

    # --- تنظيف البيانات وتدويرها (Pivot) ---
    if 'economy' in df.columns:
         df = df.rename(columns={'economy': 'CountryCode'})
         
    if 'series' in df.columns and 'value' in df.columns:
        df = df.rename(columns={'series': 'IndicatorName'})
        df_pivot = df.pivot_table(index=['CountryCode', 'Time'], columns='IndicatorName', values='value')
        df_pivot = df_pivot.reset_index()
        df_pivot = df_pivot.rename(columns={'Time': 'Year'})
        df_pivot.columns.name = None
        return df_pivot
    
    return df

def fetch_imf_data(indicator_code, country_code='all', start_year=2000, end_year=2024):
    raise NotImplementedError("IMF API integration is complex and requires custom configuration. Please use World Bank for now.")

def get_world_bank_metadata():
    """يرجع قوائم المؤشرات والدول للـ Frontend"""
    return {
        'indicators': WORLD_BANK_INDICATORS,
        'countries': WORLD_BANK_COUNTRIES
    }
