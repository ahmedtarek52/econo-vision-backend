import requests
import urllib3
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# إيقاف تحذيرات SSL للبيئات التي بها مشاكل اتصال
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# قاموس لترجمة أسماء الدول لأكواد ISO-3
COUNTRY_MAP = {
    'egypt': 'EGY', 'egy': 'EGY',
    'uae': 'ARE', 'united arab emirates': 'ARE', 'are': 'ARE',
    'saudi arabia': 'SAU', 'saudi': 'SAU', 'sau': 'SAU',
    'usa': 'USA', 'united states': 'USA', 'us': 'USA'
}

class GlobalEconomicDataHub:
    
    @staticmethod
    def _get_session():
        session = requests.Session()
        retry = Retry(connect=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        return session

    @staticmethod
    def search_indicator_list(query):
        if not query or len(query) < 2: return []
        
        url = "https://api.worldbank.org/v2/indicator"
        params = {"format": "json", "q": query, "per_page": 20} # سحب 20 نتيجة
        
        try:
            response = requests.get(url, params=params, timeout=10, verify=False)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    all_results = [{"value": item['id'], "label": item['name']} for item in data[1]]
                    
                    # 🟢 هنا السحر: ترتيب النتائج (التي تحتوي على الكلمة في اسمها ستكون في الأول)
                    query_lower = query.lower()
                    all_results.sort(key=lambda x: 0 if query_lower in x['label'].lower() else 1)
                    
                    return all_results[:10] # إرجاع أفضل 10 نتائج فقط
        except Exception as e:
            print(f"DEBUG: Search Error: {e}")
        return []

    @staticmethod
    def fetch_world_bank_data(indicator, countries=['EGY'], start_year=2010, end_year=2025):
        try:
            normalized_countries = [COUNTRY_MAP.get(c.lower(), c.upper()) for c in countries]
            country_str = ";".join(normalized_countries)
            url = f"https://api.worldbank.org/v2/country/{country_str}/indicator/{indicator}"
            
            params = {"date": f"{start_year}:{end_year}", "format": "json", "per_page": 1000}
            
            session = GlobalEconomicDataHub._get_session()
            response = session.get(url, params=params, timeout=60, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    parsed_list = []
                    for rec in data[1]:
                        if rec["value"] is not None:
                            parsed_list.append({
                                "Entity": rec["country"]["value"],
                                "Country_Code": rec["countryiso3code"],
                                "Year": int(rec["date"]),
                                indicator: float(rec["value"])
                            })
                    return pd.DataFrame(parsed_list)
        except Exception as e:
            print(f"DEBUG: Fetch Error for {indicator}: {e}")
            
        return pd.DataFrame()