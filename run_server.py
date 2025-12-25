from app import create_app
from dotenv import load_dotenv # <-- 1. استيراد هنا
import os

# --- 2. (جديد) تحميل متغيرات البيئة قبل كل شيء ---
# هذا يضمن تحميلها قبل استيراد أي شيء داخل create_app()
load_dotenv()
print("Loaded .env file from run_server.py")
# ---

# (اختياري) طباعة المفتاح للتأكد (احذفه لاحقاً)
# print(f"[DEBUG] GEMINI_API_KEY in run_server: {os.environ.get('GEMINI_API_KEY')}")

app = create_app()

if __name__ == '__main__':
    # يمكنك تحديد البورت 5000 هنا
    # استخدام host='0.0.0.0' يسمح بالوصول من خارج الجهاز (مفيد للتجارب)
    app.run(debug=True, host='0.0.0.0' ,use_reloader=False, port=int(os.environ.get('PORT', 5000)))
