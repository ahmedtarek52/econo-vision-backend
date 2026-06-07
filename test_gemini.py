import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print("API Key loaded:", api_key)

try:
    genai.configure(api_key=api_key)
    print("Listing models:")
    for m in genai.list_models():
        print(f"Model: {m.name}, Supported Methods: {m.supported_generation_methods}")
except Exception as e:
    print("Error occurred:")
    import traceback
    traceback.print_exc()
