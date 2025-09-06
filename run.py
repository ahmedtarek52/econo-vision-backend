from app import create_app
import os
from dotenv import load_dotenv

load_dotenv()
app = create_app()

if __name__ == '__main__':
    # Ensure the upload folder exists
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    # Make sure to run on port 5000 as expected by the frontend
    app.run(debug=True, host='0.0.0.0', port=5000)