from app import create_app
import os
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

app = create_app()

if __name__ == '__main__':
    # This block is for LOCAL DEVELOPMENT ONLY.
    # In production (Fly.io), a WSGI server like Gunicorn will run the app.
    
    # Ensure the upload folder exists locally
    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads') # Use a default value
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
    # The port is read from the environment variable PORT (with fallback to 5000 for local consistency).
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
