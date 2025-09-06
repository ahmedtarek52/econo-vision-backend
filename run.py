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
    
    # Get port from environment variable (for production) or use 5000 (for development)
    port = int(os.environ.get('PORT', 5000))
    
    # Check if we're in production (Fly.io sets this)
    debug_mode = os.environ.get('FLY_APP_NAME') is None
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)