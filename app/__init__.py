from flask import Flask
from flask_cors import CORS
from .config import Config
import os
import firebase_admin
from firebase_admin import credentials

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)


    # --- FIREBASE INITIALIZATION START ---
    try:
        # Path to your service account key file
        cred_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'serviceAccountKey.json')
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("Firebase App initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase App: {e}")
        # This will help debug if the key file is missing or corrupted.
    # --- FIREBASE INITIALIZATION END ---
    
    # Configure CORS based on environment
    if os.environ.get('FLY_APP_NAME'):
        # Production: Allow your production frontend domains
        allowed_origins = [
            "https://econo-vision.vercel.app",  # Your custom domain
            "https://econo-vision-euziy53vy-ahmedtarek52s-projects.vercel.app",  # Current deployment URL
            "https://econo-vision-*.vercel.app",  # Pattern for all preview deployments
        ]
        CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    else:
        # Development: Allow localhost
        CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
    
    # Register Blueprints (rest of your code stays the same)
    from .blueprints.data_upload.routes import data_upload_bp
    from .blueprints.data_preparation.routes import data_preparation_bp
    from .blueprints.dashboard.routes import dashboard_bp
    from .blueprints.statistical_tests.routes import statistical_tests_bp
    from .blueprints.modeling.routes import modeling_bp
    from .blueprints.reports.routes import reports_bp
    from .blueprints.contact.routes import contact_bp
    
    app.register_blueprint(data_upload_bp, url_prefix='/api/data')
    app.register_blueprint(data_preparation_bp, url_prefix='/api/prepare')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(statistical_tests_bp, url_prefix='/api/tests')
    app.register_blueprint(modeling_bp, url_prefix='/api/model')
    app.register_blueprint(reports_bp, url_prefix='/api/report')
    app.register_blueprint(contact_bp, url_prefix='/api/contact')
    
    return app