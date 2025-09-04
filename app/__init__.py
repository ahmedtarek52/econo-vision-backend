from flask import Flask
from flask_cors import CORS
from .config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Allow requests from your React app's origin
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

    # Register Blueprints
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