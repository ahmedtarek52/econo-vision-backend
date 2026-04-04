from flask import Blueprint, request, jsonify, session, current_app
from app.decorators import login_required

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    email = data.get('email', 'mock@example.com')
    return jsonify({
        "uid": "mock_uid_123", 
        "email": email,
        "message": "User created successfully (Mockled)."
    }), 201

@auth_bp.route('/login-session', methods=['POST'])
def create_session():
    id_token = request.json.get('idToken')
    uid = "mock_uid_123"
    session.clear()
    session['user_id'] = uid
    session.permanent = True 
    import datetime
    current_app.permanent_session_lifetime = datetime.timedelta(days=14) 
    return jsonify({"status": "success", "uid": uid}), 200

@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200

@auth_bp.route('/check-session', methods=['GET'])
def check_session():
    return jsonify({"isAuthenticated": True, "uid": "mocked_local_user"}), 200

@auth_bp.route('/save-project', methods=['POST'])
@login_required
def save_project_data():
    return jsonify({"status": "success", "message": "Project saved (Mocked)."}), 200

@auth_bp.route('/load-project', methods=['GET'])
@login_required
def load_project_data():
    return jsonify({
        "analysisData": {"fullDataset": [], "transformationHistory": [], "columns": []},
        "comparisonBasket": []
    }), 200