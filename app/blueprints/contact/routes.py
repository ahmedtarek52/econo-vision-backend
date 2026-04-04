from flask import Blueprint, request, jsonify, g
from ...decorators import login_required
import datetime

contact_bp = Blueprint('contact_bp', __name__, url_prefix='/api/contact')

@contact_bp.route('/submit', methods=['POST'])
@login_required
def submit_feedback():
    data = request.get_json()
    email = data.get('email')
    subject = data.get('subject')
    message = data.get('message')
    if not all([email, subject, message]):
        return jsonify({"error": "Email, subject, and message are required."}), 400

    return jsonify({"message": "Feedback saved successfully (Mocked)!"}), 200