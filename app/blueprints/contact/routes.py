# app/blueprints/contact/routes.py
from flask import Blueprint, request, jsonify

contact_bp = Blueprint('contact_bp', __name__)

@contact_bp.route('/submit', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        rating = data.get('rating')

        if not all([email, subject, message]):
            return jsonify({"error": "Email, subject, and message are required."}), 400

        # In a real application, you would send an email or save this to a database.
        # For this project, we'll just print it to the console to confirm receipt.
        print("--- New Feedback Received ---")
        print(f"Email: {email}")
        print(f"Subject: {subject}")
        print(f"Rating: {rating if rating > 0 else 'Not provided'}")
        print(f"Message: {message}")
        print("-----------------------------")

        return jsonify({"message": "Feedback received successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500