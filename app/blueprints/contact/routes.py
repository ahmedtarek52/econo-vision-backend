# app/blueprints/contact/routes.py
from flask import Blueprint, request, jsonify
from firebase_admin import firestore

contact_bp = Blueprint('contact_bp', __name__)

@contact_bp.route('/submit', methods=['POST'])
def submit_feedback():
    try:
        # Initialize Firestore DB client
        db = firestore.client()

        data = request.get_json()
        
        email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        rating = data.get('rating')

        if not all([email, subject, message]):
            return jsonify({"error": "Email, subject, and message are required."}), 400

        # Create a new document in the 'contacts' collection
        doc_ref = db.collection('contacts').document()
        doc_ref.set({
            'email': email,
            'subject': subject,
            'message': message,
            'rating': rating,
            'timestamp': firestore.SERVER_TIMESTAMP  # Adds a server-side timestamp
        })

        return jsonify({"message": "Feedback saved successfully!"}), 200

    except Exception as e:
        # Log the full error to the console for debugging
        print(f"An error occurred while saving to Firestore: {e}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500