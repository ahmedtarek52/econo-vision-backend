from flask import Blueprint, request, jsonify, g
from firebase_admin import firestore

# (!!!) (هذا هو الإصلاح الأول: المسار الصحيح) (!!!)
# نحن نصعد مستويين:
# من: app/blueprints/contact
# إلى: app/
# (نفترض أن ملف 'decorators.py' موجود في 'app/decorators.py')
from ...decorators import login_required

# (!!!) (هذا هو الإصلاح الثاني: إضافة url_prefix) (!!!)
contact_bp = Blueprint('contact_bp', __name__, url_prefix='/api/contact')

@contact_bp.route('/submit', methods=['POST'])
@login_required # (تطبيق الحارس)
def submit_feedback():
	try:
		# Initialize Firestore DB client
		db = firestore.client()

		data = request.get_json()
		
		# (جلب المستخدم من الجلسة الآمنة التي يوفرها الحارس)
		uid = g.uid

		email = data.get('email')
		subject = data.get('subject')
		message = data.get('message')
		rating = data.get('rating')

		if not all([email, subject, message]):
			return jsonify({"error": "Email, subject, and message are required."}), 400

		# Create a new document in the 'contacts' collection
		doc_ref = db.collection('contacts').document()
		doc_ref.set({
			'uid': uid, # (حفظ من أرسل الرسالة)
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
		import traceback
		traceback.print_exc()
		return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500