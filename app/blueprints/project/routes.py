# app/blueprints/project/routes.py

from flask import Blueprint, request, jsonify, session, current_app
from firebase_admin import firestore
import datetime
from ..decorators import login_required # (استيراد الحارس الأمني)

# --- إعداد الـ Blueprint ---
project_bp = Blueprint('project_api', __name__, url_prefix='/api/project')

# (دالة مساعدة لمعالجة الأخطاء)
def handle_project_error(e, default_message="An error occurred", status_code=500):
    error_message = str(e)
    print(f"--- PROJECT ERROR: {error_message} ---")
    import traceback
    traceback.print_exc()
    print("------------------")
    
    user_message = default_message
    if isinstance(e, (ValueError, KeyError, RuntimeError)):
        user_message = error_message
        status_code = 400
    
    return jsonify({"error": user_message}), status_code

# --- (جديد) 1. Endpoint: إنشاء مشروع جديد ---
@project_bp.route('/create', methods=['POST'])
@login_required
def create_project():
    """
    ينشئ مستند مشروع جديد فارغ في Firestore.
    """
    try:
        user_id = session['user_id']
        data = request.get_json()
        project_name = data.get('projectName')
        
        if not project_name:
            raise ValueError("Project name is required.")
            
        db = firestore.client()
        user_ref = db.collection('users').document(user_id)
        
        # إنشاء "مشروع افتراضي" فارغ
        empty_project = {
            "name": project_name,
            "analysisData": {
                "filename": "",
                "fullDataset": [],
                "columns": [],
                "transformationHistory": [],
                "isPanel": False,
                "panelIdVar": "",
                "panelTimeVar": ""
            },
            "comparisonBasket": [],
            "createdAt": firestore.SERVER_TIMESTAMP,
            "last_saved": firestore.SERVER_TIMESTAMP
        }
        
        # إنشاء مستند جديد بـ ID عشوائي
        new_project_ref = user_ref.collection('projects').document()
        new_project_ref.set(empty_project)
        
        return jsonify({"status": "success", "projectId": new_project_ref.id, "projectName": project_name}), 201

    except Exception as e:
        return handle_project_error(e, "Failed to create new project.")

# --- (جديد) 2. Endpoint: جلب كل المشاريع ---
@project_bp.route('/list', methods=['GET'])
@login_required
def list_projects():
    """
    يجلب قائمة بكل مشاريع المستخدم الحالية.
    """
    try:
        user_id = session['user_id']
        db = firestore.client()
        projects_ref = db.collection('users').document(user_id).collection('projects')
        
        projects_list = []
        # (يمكن إضافة .order_by('last_saved', direction=firestore.Query.DESCENDING) مستقبلاً)
        for doc in projects_ref.stream():
            project_data = doc.to_dict()
            # (تحويل التواريخ إلى نصوص)
            last_saved_timestamp = project_data.get('last_saved')
            last_saved_str = "Not saved yet"
            if last_saved_timestamp:
                try:
                    last_saved_str = last_saved_timestamp.strftime("%Y-%m-%d %H:%M")
                except:
                     pass # (اتركه كما هو إذا لم يكن تاريخاً)

            projects_list.append({
                "id": doc.id,
                "name": project_data.get("name", "Untitled Project"),
                "filename": project_data.get("analysisData", {}).get("filename", "No file"),
                "last_saved": last_saved_str
            })
            
        return jsonify(projects_list), 200

    except Exception as e:
        return handle_project_error(e, "Failed to list projects.")

# --- (جديد) 3. Endpoint: تحميل مشروع معين ---
@project_bp.route('/load/<project_id>', methods=['GET'])
@login_required
def load_project_data(project_id):
    """
    يقوم بتحميل آخر حالة مشروع محفوظة للمستخدم من Firestore.
    """
    try:
        user_id = session['user_id']
        db = firestore.client()
        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        
        project_doc = project_ref.get()
        
        if not project_doc.exists:
            raise ValueError("Project not found or access denied.")

        project_data = project_doc.to_dict()

        # (تنظيف البيانات قبل إرسالها)
        project_data.pop('last_saved', None) 
        project_data.pop('createdAt', None) 
        
        print(f"Project {project_id} loaded successfully for user: {user_id}")
        return jsonify(project_data), 200

    except Exception as e:
        return handle_project_error(e, "Failed to load project data.")

# --- (جديد) 4. Endpoint: حفظ مشروع معين ---
@project_bp.route('/save/<project_id>', methods=['POST'])
@login_required
def save_project_data(project_id):
    """
    يستقبل حالة المشروع ويحفظها في مستند مشروع معين.
    """
    try:
        user_id = session['user_id']
        data = request.get_json()
        
        if 'analysisData' not in data or 'comparisonBasket' not in data:
            raise ValueError("Missing 'analysisData' or 'comparisonBasket' in payload.")

        analysis_data = data['analysisData']
        comparison_basket = data['comparisonBasket']
        
        db = firestore.client()
        # (مهم) نتأكد أن المستخدم يحفظ في مشروعه فقط
        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        
        # (استخدم merge=True للحفاظ على الاسم وتاريخ الإنشاء)
        project_data = {
            "analysisData": analysis_data,
            "comparisonBasket": comparison_basket,
            "last_saved": firestore.SERVER_TIMESTAMP
        }
        
        project_ref.set(project_data, merge=True)
        
        print(f"Project {project_id} saved successfully for user: {user_id}")
        return jsonify({"status": "success", "message": "Project saved."}), 200

    except Exception as e:
        return handle_project_error(e, "Failed to save project data.")

# --- (جديد) 5. Endpoint: حذف مشروع ---
@project_bp.route('/delete/<project_id>', methods=['DELETE'])
@login_required
def delete_project(project_id):
    try:
        user_id = session['user_id']
        db = firestore.client()
        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        
        # (اختياري: تحقق من أن المشروع موجود قبل الحذف)
        if not project_ref.get().exists:
             raise ValueError("Project not found.")
             
        project_ref.delete()
        
        print(f"Project {project_id} deleted successfully for user: {user_id}")
        return jsonify({"status": "success", "message": "Project deleted."}), 200
    except Exception as e:
        return handle_project_error(e, "Failed to delete project.")

# --- (جديد) 6. Endpoint: إعادة تسمية مشروع ---
@project_bp.route('/rename/<project_id>', methods=['PUT'])
@login_required
def rename_project(project_id):
    try:
        user_id = session['user_id']
        data = request.get_json()
        new_name = data.get('newName')

        if not new_name:
            raise ValueError("New project name is required.")

        db = firestore.client()
        project_ref = db.collection('users').document(user_id).collection('projects').document(project_id)
        
        project_ref.update({"name": new_name})
        
        print(f"Project {project_id} renamed successfully for user: {user_id}")
        return jsonify({"status": "success", "message": "Project renamed."}), 200
    except Exception as e:
        return handle_project_error(e, "Failed to rename project.")