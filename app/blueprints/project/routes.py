from flask import Blueprint, request, jsonify, session
import datetime
from ..decorators import login_required

project_bp = Blueprint('project_api', __name__, url_prefix='/api/project')

def handle_project_error(e, default_message="An error occurred", status_code=500):
    error_message = str(e)
    user_message = default_message
    if isinstance(e, (ValueError, KeyError, RuntimeError)):
        user_message = error_message
        status_code = 400
    return jsonify({"error": user_message}), status_code

@project_bp.route('/create', methods=['POST'])
@login_required
def create_project():
    data = request.get_json()
    project_name = data.get('projectName')
    if not project_name:
         return handle_project_error(ValueError("Project name is required."), status_code=400)
    return jsonify({"status": "success", "projectId": "mock_project_id", "projectName": project_name}), 201

@project_bp.route('/list', methods=['GET'])
@login_required
def list_projects():
    return jsonify([]), 200

@project_bp.route('/load/<project_id>', methods=['GET'])
@login_required
def load_project_data(project_id):
    return jsonify({
        "name": "Mock Project",
        "analysisData": {
            "filename": "",
            "fullDataset": [],
            "columns": [],
            "transformationHistory": [],
            "isPanel": False,
            "panelIdVar": "",
            "panelTimeVar": ""
        },
        "comparisonBasket": []
    }), 200

@project_bp.route('/save/<project_id>', methods=['POST'])
@login_required
def save_project_data(project_id):
    return jsonify({"status": "success", "message": "Project saved (Mocked)."}), 200

@project_bp.route('/delete/<project_id>', methods=['DELETE'])
@login_required
def delete_project(project_id):
    return jsonify({"status": "success", "message": "Project deleted (Mocked)."}), 200

@project_bp.route('/rename/<project_id>', methods=['PUT'])
@login_required
def rename_project(project_id):
    return jsonify({"status": "success", "message": "Project renamed (Mocked)."}), 200