from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required, current_user
from .models import User

main = Blueprint('main', __name__)

# ----- Home Page -----
@main.route('/')
def index():
    if current_user.is_authenticated:
        return render_template('dashboard.html')
    else:
        return render_template('index.html')


# ----- Dashboard -----
@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)


# ----- My Profile -----
@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)


# ----- Debug: List All Users -----
@main.route('/debug-users')
def debug_users():
    if not current_app.config.get("DEBUG"):
        return jsonify({"error": "Access denied. DEBUG mode only."}), 403

    users = User.query.all()
    return jsonify([
        {
            "id": user.id,
            "email": user.email,
            "created_at": str(user.created_at)
        } for user in users
    ])
