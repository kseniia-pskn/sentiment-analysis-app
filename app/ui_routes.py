# ui_routes.py
from flask import Blueprint, render_template
from flask_login import login_required

ui = Blueprint('ui', __name__)

@ui.route('/')
def index():
    return render_template('index.html')

@ui.route('/auth/login')
def login():
    return render_template('login.html')

@ui.route('/auth/signup')
def signup():
    return render_template('signup.html')

@ui.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@ui.route('/profile')
@login_required
def profile():
    return render_template('profile.html')
