from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from .models import User
from . import db

auth = Blueprint('auth', __name__, url_prefix='/auth')

# === NEW: GET route to show login page ===
@auth.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

# === Existing POST login handler ===
@auth.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return redirect(url_for('ui.dashboard'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('auth.login_page'))

# === NEW: GET route to show signup page ===
@auth.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

# === Existing POST signup handler ===
@auth.route('/signup', methods=['POST'])
def signup():
    email = request.form.get('email')
    password = request.form.get('password')

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('User already exists')
        return redirect(url_for('auth.signup_page'))

    new_user = User(email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    login_user(new_user)
    return redirect(url_for('ui.dashboard'))

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login_page'))
