from flask import Blueprint, render_template, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash
from flask_login import login_user, logout_user, login_required
from .models import db, User

auth = Blueprint('auth', __name__)

# ----- Sign Up -----
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        if not email or not password:
            flash("Email and password required.", "error")
            return redirect(url_for('auth.signup'))

        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for('auth.signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "error")
            return redirect(url_for('auth.signup'))

        new_user = User(email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("✅ Account created. Please log in.", "success")
        return redirect(url_for('auth.login'))

    return render_template('signup.html')


# ----- Login -----
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email').strip().lower()
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("❌ User not found. Please check your email or sign up.", 'danger')
            return render_template('login.html', email=email)

        if not user.check_password(password):
            flash("❌ Incorrect password. Please try again.", 'danger')
            return render_template('login.html', email=email)

        login_user(user)
        flash("✅ Logged in successfully!", 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('login.html')


# ----- Logout -----
@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You’ve been logged out.", "info")
    return redirect(url_for('main.index'))
