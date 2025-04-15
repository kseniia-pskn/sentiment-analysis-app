from flask import Blueprint, render_template, redirect, url_for, flash, request
from werkzeug.security import generate_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from .models import db, User

auth = Blueprint('auth', __name__)

# ----- Sign Up -----
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')
        confirm = request.form.get('confirm')

        print(f"[DEBUG] Signup attempt: email={email}")

        if not email or not password:
            flash("Email and password required.", "error")
            print("[DEBUG] Missing email or password.")
            return redirect(url_for('auth.signup'))

        if password != confirm:
            flash("Passwords do not match.", "error")
            print("[DEBUG] Passwords did not match.")
            return redirect(url_for('auth.signup'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered.", "error")
            print("[DEBUG] User already exists.")
            return redirect(url_for('auth.signup'))

        try:
            new_user = User(email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            print(f"[DEBUG] ✅ New user created: {new_user.email}")
            flash("✅ Account created. Please log in.", "success")
        except Exception as e:
            db.session.rollback()
            print(f"[DEBUG] ❌ DB commit failed: {e}")
            flash("Something went wrong. Please try again later.", "error")
            return redirect(url_for('auth.signup'))

        return redirect(url_for('auth.login'))

    return render_template('signup.html')


# ----- Login -----
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password')

        print(f"[DEBUG] Login attempt for: {email}")

        user = User.query.filter_by(email=email).first()

        if user:
            print(f"[DEBUG] Found user: {user.email}")
        else:
            print("[DEBUG] ❌ No user found with that email.")

        if not user:
            flash("❌ User not found. Please check your email or sign up.", 'danger')
            return render_template('login.html', email=email)

        if not user.check_password(password):
            flash("❌ Incorrect password. Please try again.", 'danger')
            print("[DEBUG] ❌ Incorrect password entered.")
            return render_template('login.html', email=email)

        login_user(user)
        print(f"[DEBUG] ✅ Login successful for: {user.email}")
        flash("✅ Logged in successfully!", 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('login.html')


# ----- Logout (POST only) -----
@auth.route('/logout', methods=['POST'])
@login_required
def logout():
    print(f"[DEBUG] Logging out user: {getattr(current_user, 'email', 'unknown')}")
    logout_user()
    flash("You’ve been logged out.", "info")
    return redirect(url_for('main.index'))
