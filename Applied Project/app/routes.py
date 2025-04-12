from flask import Blueprint, render_template
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

# ----- Home Page -----
@main.route('/')
def index():
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
