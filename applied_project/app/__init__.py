from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from .models import db, User
import os

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration from config.py (absolute path, not relative)
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    app.config.from_pyfile(config_path, silent=False)

    # Enable CORS
    CORS(app)

    # Initialize SQLAlchemy
    db.init_app(app)

    # Create database tables and confirm it
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully.")
        print(f"📂 Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")

    # Set up Flask-Login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register Blueprints
    from .routes import main as main_blueprint
    from .auth_routes import auth as auth_blueprint
    from .api_routes import api as api_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app
