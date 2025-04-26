import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS

from .models import db, User
from .utils import get_sentiment_pipeline, get_nlp

# App factory pattern
def create_app():
    app = Flask(__name__)

    # ----------------------
    # Configuration
    # ----------------------
    app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "default-secret-key")

    #  fallback to SQLite if DATABASE_URL is not provided
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        "DATABASE_URL", 
        "sqlite:///local_database.db"
    )

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ----------------------
    # Extensions
    # ----------------------
    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = "auth.login"
    login_manager.init_app(app)

    CORS(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ----------------------
    # Blueprints
    # ----------------------
    from .api_routes import api as api_bp
    from .auth_routes import auth as auth_bp
    from .ui_routes import ui as ui_bp

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(ui_bp)

    # ----------------------
    # Self-Diagnostics
    # ----------------------
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully.")
        except Exception as e:
            print("[ERROR] Failed to initialize database tables:", e)

        try:
            _ = get_sentiment_pipeline()
        except Exception as e:
            print("[ERROR] Failed to initialize sentiment model:", e)

        try:
            _ = get_nlp()
        except Exception as e:
            print("[ERROR] Failed to load SpaCy NLP model:", e)

    return app
