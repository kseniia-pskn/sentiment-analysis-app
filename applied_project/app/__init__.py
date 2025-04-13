from flask import Flask
from flask_cors import CORS
from flask_login import LoginManager
from .models import db, User

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Load app configuration
    app.config.from_pyfile("../config.py")

    # Enable CORS
    CORS(app)

    # Initialize the database
    db.init_app(app)

    # Set up login manager
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
