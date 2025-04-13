import os

# If deployed on Render with PostgreSQL, use that; fallback to local SQLite for development
SQLALCHEMY_DATABASE_URI = os.getenv(
    "DATABASE_URL",  # PostgreSQL URI (Render)
    f"sqlite:///{os.path.join(os.path.abspath(os.path.dirname(__file__)), 'sentiments.db')}"  # Local fallback
)

# Automatically fix PostgreSQL URI format if needed (Render sometimes gives `postgres://` which is deprecated)
if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

SQLALCHEMY_TRACK_MODIFICATIONS = False

# Secret key for sessions, fallback for local dev
SECRET_KEY = os.getenv("SECRET_KEY", "")

# Optional: turn on debug if needed
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
