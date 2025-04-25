import os

# PostgreSQL URI
SQLALCHEMY_DATABASE_URI = "postgresql://sentiments_fr6x_user:k5zqfbhqJZ8p1iD68taYALpavOAn8V44@dpg-cvtuq09r0fns73e1fg3g-a.oregon-postgres.render.com/sentiments_fr6x"


SQLALCHEMY_TRACK_MODIFICATIONS = False

# Secret key for sessions, fallback for local dev
SECRET_KEY = os.getenv("SECRET_KEY", "")

# Optional: turn on debug if needed
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
