from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# User table
class User(UserMixin, db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships
    history = db.relationship('ReviewHistory', backref='user', lazy=True, cascade="all, delete-orphan")
    favorites = db.relationship('FavoriteASIN', backref='user', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"


# Stores each product search by the user
class ReviewHistory(db.Model):
    __tablename__ = 'review_history'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    search_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<ReviewHistory ASIN={self.asin} UserID={self.user_id}>"


# Stores user-marked favorite ASINs with metadata
class FavoriteASIN(db.Model):
    __tablename__ = 'favorite_asin'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(512), nullable=True)
    price = db.Column(db.Float, nullable=True)
    added_on = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<FavoriteASIN ASIN={self.asin} Title={self.title} Price={self.price} UserID={self.user_id}>"
