from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

# ---------------------
# User Table
# ---------------------
class User(UserMixin, db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    # Relationships
    history = db.relationship('ReviewHistory', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    favorites = db.relationship('FavoriteASIN', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    snapshots = db.relationship('SentimentSnapshot', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# ---------------------
# Review History
# ---------------------
class ReviewHistory(db.Model):
    __tablename__ = 'review_history'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    search_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<ReviewHistory ASIN={self.asin} UserID={self.user_id}>"


# ---------------------
# Favorites
# ---------------------
class FavoriteASIN(db.Model):
    __tablename__ = 'favorite_asin'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    added_on = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<FavoriteASIN ASIN={self.asin} UserID={self.user_id}>"


# ---------------------
# Cached Sentiment Snapshots
# ---------------------
class SentimentSnapshot(db.Model):
    __tablename__ = 'sentiment_snapshot'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_name = db.Column(db.String(300))
    manufacturer = db.Column(db.String(200))
    price = db.Column(db.Float)
    median_score = db.Column(db.Float)
    top_adjectives = db.Column(db.Text)
    competitor_mentions = db.Column(db.Text)
    review_dates = db.Column(db.Text)
    positive_scores = db.Column(db.Text)
    negative_scores = db.Column(db.Text)
    neutral_scores = db.Column(db.Text)
    positive_percentage = db.Column(db.Float)
    negative_percentage = db.Column(db.Float)
    neutral_percentage = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f"<Snapshot {self.asin} by User {self.user_id}>"

    def to_dict(self):
        def _safe_load(field):
            try:
                return json.loads(field or "[]")
            except json.JSONDecodeError:
                return []

        return {
            "product_name": self.product_name,
            "manufacturer": self.manufacturer,
            "price": self.price,
            "median_score": self.median_score,
            "top_adjectives": _safe_load(self.top_adjectives),
            "competitor_mentions": _safe_load(self.competitor_mentions),
            "review_dates": _safe_load(self.review_dates),
            "positive_scores": _safe_load(self.positive_scores),
            "negative_scores": _safe_load(self.negative_scores),
            "neutral_scores": _safe_load(self.neutral_scores),
            "positive_percentage": self.positive_percentage,
            "negative_percentage": self.negative_percentage,
            "neutral_percentage": self.neutral_percentage
        }
