import json
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.sql import func

db = SQLAlchemy()

# ==============================
# User Table
# ==============================
class User(UserMixin, db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    history = db.relationship('ReviewHistory', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    favorites = db.relationship('FavoriteASIN', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    snapshots = db.relationship('SentimentSnapshot', backref='user', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.email}>"

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# ==============================
# Review History Table
# ==============================
class ReviewHistory(db.Model):
    __tablename__ = 'review_history'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False, index=True)
    search_date = db.Column(db.DateTime(timezone=True), server_default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"<ReviewHistory ASIN={self.asin} UserID={self.user_id}>"


# ==============================
# Favorite ASINs Table
# ==============================
class FavoriteASIN(db.Model):
    __tablename__ = 'favorite_asin'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    added_on = db.Column(db.DateTime(timezone=True), server_default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'asin', name='uq_user_asin'),
    )

    def __repr__(self):
        return f"<FavoriteASIN ASIN={self.asin} UserID={self.user_id}>"


# ==============================
# Sentiment Snapshots Table (Cached)
# ==============================
class SentimentSnapshot(db.Model):
    __tablename__ = 'sentiment_snapshot'

    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_name = db.Column(db.String(300))
    manufacturer = db.Column(db.String(200))
    price = db.Column(db.Float)
    median_score = db.Column(db.Float)

    # JSON fields
    top_adjectives = db.Column(db.Text)
    competitor_mentions = db.Column(db.Text)
    gpt_competitors = db.Column(db.Text)
    review_dates = db.Column(db.Text)
    positive_scores = db.Column(db.Text)
    negative_scores = db.Column(db.Text)
    neutral_scores = db.Column(db.Text)
    country_sentiment = db.Column(db.Text)
    top_helpful_reviews = db.Column(db.Text)
    total_reviews_scraped = db.Column(db.Integer, default=0)

    positive_percentage = db.Column(db.Float)
    negative_percentage = db.Column(db.Float)
    neutral_percentage = db.Column(db.Float)

    timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.Index('ix_snapshot_user_asin', 'user_id', 'asin'),
    )

    def __repr__(self):
        return f"<Snapshot ASIN={self.asin} UserID={self.user_id}>"

    def to_dict(self):
        """Deserialize all text fields for API response."""

        def _safe_load(field, default="[]"):
            try:
                return json.loads(field or default)
            except (json.JSONDecodeError, TypeError):
                return json.loads(default)

        return {
            "id": self.id,
            "asin": self.asin,
            "product_name": self.product_name,
            "manufacturer": self.manufacturer,
            "price": self.price,
            "median_score": self.median_score,
            "top_adjectives": _safe_load(self.top_adjectives, "{}"),
            "competitor_mentions": _safe_load(self.competitor_mentions, "{}"),
            "gpt_competitors": _safe_load(self.gpt_competitors),
            "review_dates": _safe_load(self.review_dates),
            "positive_scores": _safe_load(self.positive_scores),
            "negative_scores": _safe_load(self.negative_scores),
            "neutral_scores": _safe_load(self.neutral_scores),
            "positive_percentage": self.positive_percentage,
            "negative_percentage": self.negative_percentage,
            "neutral_percentage": self.neutral_percentage,
            "country_sentiment": _safe_load(self.country_sentiment, "{}"),
            "top_helpful_reviews": _safe_load(self.top_helpful_reviews)
        }


# ==============================
# GPT Competitor Cache
# ==============================
class CompetitorCache(db.Model):
    __tablename__ = 'competitor_cache'

    id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(300), nullable=False)
    manufacturer = db.Column(db.String(200), nullable=False)
    names = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<GPTCache {self.product_name} by {self.manufacturer}>"
