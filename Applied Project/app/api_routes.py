from flask import Blueprint, request, jsonify
from flask_login import current_user, login_required
import requests
import os
import numpy as np
from datetime import datetime
from transformers import pipeline
from .utils import extract_adjectives_and_competitors
from .models import db, ReviewHistory, FavoriteASIN

api = Blueprint('api', __name__)

# Load Oxylabs credentials
USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")

# Load Sentiment Model
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Label mapping
LABEL_MAPPING = {
    "LABEL_0": "VERY NEGATIVE",
    "LABEL_1": "NEGATIVE",
    "LABEL_2": "NEUTRAL",
    "LABEL_3": "POSITIVE",
    "LABEL_4": "VERY POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
    "NEUTRAL": "NEUTRAL"
}

@api.route('/fetch_reviews', methods=['GET'])
@login_required
def fetch_reviews():
    asin = request.args.get('asin')
    if not asin:
        return jsonify({"error": "ASIN is required."}), 400

    # ✅ Save to ReviewHistory
    history_entry = ReviewHistory(asin=asin, user_id=current_user.id)
    db.session.add(history_entry)
    db.session.commit()

    # ------- Get Metadata -------
    meta_payload = {
        'source': 'amazon_product',
        'query': asin,
        'parse': True
    }

    meta_response = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=meta_payload)
    meta_json = meta_response.json()
    product = meta_json['results'][0]['content'] if 'results' in meta_json and meta_json['results'] else {}

    product_name = product.get("title", "Unknown Product")
    manufacturer = product.get("manufacturer", "Unknown")
    price = product.get("price", 0.0)

    # ------- Get Reviews -------
    review_payload = {
        "source": "amazon_reviews",
        "query": asin,
        "page": 1,
        "pages": 5,
        "context": [{"key": "sort_by", "value": "recent"}],
        "geo_location": "90210",
        "parse": True
    }

    review_response = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=review_payload)
    data = review_response.json()
    reviews_data = data.get("results", [{}])[0].get("content", {}).get("reviews", [])

    reviews = []
    review_dates = []
    for r in reviews_data:
        content = r.get("content", "").strip()
        timestamp = r.get("timestamp", "").strip()
        if content:
            reviews.append(content)
            try:
                date = datetime.strptime(timestamp.replace("Reviewed in the United States", "").strip(), "%B %d, %Y").strftime("%Y-%m-%d")
            except:
                date = "Unknown"
            review_dates.append(date)

    if not reviews:
        return jsonify({"error": "No reviews found."}), 404

    # ------- NLP Analysis -------
    results = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)
    top_adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    positive_scores, negative_scores, neutral_scores = [], [], []

    for r in results:
        sentiment = LABEL_MAPPING.get(r["label"].upper(), "NEUTRAL")
        score = r["score"] * 10
        sentiment_counts[sentiment] += 1

        if sentiment == "POSITIVE":
            positive_scores.append(score)
            negative_scores.append(0)
            neutral_scores.append(0)
        elif sentiment == "NEGATIVE":
            positive_scores.append(0)
            negative_scores.append(score)
            neutral_scores.append(0)
        else:
            positive_scores.append(0)
            negative_scores.append(0)
            neutral_scores.append(score)

    all_scores = positive_scores + negative_scores + neutral_scores
    median_score = round(np.median([s for s in all_scores if s > 0]), 2) if all_scores else None

    total_reviews = sum(sentiment_counts.values())
    positive_percentage = round((sentiment_counts["POSITIVE"] / total_reviews) * 100, 2) if total_reviews else 0
    negative_percentage = round((sentiment_counts["NEGATIVE"] / total_reviews) * 100, 2) if total_reviews else 0
    neutral_percentage = round((sentiment_counts["NEUTRAL"] / total_reviews) * 100, 2) if total_reviews else 0

    return jsonify({
        "product_name": product_name,
        "manufacturer": manufacturer,
        "price": price,
        "total_reviews": len(reviews),
        "median_score": median_score,
        "top_adjectives": top_adjectives,
        "competitor_mentions": dict(competitor_mentions),
        "review_dates": review_dates,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "neutral_scores": neutral_scores,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage
    })


# ✅ Add Favorite ASIN
@api.route('/favorite', methods=['POST'])
@login_required
def add_favorite():
    data = request.get_json()
    asin = data.get('asin')
    title = data.get('title')
    price = data.get('price', 0.0)

    if not asin:
        return jsonify({"error": "ASIN is required."}), 400

    existing = FavoriteASIN.query.filter_by(user_id=current_user.id, asin=asin).first()
    if existing:
        return jsonify({"message": "Already in favorites."}), 200

    favorite = FavoriteASIN(
        asin=asin,
        title=title,
        price=price,
        user_id=current_user.id
    )
    db.session.add(favorite)
    db.session.commit()
    return jsonify({"message": "Added to favorites."}), 201


# ✅ Remove Favorite ASIN
@api.route('/favorite/<asin>', methods=['DELETE'])
@login_required
def remove_favorite(asin):
    favorite = FavoriteASIN.query.filter_by(user_id=current_user.id, asin=asin).first()
    if not favorite:
        return jsonify({"error": "Favorite not found."}), 404

    db.session.delete(favorite)
    db.session.commit()
    return jsonify({"message": "Removed from favorites."}), 200
