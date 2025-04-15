from flask import Blueprint, request, jsonify
import requests
import os
import numpy as np
from datetime import datetime
from transformers import pipeline
from flask_login import current_user, login_required
from .utils import extract_adjectives_and_competitors
from .models import db, ReviewHistory, SentimentSnapshot
import json

api = Blueprint('api', __name__)

USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")

print(f"[DEBUG] Loaded Oxylabs USERNAME: {USERNAME}")
print(f"[DEBUG] Loaded Oxylabs PASSWORD: {PASSWORD}")

sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

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
        print("[DEBUG] ❌ No ASIN provided in request.")
        return jsonify({"error": "ASIN is required."}), 400

    print(f"[DEBUG] 📦 ASIN requested: {asin} by user {current_user.id}")

    snapshot = SentimentSnapshot.query.filter_by(asin=asin, user_id=current_user.id).order_by(SentimentSnapshot.timestamp.desc()).first()
    if snapshot:
        print("[DEBUG] ✅ Returning cached snapshot for ASIN:", asin)
        return jsonify(snapshot.to_dict())

    meta_payload = {
        'source': 'amazon_product',
        'query': asin,
        'parse': True
    }
    meta_response = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=meta_payload)
    print("[DEBUG] 🔄 Metadata status:", meta_response.status_code)

    if meta_response.status_code != 200:
        print("[DEBUG] ❌ Failed to fetch metadata.")
        return jsonify({"error": "Failed to fetch product metadata."}), 401

    meta_json = meta_response.json()
    print("[DEBUG] 📦 Metadata response received:", json.dumps(meta_json, indent=2))
    product = meta_json.get('results', [{}])[0].get('content', {})
    product_name = product.get("title", "Unknown Product")
    manufacturer = product.get("manufacturer", "Unknown")
    price = product.get("price", 0.0)

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
    print("[DEBUG] 🔄 Reviews status:", review_response.status_code)

    if review_response.status_code != 200:
        print("[DEBUG] ❌ Failed to fetch reviews.")
        return jsonify({"error": "Failed to fetch reviews."}), 401

    review_json = review_response.json()
    print("[DEBUG] 📦 Review response JSON:", json.dumps(review_json, indent=2))

    reviews_data = review_json.get("results", [{}])[0].get("content", {}).get("reviews", [])
    reviews, review_dates = [], []

    for r in reviews_data:
        content = r.get("content", "").strip()
        timestamp = r.get("timestamp", "").strip()
        if content:
            reviews.append(content)
            try:
                date = datetime.strptime(timestamp.replace("Reviewed in the United States", "").strip(), "%B %d, %Y").strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[DEBUG] ⚠️ Date parse failed: {e}")
                date = "Unknown"
            review_dates.append(date)

    if not reviews:
        print("[DEBUG] ❌ No reviews parsed. Raw response:", json.dumps(review_json, indent=2))
        return jsonify({"error": "No reviews found."}), 404

    print(f"[DEBUG] ✅ Parsed {len(reviews)} reviews")

    results = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)
    print("[DEBUG] 🤖 Sentiment analysis results:", results)

    top_adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    print(f"[DEBUG] 📈 Top adjectives: {top_adjectives}")
    print(f"[DEBUG] 🏷 Competitor mentions: {competitor_mentions}")

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

    try:
        history_entry = ReviewHistory(asin=asin, user_id=current_user.id)
        db.session.add(history_entry)

        snapshot = SentimentSnapshot(
            asin=asin,
            user_id=current_user.id,
            product_name=product_name,
            manufacturer=manufacturer,
            price=price,
            median_score=median_score,
            top_adjectives=json.dumps(top_adjectives),
            competitor_mentions=json.dumps(dict(competitor_mentions)),
            review_dates=json.dumps(review_dates),
            positive_scores=json.dumps(positive_scores),
            negative_scores=json.dumps(negative_scores),
            neutral_scores=json.dumps(neutral_scores),
            positive_percentage=positive_percentage,
            negative_percentage=negative_percentage,
            neutral_percentage=neutral_percentage
        )
        db.session.add(snapshot)
        db.session.commit()
        print("[DEBUG] ✅ Snapshot saved to database.")
    except Exception as e:
        db.session.rollback()
        print(f"[DEBUG] ❌ Error saving snapshot to DB: {e}")
        return jsonify({"error": "Failed to save analysis."}), 500

    return jsonify(snapshot.to_dict())


@api.route('/debug-oxylabs-auth')
def debug_oxylabs_auth():
    if not USERNAME or not PASSWORD:
        return jsonify({"error": "Missing credentials"}), 400

    test_payload = {
        "source": "amazon_product",
        "query": "B0BTRWBVTH",
        "parse": True
    }
    r = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=test_payload)
    try:
        return jsonify({
            "status_code": r.status_code,
            "response": r.json() if r.status_code != 401 else "Unauthorized"
        })
    except Exception as e:
        return jsonify({"status_code": r.status_code, "error": str(e)})
