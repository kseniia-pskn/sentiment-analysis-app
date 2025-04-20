from flask import Blueprint, request, jsonify
import requests
import os
import numpy as np
from datetime import datetime
from transformers import pipeline
from flask_login import current_user, login_required
from .utils import extract_adjectives_and_competitors
from .models import db, ReviewHistory, SentimentSnapshot
from collections import defaultdict
import json

api = Blueprint('api', __name__)

USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")

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
        return jsonify({"error": "ASIN is required."}), 400

    try:
        snapshot = SentimentSnapshot.query.filter_by(asin=asin, user_id=current_user.id).order_by(SentimentSnapshot.timestamp.desc()).first()
        if snapshot:
            return jsonify(snapshot.to_dict())
    except Exception as db_err:
        print("[ERROR] Snapshot retrieval failed:", db_err)

    meta_payload = {
        'source': 'amazon_product',
        'query': asin,
        'parse': True
    }
    meta_response = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=meta_payload)
    if meta_response.status_code != 200:
        return jsonify({"error": "Failed to fetch product metadata."}), 401

    product = meta_response.json().get('results', [{}])[0].get('content', {})
    product_name = product.get("title", "Unknown Product")
    manufacturer = product.get("manufacturer", "Unknown")
    price = product.get("price", 0.0)

    reviews_data = []
    for page in range(1, 6):
        review_payload = {
            "source": "amazon_reviews",
            "query": asin,
            "page": page,
            "context": [{"key": "sort_by", "value": "recent"}],
            "geo_location": "90210",
            "parse": True
        }
        try:
            response = requests.post("https://realtime.oxylabs.io/v1/queries", auth=(USERNAME, PASSWORD), json=review_payload)
            if response.status_code == 200:
                page_reviews = response.json().get("results", [{}])[0].get("content", {}).get("reviews", [])
                reviews_data.extend(page_reviews)
        except Exception as review_err:
            print("[ERROR] Review page fetch failed:", review_err)

    reviews, review_dates, countries = [], [], []
    review_meta = []

    for r in reviews_data:
        content = r.get("content", "").strip()
        timestamp = r.get("timestamp", "").strip()
        country = "USA"
        if "Reviewed in" in timestamp:
            try:
                country = timestamp.split("Reviewed in")[-1].split("on")[0].strip()
                date = timestamp.split("on")[-1].strip()
            except:
                date = "Unknown"
        else:
            date = timestamp

        try:
            formatted_date = datetime.strptime(date, "%B %d, %Y").strftime("%Y-%m-%d")
        except:
            formatted_date = "Unknown"

        if content:
            reviews.append(content)
            review_dates.append(formatted_date)
            countries.append(country)
            review_meta.append({
                "title": r.get("title", ""),
                "content": content,
                "helpful_count": r.get("helpful_count", 0),
                "country": country
            })

    if not reviews:
        return jsonify({"error": "No reviews found."}), 404

    try:
        results = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)
    except Exception as sa_err:
        print("[ERROR] Sentiment analysis failed:", sa_err)
        return jsonify({"error": "Sentiment analysis failed."}), 500

    try:
        top_adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    except Exception as nlp_err:
        print("[ERROR] Tokenization failed:", nlp_err)
        top_adjectives, competitor_mentions = [], {}

    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    positive_scores, negative_scores, neutral_scores = [], [], []

    country_sentiment = defaultdict(lambda: {"positive": 0, "negative": 0})

    for i, r in enumerate(results):
        sentiment = LABEL_MAPPING.get(r["label"].upper(), "NEUTRAL")
        score = r["score"] * 10
        sentiment_counts[sentiment] += 1
        country = countries[i]

        if sentiment == "POSITIVE":
            positive_scores.append(score)
            negative_scores.append(0)
            neutral_scores.append(0)
            country_sentiment[country]["positive"] += 1
        elif sentiment == "NEGATIVE":
            positive_scores.append(0)
            negative_scores.append(score)
            neutral_scores.append(0)
            country_sentiment[country]["negative"] += 1
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

    top_helpful_reviews = sorted(review_meta, key=lambda x: x.get("helpful_count", 0), reverse=True)[:3]

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
            neutral_percentage=neutral_percentage,
            country_sentiment=json.dumps(dict(country_sentiment)),
            top_helpful_reviews=json.dumps(top_helpful_reviews)
        )
        db.session.add(snapshot)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("[ERROR] Failed to save snapshot:", e)
        return jsonify({"error": "Failed to save analysis."}), 500

    return jsonify({
        **snapshot.to_dict(),
        "country_sentiment": dict(country_sentiment),
        "top_helpful_reviews": top_helpful_reviews
    })
