from flask import Blueprint, request, jsonify
import requests, os, json
from datetime import datetime
import numpy as np
from transformers import pipeline
from flask_login import current_user, login_required
from collections import defaultdict

from .utils import extract_adjectives_and_competitors, fetch_competitor_names
from .models import db, ReviewHistory, SentimentSnapshot, CompetitorCache

api = Blueprint('api', __name__)

USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

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
        return jsonify({"error": "ASIN is required"}), 400

    # Check for existing data
    existing = SentimentSnapshot.query.filter_by(user_id=current_user.id, asin=asin)                                      .order_by(SentimentSnapshot.timestamp.desc())                                      .first()
    existing_dates = set(json.loads(existing.review_dates)) if existing else set()

    # Fetch product metadata
    try:
        meta_res = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(USERNAME, PASSWORD),
            json={"source": "amazon_product", "query": asin, "parse": True}
        )
        product_data = meta_res.json()["results"][0]["content"]
        product_name = product_data.get("title", "Unknown")
        manufacturer = product_data.get("manufacturer", "Unknown")
        price = product_data.get("price", 0.0)
    except Exception:
        return jsonify({"error": "Failed to fetch metadata"}), 500

    # Fetch new reviews (ignore existing dates)
    all_reviews = []
    for page in range(1, 6):
        try:
            resp = requests.post(
                "https://realtime.oxylabs.io/v1/queries",
                auth=(USERNAME, PASSWORD),
                json={
                    "source": "amazon_reviews",
                    "query": asin,
                    "page": page,
                    "context": [{"key": "sort_by", "value": "recent"}],
                    "geo_location": "90210",
                    "parse": True
                }
            )
            data = resp.json()["results"][0]["content"]["reviews"]
            all_reviews.extend(data)
        except Exception:
            continue

    reviews, review_dates, countries, review_meta = [], [], [], []
    for r in all_reviews:
        text = r.get("content", "").strip()
        timestamp = r.get("timestamp", "")
        date = timestamp.split("on")[-1].strip() if "on" in timestamp else timestamp
        try:
            formatted = datetime.strptime(date, "%B %d, %Y").strftime("%Y-%m-%d")
        except:
            formatted = "Unknown"
        if text and formatted not in existing_dates:
            reviews.append(text)
            review_dates.append(formatted)
            countries.append("USA")
            review_meta.append({
                "title": r.get("title", ""),
                "content": text,
                "helpful_count": r.get("helpful_count", 0),
                "country": "USA"
            })

    if not reviews:
        return jsonify(existing.to_dict() if existing else {"message": "No new reviews."})

    try:
        sentiments = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)
        adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    except Exception:
        return jsonify({"error": "NLP failure"}), 500

    try:
        gpt_cache = CompetitorCache.query.filter_by(product_name=product_name, manufacturer=manufacturer).first()
        if gpt_cache:
            try:
                gpt_competitors = json.loads(gpt_cache.names)
                if not isinstance(gpt_competitors, list):
                    gpt_competitors = json.loads(gpt_competitors)
            except Exception as parse_err:
                print(f"[WARNING] Failed to parse GPT cache: {parse_err}")
                gpt_competitors = []
        else:
            gpt_competitors = fetch_competitor_names(product_name, manufacturer)
            db.session.add(CompetitorCache(product_name=product_name, manufacturer=manufacturer, names=json.dumps(gpt_competitors)))
            db.session.commit()
        for comp in gpt_competitors:
            competitor_mentions[comp.lower()] += 1
    except Exception:
        gpt_competitors = []

    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    pos, neg, neu = [], [], []
    country_sent = defaultdict(lambda: {"positive": 0, "negative": 0})

    for i, s in enumerate(sentiments):
        label = LABEL_MAPPING.get(s["label"].upper(), "NEUTRAL")
        score = s["score"] * 10
        counts[label] += 1
        if label == "POSITIVE":
            pos.append(score)
            country_sent[countries[i]]["positive"] += 1
        elif label == "NEGATIVE":
            neg.append(score)
            country_sent[countries[i]]["negative"] += 1
        else:
            neu.append(score)

    all_scores = pos + neg + neu
    median = round(np.median([x for x in all_scores if x > 0]), 2)
    total = sum(counts.values()) or 1
    pos_pct = round((counts["POSITIVE"] / total) * 100, 2)
    neg_pct = round((counts["NEGATIVE"] / total) * 100, 2)
    neu_pct = round((counts["NEUTRAL"] / total) * 100, 2)
    top_helpful = sorted(review_meta, key=lambda x: x.get("helpful_count", 0), reverse=True)[:3]

    try:
        db.session.add(ReviewHistory(asin=asin, user_id=current_user.id))
        new_snapshot = SentimentSnapshot(
            asin=asin,
            user_id=current_user.id,
            product_name=product_name,
            manufacturer=manufacturer,
            price=price,
            median_score=median,
            top_adjectives=json.dumps(adjectives),
            competitor_mentions=json.dumps(dict(competitor_mentions)),
            gpt_competitors=json.dumps(gpt_competitors),
            review_dates=json.dumps(review_dates),
            positive_scores=json.dumps(pos),
            negative_scores=json.dumps(neg),
            neutral_scores=json.dumps(neu),
            positive_percentage=pos_pct,
            negative_percentage=neg_pct,
            neutral_percentage=neu_pct,
            country_sentiment=json.dumps(dict(country_sent)),
            top_helpful_reviews=json.dumps(top_helpful)
        )
        db.session.add(new_snapshot)
        db.session.commit()
        return jsonify(new_snapshot.to_dict())
    except Exception:
        db.session.rollback()
        return jsonify({"error": "DB commit failed."}), 500
