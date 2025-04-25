import os
import re
import json
import hashlib
import traceback
from datetime import datetime
from collections import defaultdict, Counter

import requests
import numpy as np
from flask import Blueprint, request, jsonify
from flask_login import current_user, login_required

from .models import db, ReviewHistory, SentimentSnapshot, CompetitorCache
from .utils import (
    extract_adjectives_and_competitors,
    fetch_competitor_names,
    get_nlp,
    get_sentiment_pipeline,
)

api = Blueprint('api', __name__)

USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")

LABEL_MAPPING = {
    "LABEL_0": "VERY NEGATIVE", "LABEL_1": "NEGATIVE", "LABEL_2": "NEUTRAL",
    "LABEL_3": "POSITIVE", "LABEL_4": "VERY POSITIVE",
    "NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE", "NEUTRAL": "NEUTRAL"
}


def compute_review_hashes_and_filter(all_reviews, existing_snapshot):
    reviews, review_dates, countries, review_meta = [], [], [], []
    existing_hashes = set()

    if existing_snapshot:
        try:
            for rev in json.loads(existing_snapshot.top_helpful_reviews):
                h = hashlib.sha256(rev.get("content", "").strip().encode()).hexdigest()
                existing_hashes.add(h)
        except Exception as e:
            print("[WARNING] Failed to load existing hashes:", e)

    for r in all_reviews:
        text = r.get("content", "").strip()
        timestamp = r.get("timestamp", "").strip()
        country = "USA"
        formatted = "Unknown"

        try:
            if "Reviewed in" in timestamp and "on" in timestamp:
                country = timestamp.split("Reviewed in")[1].split("on")[0].strip()
                date = timestamp.split("on")[-1].strip()
                formatted = datetime.strptime(date, "%B %d, %Y").strftime("%Y-%m-%d")
            else:
                match = re.search(r'on\s([A-Za-z]+\s\d{1,2},\s\d{4})', timestamp)
                if match:
                    date_str = match.group(1)
                    formatted = datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[WARNING] Failed to parse timestamp: {timestamp} -> {e}")

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if text and content_hash not in existing_hashes:
            reviews.append(text)
            review_dates.append(formatted)
            countries.append(country)
            review_meta.append({
                "title": r.get("title", ""),
                "content": text,
                "helpful_count": r.get("helpful_count", 0),
                "country": country
            })
            existing_hashes.add(content_hash)

    return reviews, review_dates, countries, review_meta


def snapshot_to_dict(snapshot):
    """Safely deserialize all fields for clean frontend consumption."""
    return {
        "asin": snapshot.asin,
        "product_name": snapshot.product_name,
        "manufacturer": snapshot.manufacturer,
        "price": snapshot.price,
        "median_score": snapshot.median_score,
        "top_adjectives": json.loads(snapshot.top_adjectives or "[]"),
        "competitor_mentions": json.loads(snapshot.competitor_mentions or "{}"),
        "gpt_competitors": json.loads(snapshot.gpt_competitors or "[]"),
        "review_dates": json.loads(snapshot.review_dates or "[]"),
        "positive_scores": json.loads(snapshot.positive_scores or "[]"),
        "negative_scores": json.loads(snapshot.negative_scores or "[]"),
        "neutral_scores": json.loads(snapshot.neutral_scores or "[]"),
        "positive_percentage": snapshot.positive_percentage,
        "negative_percentage": snapshot.negative_percentage,
        "neutral_percentage": snapshot.neutral_percentage,
        "country_sentiment": json.loads(snapshot.country_sentiment or "{}"),
        "top_helpful_reviews": json.loads(snapshot.top_helpful_reviews or "[]")
    }


@api.route('/fetch_reviews', methods=['GET'])
@login_required
def fetch_reviews():
    try:
        asin = request.args.get('asin')
        if not asin:
            return jsonify({"error": "ASIN is required"}), 400

        existing = SentimentSnapshot.query.filter_by(
            user_id=current_user.id, asin=asin
        ).order_by(SentimentSnapshot.timestamp.desc()).first()

        try:
            meta = requests.post(
                "https://realtime.oxylabs.io/v1/queries",
                auth=(USERNAME, PASSWORD),
                json={"source": "amazon_product", "query": asin, "parse": True}
            )
            product_data = meta.json()["results"][0]["content"]
            product_name = product_data.get("title", "Unknown")
            manufacturer = product_data.get("manufacturer", "Unknown")
            price = product_data.get("price", 0.0)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "Failed to fetch metadata"}), 500

        all_reviews = []
        for page in range(1, 6):
            try:
                resp = requests.post(
                    "https://realtime.oxylabs.io/v1/queries",
                    auth=(USERNAME, PASSWORD),
                    json={
                        "source": "amazon_reviews", "query": asin,
                        "page": page, "context": [{"key": "sort_by", "value": "recent"}],
                        "geo_location": "90210", "parse": True
                    }
                )
                all_reviews.extend(resp.json()["results"][0]["content"]["reviews"])
            except Exception as e:
                print(f"[ERROR] Failed to fetch page {page} reviews: {e}")
                continue

        reviews, review_dates, countries, review_meta = compute_review_hashes_and_filter(all_reviews, existing)
        if not reviews:
            return jsonify(snapshot_to_dict(existing) if existing else {"message": "No new reviews."})

        try:
            sentiment_analyzer = get_sentiment_pipeline()
            print("🔍 Performing sentiment analysis on", len(reviews), "reviews...")
            sentiments = sentiment_analyzer(
                reviews, truncation=True, max_length=512, padding=True, batch_size=8
            )

            nlp = get_nlp()
            print("🔍 Extracting adjectives and competitors...")
            adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews, nlp)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": "NLP failure"}), 500

        try:
            gpt_cache = CompetitorCache.query.filter_by(
                product_name=product_name, manufacturer=manufacturer
            ).first()
            if gpt_cache:
                gpt_competitors = json.loads(gpt_cache.names)
            else:
                gpt_competitors = fetch_competitor_names(product_name, manufacturer)
                if gpt_competitors:
                    db.session.add(CompetitorCache(
                        product_name=product_name,
                        manufacturer=manufacturer,
                        names=json.dumps(gpt_competitors)
                    ))
                    db.session.commit()

            for comp in gpt_competitors:
                competitor_mentions[comp.lower()] += 1
        except Exception as e:
            print("[WARNING] GPT fallback failed:", e)
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

        snapshot = SentimentSnapshot(
            asin=asin,
            user_id=current_user.id,
            product_name=product_name,
            manufacturer=manufacturer,
            price=price,
            median_score=median,
            top_adjectives=json.dumps(adjectives),
            competitor_mentions=json.dumps(competitor_mentions),
            gpt_competitors=json.dumps(gpt_competitors),
            review_dates=json.dumps(review_dates),
            positive_scores=json.dumps(pos),
            negative_scores=json.dumps(neg),
            neutral_scores=json.dumps(neu),
            positive_percentage=pos_pct,
            negative_percentage=neg_pct,
            neutral_percentage=neu_pct,
            country_sentiment=json.dumps(country_sent),
            top_helpful_reviews=json.dumps(top_helpful)
        )
        db.session.add(ReviewHistory(asin=asin, user_id=current_user.id))
        db.session.add(snapshot)
        db.session.commit()

        return jsonify(snapshot_to_dict(snapshot))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500
