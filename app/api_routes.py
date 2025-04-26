import os
import re
import json
import hashlib
import traceback
from datetime import datetime
from collections import defaultdict, Counter
import math

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
    compute_review_hashes_and_filter
)

api = Blueprint('api', __name__)

USERNAME = os.getenv("OXYLABS_USERNAME")
PASSWORD = os.getenv("OXYLABS_PASSWORD")
LABEL_MAPPING = {
    "LABEL_0": "VERY NEGATIVE", "LABEL_1": "NEGATIVE", "LABEL_2": "NEUTRAL",
    "LABEL_3": "POSITIVE", "LABEL_4": "VERY POSITIVE",
    "NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE", "NEUTRAL": "NEUTRAL"
}

def snapshot_to_dict(snapshot, total_reviews_scraped=None):
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
        "top_helpful_reviews": json.loads(snapshot.top_helpful_reviews or "[]"),
        "total_reviews_scraped": total_reviews_scraped
    }

@api.route('/fetch_reviews', methods=['GET'])
@login_required
def fetch_reviews():
    try:
        asin = request.args.get('asin')
        count = int(request.args.get('count', 50))

        if not asin:
            return jsonify({"error": "ASIN is required"}), 400
        if count < 1 or count > 500:
            return jsonify({"error": "Review count must be between 1 and 500"}), 400

        existing = SentimentSnapshot.query.filter_by(
            user_id=current_user.id, asin=asin
        ).order_by(SentimentSnapshot.timestamp.desc()).first()

        meta = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(USERNAME, PASSWORD),
            json={"source": "amazon_product", "query": asin, "parse": True}
        )
        product_data = meta.json()["results"][0]["content"]
        product_name = product_data.get("title", "Unknown")
        manufacturer = product_data.get("manufacturer", "Unknown")
        price = product_data.get("price", 0.0)

        all_reviews = []
        pages = math.ceil(count / 5)
        sort_by = "recent"
        print(f"🔎 Need to fetch {count} reviews -> Estimating {pages} pages...")

        for page in range(1, pages + 1):
            payload = {
                "source": "amazon_reviews",
                "query": asin,
                "page": page,
                "pages": 1,
                "context": [{"key": "sort_by", "value": sort_by}],
                "geo_location": "90210",
                "parse": True
            }
            resp = requests.post(
                "https://realtime.oxylabs.io/v1/queries",
                auth=(USERNAME, PASSWORD),
                json=payload
            )
            page_reviews = resp.json().get("results", [])[0].get("content", {}).get("reviews", [])
            print(f"[DEBUG] Page {page} fetched {len(page_reviews)} reviews.")
            if not page_reviews:
                break
            all_reviews.extend(page_reviews)
            if len(all_reviews) >= count:
                break

        print(f"[DEBUG] Total reviews collected: {len(all_reviews)}")

        reviews, review_dates, countries, review_meta = compute_review_hashes_and_filter(all_reviews, existing)

        # Trim if needed
        reviews = reviews[:count]
        review_dates = review_dates[:count]
        countries = countries[:count]
        review_meta = review_meta[:count]

        if not reviews:
            return jsonify(snapshot_to_dict(existing, total_reviews_scraped=len(all_reviews)) if existing else {"message": "No new reviews."})

        # NLP Analysis
        sentiment_analyzer = get_sentiment_pipeline()
        sentiments = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)

        nlp = get_nlp()
        adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews, nlp)

        # GPT competitors
        gpt_cache = CompetitorCache.query.filter_by(product_name=product_name, manufacturer=manufacturer).first()
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
            competitor_mentions[comp.lower()] = competitor_mentions.get(comp.lower(), 0) + 1

        # Sentiment aggregation
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

        review_dates_sorted = sorted(review_dates, key=lambda x: (x != "Unknown", x))
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
            review_dates=json.dumps(review_dates_sorted),
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

        return jsonify(snapshot_to_dict(snapshot, total_reviews_scraped=len(all_reviews)))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500
