from flask import Blueprint, request, jsonify
import requests, os, json, hashlib, re
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

def compute_review_hashes_and_filter(all_reviews, existing_snapshot):
    print("🔎 Filtering new reviews based on hashes and dates...")
    reviews, review_dates, countries, review_meta = [], [], [], []
    existing_dates = set(json.loads(existing_snapshot.review_dates)) if existing_snapshot else set()
    existing_hashes = set()

    if existing_snapshot:
        try:
            old_reviews = json.loads(existing_snapshot.top_helpful_reviews)
            for rev in old_reviews:
                content_hash = hashlib.sha256(rev.get("content", "").strip().encode()).hexdigest()
                existing_hashes.add(content_hash)
            print(f"✅ Loaded {len(existing_hashes)} existing hashes.")
        except Exception as e:
            print(f"[WARNING] Failed to extract existing content hashes: {e}")

    for r in all_reviews:
        text = r.get("content", "").strip()
        timestamp = r.get("timestamp", "").strip()
        country = "USA"
        formatted = "Unknown"

        if "Reviewed in" in timestamp and "on" in timestamp:
            try:
                country = timestamp.split("Reviewed in")[1].split("on")[0].strip()
                date = timestamp.split("on")[-1].strip()
                formatted = datetime.strptime(date, "%B %d, %Y").strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[WARNING] Failed to parse date from standard format: {timestamp} | {e}")
        else:
            # Attempt regex fallback
            try:
                date_match = re.search(r'on\s([A-Za-z]+\s\d{1,2},\s\d{4})', timestamp)
                if date_match:
                    date_str = date_match.group(1)
                    formatted = datetime.strptime(date_str, "%B %d, %Y").strftime("%Y-%m-%d")
            except Exception as e:
                print(f"[WARNING] Regex fallback failed for timestamp: {timestamp} | {e}")

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

    print(f"🧾 {len(reviews)} new reviews identified.")
    return reviews, review_dates, countries, review_meta

@api.route('/fetch_reviews', methods=['GET'])
@login_required
def fetch_reviews():
    asin = request.args.get('asin')
    print(f"🔍 Requested ASIN: {asin}")
    if not asin:
        return jsonify({"error": "ASIN is required"}), 400

    print("📦 Checking existing snapshots...")
    existing = SentimentSnapshot.query.filter_by(user_id=current_user.id, asin=asin)\
                                      .order_by(SentimentSnapshot.timestamp.desc()).first()

    try:
        print("📦 Fetching product metadata...")
        meta_res = requests.post(
            "https://realtime.oxylabs.io/v1/queries",
            auth=(USERNAME, PASSWORD),
            json={"source": "amazon_product", "query": asin, "parse": True}
        )
        product_data = meta_res.json()["results"][0]["content"]
        product_name = product_data.get("title", "Unknown")
        manufacturer = product_data.get("manufacturer", "Unknown")
        price = product_data.get("price", 0.0)
        print(f"🛍️ Product: {product_name} | Manufacturer: {manufacturer} | Price: {price}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch metadata: {e}")
        return jsonify({"error": "Failed to fetch metadata"}), 500

    print("🔄 Fetching recent reviews...")
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
            print(f"📄 Page {page} reviews fetched: {len(data)}")
            all_reviews.extend(data)
        except Exception as e:
            print(f"[ERROR] Failed to fetch page {page} reviews: {e}")
            continue

    reviews, review_dates, countries, review_meta = compute_review_hashes_and_filter(all_reviews, existing)

    if not reviews:
        print("🛑 No new reviews to analyze.")
        return jsonify(existing.to_dict() if existing else {"message": "No new reviews."})

    try:
        print("🤖 Running sentiment analysis...")
        sentiments = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)
        print("📌 Sentiment analysis completed.")
        adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    except Exception as e:
        print(f"[ERROR] NLP processing failed: {e}")
        return jsonify({"error": "NLP failure"}), 500

    try:
        print("💡 Fetching GPT competitor suggestions...")
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
            if gpt_competitors:
                db.session.add(CompetitorCache(
                    product_name=product_name,
                    manufacturer=manufacturer,
                    names=json.dumps(gpt_competitors)
                ))
                db.session.commit()
            else:
                print("[INFO] GPT returned no competitor names. Skipping cache save.")
        for comp in gpt_competitors:
            competitor_mentions[comp.lower()] += 1
    except Exception as e:
        print(f"[WARNING] GPT competitor fallback: {e}")
        gpt_competitors = []

    print("📊 Aggregating sentiment metrics...")
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

    print(f"📈 POS: {pos_pct}% | NEG: {neg_pct}% | NEU: {neu_pct}% | Median Score: {median}")

    try:
        print("💾 Saving snapshot to database...")
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
        print("✅ Snapshot saved successfully.")
        return jsonify(new_snapshot.to_dict())
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Failed to save snapshot: {e}")
        return jsonify({"error": "DB commit failed."}), 500
