from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import current_user
from transformers import pipeline
import requests
import numpy as np
import nltk
import spacy
import spacy.cli
import re
from datetime import datetime
from collections import Counter
import os

# === NLP Resources ===
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("\U0001F501 SpaCy model not found. Downloading...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# === App Configuration ===
app = Flask(__name__)
CORS(app)
app.config.from_pyfile("config.py")

db = SQLAlchemy(app)

# === Load Environment Variables ===
OXYLABS_USERNAME = os.getenv("OXYLABS_USERNAME")
OXYLABS_PASSWORD = os.getenv("OXYLABS_PASSWORD")

print("✅ Flask app initialized.")

# === Sentiment Analyzer ===
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("✅ Sentiment model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")

LABEL_MAPPING = {
    "NEGATIVE": "NEGATIVE",
    "POSITIVE": "POSITIVE",
    "NEUTRAL": "NEUTRAL"
}

# === Models ===
class SentimentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.String(20), nullable=False)
    review = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    score = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)

with app.app_context():
    db.create_all()

# === Helpers ===
def get_reviews_oxylabs(asin, pages=5, sort_by="recent"):
    url = "https://realtime.oxylabs.io/v1/queries"
    all_reviews, all_dates = [], []
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
        try:
            response = requests.post(url, auth=(OXYLABS_USERNAME, OXYLABS_PASSWORD), json=payload)
            data = response.json()
            reviews_data = data.get("results", [{}])[0].get("content", {}).get("reviews", [])
            for r in reviews_data:
                content = r.get("content", "").strip()
                timestamp = r.get("timestamp", "").strip()
                if content and timestamp:
                    all_reviews.append(content)
                    all_dates.append(timestamp)
        except Exception as e:
            print(f"❌ API request failed: {e}")
    return all_reviews, all_dates

def get_product_metadata(asin):
    url = "https://realtime.oxylabs.io/v1/queries"
    payload = {"source": "amazon_product", "query": asin, "parse": True}
    try:
        response = requests.post(url, auth=(OXYLABS_USERNAME, OXYLABS_PASSWORD), json=payload)
        content = response.json().get("results", [{}])[0].get("content", {})
        return {
            "product_name": content.get("title", "Unknown Title"),
            "manufacturer": content.get("manufacturer", "Unknown Manufacturer"),
            "price": content.get("price", "Unknown Price")
        }
    except Exception as e:
        print(f"❌ Metadata fetch failed: {e}")
        return {"product_name": "Unknown", "manufacturer": "Unknown", "price": "Unknown"}

def extract_adjectives_and_competitors(reviews):
    words = nltk.word_tokenize(" ".join(reviews).lower())
    tagged_words = nltk.pos_tag(words)
    adjectives = [w for w, t in tagged_words if t in ["JJ", "JJR", "JJS"] and w.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    adjectives = [adj for adj in adjectives if adj not in stopwords]
    top_adjectives = Counter(adjectives).most_common(10)
    competitors = ["nivea", "neutrogena", "eucerin", "cetaphil", "cerave", "aveeno", "olay", "lubriderm", "dove", "gold bond"]
    mentions = {brand: 0 for brand in competitors}
    for w in words:
        if w in mentions:
            mentions[w] += 1
    mentions = {k: v for k, v in mentions.items() if v > 0}
    return top_adjectives, mentions

def parse_review_date(d):
    try:
        return datetime.strptime(d.replace("Reviewed in the United States", "").strip(), "%B %d, %Y").strftime("%Y-%m-%d")
    except:
        return "Unknown"

# === Routes ===
@app.route('/')
def index():
    return render_template("dashboard.html") if current_user.is_authenticated else render_template("index.html")

@app.route('/fetch_reviews', methods=['GET'])
def fetch_reviews():
    asin = request.args.get('asin')
    if not asin:
        return jsonify({"error": "ASIN is required."}), 400

    reviews, dates = get_reviews_oxylabs(asin)
    if not reviews:
        return jsonify({"error": "No reviews found."}), 404

    metadata = get_product_metadata(asin)
    parsed_dates = [parse_review_date(d) for d in dates]
    top_adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    results = sentiment_analyzer(reviews, truncation=True, max_length=512, padding=True, batch_size=8)

    pos_scores, neg_scores, neu_scores, scores = [], [], [], []
    for review, result, date in zip(reviews, results, parsed_dates):
        sentiment = LABEL_MAPPING.get(result['label'].upper(), "NEUTRAL")
        score = result['score'] * 10
        scores.append(score)

        pos_scores.append(score if sentiment == "POSITIVE" else 0)
        neg_scores.append(score if sentiment == "NEGATIVE" else 0)
        neu_scores.append(score if sentiment == "NEUTRAL" else 0)

        db.session.add(SentimentRecord(
            asin=asin,
            review=review,
            sentiment=sentiment,
            score=score,
            date=date,
            user_id=current_user.id if current_user.is_authenticated else None
        ))
    db.session.commit()

    median_score = round(np.median(scores), 2) if scores else None
    total = len(scores)

    def score_pct(score_list):
        return round((sum(1 for s in score_list if s > 0) / total) * 100, 2) if total else 0

    return jsonify({
        "product_name": metadata["product_name"],
        "manufacturer": metadata["manufacturer"],
        "price": metadata["price"],
        "total_reviews": total,
        "median_score": median_score,
        "top_adjectives": top_adjectives,
        "competitor_mentions": competitor_mentions,
        "review_dates": parsed_dates,
        "positive_scores": pos_scores,
        "negative_scores": neg_scores,
        "neutral_scores": neu_scores,
        "positive_percentage": score_pct(pos_scores),
        "negative_percentage": score_pct(neg_scores),
        "neutral_percentage": score_pct(neu_scores)
    })
