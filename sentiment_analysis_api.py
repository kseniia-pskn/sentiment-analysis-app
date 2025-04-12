from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import requests
import numpy as np
import sqlite3
import nltk
import spacy  # Named Entity Recognition (NER)
import re
from datetime import datetime
from collections import Counter

# Download necessary NLP resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

# Load SpaCy English model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Oxylabs API Credentials (UNCHANGED)
OXYLABS_USERNAME = "kseniia_pskn_yMCCh"
OXYLABS_PASSWORD = "Mello4188317="

print("✅ Flask app initialized.")

# Load sentiment analysis model
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("✅ Sentiment analysis model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")

# Sentiment Label Mapping
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

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect("sentiments.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT,
            review TEXT,
            sentiment TEXT,
            score REAL,
            date TEXT,
            review_id TEXT UNIQUE
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")

init_db()

# Fetch Amazon reviews using Oxylabs API (multi-page support)
def get_reviews_oxylabs(asin, pages=5, sort_by="recent"):
    print(f"🔍 Fetching reviews for ASIN: {asin} | Pages: {pages} | Sort By: {sort_by}")
    url = "https://realtime.oxylabs.io/v1/queries"
    all_reviews = []
    all_dates = []
    
    for page in range(1, pages + 1):
        payload = {
            "source": "amazon_reviews",
            "query": asin,
            "page": page,
            "pages": 1,
            "context": [
                {"key": "sort_by", "value": sort_by}
            ],
            "geo_location": "90210",
            "parse": True
        }
        try:
            response = requests.post(url, auth=(OXYLABS_USERNAME, OXYLABS_PASSWORD), json=payload)
            data = response.json()
            
            if response.status_code != 200:
                print(f"❌ API request failed: {response.status_code}, Response: {data}")
                continue
            
            product_name = data.get("results", [{}])[0].get("content", {}).get("title", "Unknown Product")
            reviews_data = data.get("results", [{}])[0].get("content", {}).get("reviews", [])

            print(f"📌 Page {page} - Found {len(reviews_data)} reviews.")

            for r in reviews_data:
                content = r.get("content", "").strip()
                timestamp = r.get("timestamp", "").strip()
                if content and timestamp:
                    print(f"✅ Review: {content[:100]}... | Date: {timestamp}")  # Only show first 100 chars
                    all_reviews.append(content)
                    all_dates.append(timestamp)

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            continue
    
    print(f"📊 Total reviews collected: {len(all_reviews)}")
    return all_reviews, all_dates, product_name

# Extract adjectives and competitor mentions
def extract_adjectives_and_competitors(reviews):
    words = nltk.word_tokenize(" ".join(reviews).lower())
    tagged_words = nltk.pos_tag(words)
    
    # Keep only proper adjectives
    adjectives = [word for word, tag in tagged_words if tag in ["JJ", "JJR", "JJS"] and word.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    adjectives = [adj for adj in adjectives if adj not in stopwords]
    top_adjectives = Counter(adjectives).most_common(10)

    # Detect competitors dynamically
    competitors = ["Nivea", "Neutrogena", "Eucerin", "Cetaphil", "CeraVe", "Aveeno", "Olay", "Lubriderm", "Dove", "Gold Bond"]
    competitor_mentions = {brand.lower(): 0 for brand in competitors}
    
    for word in words:
        if word in competitor_mentions:
            competitor_mentions[word] += 1
    
    competitor_mentions = {k: v for k, v in competitor_mentions.items() if v > 0}
    
    print(f"🔍 Top adjectives: {top_adjectives}")
    print(f"✅ Competitor mentions: {competitor_mentions}")
    
    return top_adjectives, competitor_mentions

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/fetch_reviews', methods=['GET'])
def fetch_reviews():
    asin = request.args.get('asin')
    pages = int(request.args.get('pages', 5))
    sort_by = request.args.get('sort_by', 'recent')

    if not asin:
        return jsonify({"error": "ASIN is required."}), 400

    reviews, dates, product_name = get_reviews_oxylabs(asin, pages=pages, sort_by=sort_by)
    if not reviews:
        return jsonify({"error": "No reviews found for this product."}), 404

    # ✅ Format review dates correctly
    def parse_review_date(raw_date):
        try:
            return datetime.strptime(raw_date.replace("Reviewed in the United States", "").strip(), "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            return "Unknown Date"

    review_dates = [parse_review_date(date) for date in dates]
    print(f"📅 Formatted Dates: {review_dates}")  # Debugging

    # ✅ Extract adjectives and competitor mentions
    top_adjectives, competitor_mentions = extract_adjectives_and_competitors(reviews)
    
    print("🚀 Running Sentiment Analysis...")
    
    try:
        results = sentiment_analyzer(list(reviews), truncation=True, max_length=512, padding=True, batch_size=8)
        print(f"📊 Raw Sentiment Analysis Results: {results}")
    except Exception as e:
        return jsonify({"error": "Sentiment analysis failed."}), 500

    response = []
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}

    positive_scores, negative_scores, neutral_scores = [], [], []

    for r, res, d in zip(reviews, results, review_dates):  # ✅ Use parsed dates
        sentiment = LABEL_MAPPING.get(res["label"].upper(), "NEUTRAL")  # ✅ Fix case sensitivity
        score = res["score"] * 10

        sentiment_counts[sentiment] += 1

        # ✅ Fix issue where all arrays were defaulting to zero
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

        response.append({"review": r, "sentiment": sentiment, "score": score, "date": d})

    print(f"✅ Sentiment Counts: {sentiment_counts}")
    print(f"📈 Positive Scores: {positive_scores}")
    print(f"📉 Negative Scores: {negative_scores}")
    print(f"⚖️ Neutral Scores: {neutral_scores}")

    # Compute median score
    scores = [r["score"] for r in response]
    median_score = round(np.median(scores), 2) if scores else None

    # Compute sentiment percentages
    total_reviews = sum(sentiment_counts.values())
    positive_percentage = round((sentiment_counts["POSITIVE"] / total_reviews) * 100, 2) if total_reviews else 0
    negative_percentage = round((sentiment_counts["NEGATIVE"] / total_reviews) * 100, 2) if total_reviews else 0
    neutral_percentage = round((sentiment_counts["NEUTRAL"] / total_reviews) * 100, 2) if total_reviews else 0

    print(f"📊 Final Sentiment Counts: {sentiment_counts}")
    print(f"📈 Positive: {positive_percentage}% | 📉 Negative: {negative_percentage}% | ⚖️ Neutral: {neutral_percentage}%")

    return jsonify({
        "product_name": product_name,
        "total_reviews": len(reviews),
        "median_score": median_score,
        "top_adjectives": top_adjectives,
        "competitor_mentions": dict(competitor_mentions),
        "review_dates": review_dates,  # ✅ Fixed from `dates`
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "neutral_scores": neutral_scores,
        "positive_percentage": positive_percentage,
        "negative_percentage": negative_percentage,
        "neutral_percentage": neutral_percentage
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
