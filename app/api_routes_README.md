# 📚 GetToKnow Project - `api_routes.py` Quick Summary

This file defines the **backend API endpoints** for interacting with the Sentiment Analysis system.

---

## 🚀 Main API Endpoints

### 1. **`/fetch_reviews`** (GET)
- **Purpose**: Fetches product reviews by ASIN, runs sentiment analysis, extracts top adjectives, competitor mentions, and saves a snapshot.
- **Input**:
  - Query Parameters:
    - `asin` (required): The Amazon ASIN to fetch reviews for.
    - `count` (optional): Number of reviews the user wants to scrape (default: 50 if missing).
- **Process**:
  - Fetches product metadata (title, manufacturer, price).
  - Traverses pages of reviews dynamically until enough reviews are collected.
  - Analyzes:
    - Sentiment (positive/negative/neutral)
    - Adjectives
    - Competitor mentions (NER + GPT)
  - Creates a new SentimentSnapshot and saves it into the database.
- **Returns**:
  - A fully serialized JSON object with:
    - Product details
    - Sentiment scores
    - Top adjectives
    - Competitor mentions
    - Top helpful reviews
    - Country sentiment distribution


---

## 🛠 Internals Used
- `compute_review_hashes_and_filter()`: Parse and deduplicate review texts.
- `get_sentiment_pipeline()`: RoBERTa sentiment model.
- `extract_adjectives_and_competitors()`: SpaCy-based adjective and competitor extractor.
- `fetch_competitor_names()`: GPT-3.5-based competitor fetch if needed.
- `snapshot_to_dict()`: Safely serialize SentimentSnapshot to clean JSON.


---

## 📦 Additional Notes
- If a cached snapshot already exists, it can be reused to avoid redundant API scraping.
- GPT fallback is used only if no competitor cache exists.
- Logs and error handling are robust to avoid frontend crashes.
- Designed to **support custom review counts** per user request.


---

