# 📚 GetToKnow Project - `utils.py` Quick Summary

This file defines helper functions and models to support NLP and competitor analysis. It is loaded automatically when Flask app starts.

---

## 🚀 Main Components

### 1. **`get_nlp()`**
- Loads **SpaCy `en_core_web_sm`** model for Natural Language Processing.
- Downloads the model automatically if missing.
- Used for extracting **adjectives** and **organization names** (competitors) from reviews.

### 2. **`get_sentiment_pipeline()`**
- Loads **HuggingFace RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) model.
- Used for **sentiment analysis** (positive/negative/neutral) of review texts.

### 3. **`extract_adjectives_and_competitors(reviews, nlp=None)`**
- Takes a list of reviews.
- Extracts the top **10 adjectives**.
- Detects **organization names** (brands/competitors).
- Returns both as counters.

### 4. **`fetch_competitor_names(product_name, manufacturer)`**
- Calls **OpenAI GPT-3.5** to find similar or competing brands.
- Requires an environment variable `OPENAI_API_KEY`.
- Safely handles failures.

### 5. **`compute_review_hashes_and_filter(all_reviews, existing_snapshot)`**
- Parses review timestamps to dates.
- Deduplicates reviews using SHA-256 hashes.
- Extracts country and review meta information.
- Used to prepare reviews for analysis.

### 6. **Startup Diagnostics (`run_diagnostics_on_startup()`)**
- Runs when app boots.
- Verifies:
  - SpaCy model loaded ✅
  - Sentiment model loaded ✅
  - GPT competitor call works ✅

---

## 🛠 Notes
- SpaCy and Sentiment pipelines are **lazily loaded** (only once per app lifetime).
- All exceptions are **caught and logged** but don't crash the app.
- No model loads during import — only when first needed (performance boost).

---


