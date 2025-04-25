import os
import json
import threading
from collections import Counter
from openai import OpenAI
from transformers import pipeline
import spacy

_sentiment_pipeline = None
_nlp_model = None

# ---------------------
# NLP + Pipeline Lazy Loaders
# ---------------------
def get_nlp():
    global _nlp_model
    if _nlp_model is None:
        print("📦 Loading SpaCy model...")
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            print("✅ SpaCy model loaded.")
        except OSError:
            print("⬇️ Downloading SpaCy model...")
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            _nlp_model = spacy.load("en_core_web_sm")
            print("✅ SpaCy model downloaded and loaded.")
    return _nlp_model

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        print("📦 Loading sentiment analysis model...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        print("✅ Sentiment pipeline loaded.")
    return _sentiment_pipeline

# ---------------------
# Core Extractors
# ---------------------
def extract_adjectives_and_competitors(reviews, nlp=None):
    print("🔍 Extracting adjectives and competitor mentions...")
    nlp = nlp or get_nlp()

    adjectives = Counter()
    competitor_mentions = Counter()

    try:
        for doc in nlp.pipe(reviews, disable=["parser"]):
            adjectives.update(
                token.text.lower() for token in doc if token.pos_ == "ADJ" and token.is_alpha
            )
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    competitor_mentions[ent.text.strip().lower()] += 1
    except Exception as e:
        print(f"[ERROR] SpaCy processing failed: {e}")
        return [], {}

    top_adjectives = adjectives.most_common(10)
    top_competitors = {k: v for k, v in competitor_mentions.items() if v > 0}
    return top_adjectives, top_competitors

def fetch_competitor_names(product_name, manufacturer):
    print(f"🤖 GPT call for competitors: {product_name} by {manufacturer}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] OpenAI API key missing")
        return []

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = (
            "You are a product analysis assistant. Given a product name and manufacturer, "
            "return a JSON list of similar or competing brands/products.\n\n"
            f"Product: {product_name}\nManufacturer: {manufacturer}\n\n"
            "Example: [\"L'Oréal\", \"Vaseline\", \"CeraVe\"]"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{product_name} by {manufacturer}"}
            ],
            temperature=0.3,
            max_tokens=200
        )

        raw_output = response.choices[0].message.content.strip()
        competitors = json.loads(raw_output)
        if not isinstance(competitors, list):
            print("[WARNING] GPT response not a list. Raw Output:", raw_output)
            return []

        return [comp.strip() for comp in competitors if isinstance(comp, str)]

    except Exception as e:
        print(f"[ERROR] GPT competitor fetch failed: {e}")
        return []

# ---------------------
# Self-Diagnostics on Startup
# ---------------------
def run_startup_diagnostics():
    print("🛠️ Running startup diagnostics...")

    try:
        nlp = get_nlp()
        doc = nlp("The camera quality is amazing, but battery life is disappointing.")
        print(f"✅ SpaCy works: found {len(doc)} tokens.")
    except Exception as e:
        print(f"[ERROR] SpaCy failed to load: {e}")

    try:
        pipeline = get_sentiment_pipeline()
        result = pipeline("I love this product!")
        print(f"✅ Sentiment pipeline works: {result}")
    except Exception as e:
        print(f"[ERROR] Sentiment pipeline failed: {e}")

    try:
        competitors = fetch_competitor_names("Nivea Cream", "Nivea")
        print(f"✅ OpenAI GPT competitors test passed: {competitors}")
    except Exception as e:
        print(f"[ERROR] GPT fetch failed: {e}")

def run_diagnostics_on_startup():
    threading.Thread(target=run_startup_diagnostics).start()

# Trigger diagnostics
run_diagnostics_on_startup()
