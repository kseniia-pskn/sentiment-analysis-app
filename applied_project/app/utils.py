import os
import json
import spacy
from collections import Counter
from openai import OpenAI

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded.")
except OSError:
    print("⬇️ Downloading SpaCy model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model downloaded and loaded.")

def extract_adjectives_and_competitors(reviews):
    print("🔍 Extracting adjectives and competitor mentions...")

    adjectives = Counter()
    competitor_mentions = Counter()

    try:
        for doc in nlp.pipe(reviews, disable=["parser"]):
            adjectives.update([token.text.lower() for token in doc if token.pos_ == "ADJ" and token.is_alpha])
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

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a product analysis assistant. Given a product name and manufacturer, return a JSON list of similar or competing brands/products."
        f"\n\nProduct: {product_name}\nManufacturer: {manufacturer}\n\nExample: [\"L'Oréal\", \"Vaseline\", \"CeraVe\"]"
    )

    try:
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
        if isinstance(competitors, list):
            return [comp.strip() for comp in competitors if isinstance(comp, str)]
        return []
    except Exception as e:
        print(f"[ERROR] GPT competitor fetch failed: {e}")
        return []
