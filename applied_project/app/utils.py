import nltk
import spacy
import os
import json
from collections import Counter
from openai import OpenAI

# Set up NLTK data path
nltk_data_path = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
nltk.data.path.append(nltk_data_path)

# Define required NLTK resources and their categories
nltk_resources = [
    ("punkt", "tokenizers"),
    ("stopwords", "corpora"),
    ("averaged_perceptron_tagger", "taggers")
]

# Ensure required NLTK resources
for resource, category in nltk_resources:
    try:
        nltk.data.find(f"{category}/{resource}")
        print(f"✅ NLTK resource '{resource}' available.")
    except LookupError:
        print(f"⬇️ Downloading NLTK resource '{resource}' to {nltk_data_path}...")
        try:
            nltk.download(resource, download_dir=nltk_data_path)
            print(f"✅ Successfully downloaded '{resource}'.")
        except Exception as e:
            print(f"[ERROR] Failed to download '{resource}': {e}")

# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded.")
except OSError:
    try:
        print("⬇️ Downloading SpaCy model...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("✅ SpaCy model downloaded and loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load SpaCy model: {e}")
        raise

def extract_adjectives_and_competitors(reviews):
    print("🔍 Starting extraction of adjectives and competitor mentions...")
    text = " ".join(reviews).lower()

    try:
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
        print(f"🔠 Tokenized {len(words)} words.")
    except Exception as e:
        print(f"[ERROR] NLTK tokenization failed: {e}")
        return [], {}

    try:
        stopwords = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        print("[WARNING] Missing stopwords, continuing without.")
        stopwords = set()

    adjectives = [word for word, tag in tagged_words if tag.startswith("JJ") and word.isalpha()]
    filtered_adjectives = [adj for adj in adjectives if adj not in stopwords]
    top_adjectives = Counter(filtered_adjectives).most_common(10)

    competitor_mentions = Counter()
    try:
        for doc in nlp.pipe(reviews, disable=["parser"]):
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    competitor_mentions[ent.text.lower()] += 1
    except Exception as e:
        print(f"[ERROR] SpaCy NER failed: {e}")
        return top_adjectives, {}

    filtered_mentions = {k: v for k, v in competitor_mentions.items() if v > 0}
    return top_adjectives, filtered_mentions

def fetch_competitor_names(product_name, manufacturer):
    print(f"🤖 Calling GPT for competitors: {product_name} by {manufacturer}...")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[ERROR] No OpenAI key")
        return []

    client = OpenAI(api_key=openai_key)
    system_prompt = (
        "You are an intelligent assistant. Based on the product name and manufacturer, "
        "return a JSON array of competing brands and products."
        f"\n\nProduct Name: {product_name}\nManufacturer: {manufacturer}\n\nExample: [\"L'Oréal\", \"Vaseline\", \"CeraVe\"]"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Product Name: {product_name}\nManufacturer: {manufacturer}"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        return json.loads(output)
    except Exception as e:
        print(f"[ERROR] GPT call failed: {e}")
        return []
