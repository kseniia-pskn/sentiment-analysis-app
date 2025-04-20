import nltk
import spacy
import os
import json
from openai import OpenAI
from collections import Counter

# Optional: Load .env if running locally (uncomment if needed)
# from dotenv import load_dotenv
# load_dotenv()

# Setup custom NLTK data path
nltk_data_path = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
nltk.data.path.append(nltk_data_path)

# Ensure NLTK models are available
required_resources = [
    ("punkt", "tokenizers"),
    ("stopwords", "corpora"),
    ("averaged_perceptron_tagger", "taggers")
]

for resource, category in required_resources:
    try:
        nltk.data.find(f"{category}/{resource}")
        print(f"✅ NLTK resource '{resource}' is available.")
    except LookupError:
        try:
            print(f"⬇️ Downloading NLTK resource '{resource}'...")
            nltk.download(resource, download_dir=nltk_data_path)
            print(f"✅ Successfully downloaded '{resource}'.")
        except Exception as e:
            print(f"[WARNING] Failed to download NLTK resource '{resource}': {e}")

# Load SpaCy English model
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

# Extract adjectives and competitor mentions from reviews using NLTK and SpaCy
def extract_adjectives_and_competitors(reviews):
    print("🔍 Starting extraction of adjectives and competitor mentions...")
    text = " ".join(reviews).lower()

    try:
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
        print(f"🔠 Tokenized {len(words)} words.")
    except Exception as e:
        print(f"[ERROR] NLTK tokenization or tagging failed: {e}")
        return [], {}

    try:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        print("[WARNING] Stopwords resource not found. Proceeding without filtering stopwords.")
        stopwords = set()

    adjectives = [word for word, tag in tagged_words if tag in ["JJ", "JJR", "JJS"] and word.isalpha()]
    filtered_adjectives = [adj for adj in adjectives if adj not in stopwords]
    top_adjectives = Counter(filtered_adjectives).most_common(10)
    print(f"📝 Found {len(top_adjectives)} top adjectives.")

    competitor_mentions = Counter()
    try:
        for doc in nlp.pipe(reviews, disable=["parser"]):
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    competitor_mentions[ent.text.lower()] += 1
        print(f"🏷️ Extracted {len(competitor_mentions)} potential competitor brand mentions.")
    except Exception as e:
        print(f"[ERROR] SpaCy NER processing failed: {e}")
        return top_adjectives, {}

    filtered_mentions = {k: v for k, v in competitor_mentions.items() if v > 0}
    print(f"✅ Filtered to {len(filtered_mentions)} valid competitor mentions.")
    return top_adjectives, filtered_mentions

# Fetch competitor brand names using OpenAI GPT
def fetch_competitor_names(product_name, manufacturer):
    print(f"🤖 Calling GPT to fetch competitors for: {product_name} by {manufacturer}...")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("[ERROR] OPENAI_API_KEY is missing from environment.")
        return []

    client = OpenAI(api_key=openai_key)

    system_prompt = (
        "You are a smart assistant integrated into a sentiment analysis tool for Amazon products. "
        "Your task is to generate a list of competitor product names and common abbreviations of competing brands or models "
        "based on the following input:\n\n"
        f"- Product Name: {product_name}\n"
        f"- Manufacturer: {manufacturer}\n\n"
        "Use your general knowledge and logical inference to identify:\n"
        "- Brands and manufacturers that make similar products.\n"
        "- Popular or competing product names (e.g., suggest \"Cortez\" if analyzing \"Samba\").\n"
        "- Common abbreviations and shorthand used by users to refer to those brands or products.\n\n"
        "Respond strictly with a JSON array of strings. No explanations. No extra output.\n\n"
        "Example:\n[\"Nike\", \"Cortez\", \"NB\", \"New Balance\", \"Asics\", \"Puma\"]"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Product Name: {product_name}\nManufacturer: {manufacturer}"}
            ],
            temperature=0.4,
            max_tokens=512
        )
        result = response.choices[0].message.content.strip()
        print("✅ GPT response received.")
        return json.loads(result)
    except Exception as e:
        print(f"[ERROR] GPT API call failed: {e}")
        return []
