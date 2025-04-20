import nltk
import spacy
import os
from collections import Counter

# Setup custom NLTK data path (for deployment environments like Render)
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

# Extract adjectives and competitor mentions
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
