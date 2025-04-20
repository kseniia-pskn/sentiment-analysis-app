import nltk
import spacy
import os
from collections import Counter

# Setup custom NLTK data path (for deployment environments like Render)
nltk_data_path = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
nltk.data.path.append(nltk_data_path)

# Ensure necessary NLTK models are available with robust fallback
for resource, path_type in [("punkt", "tokenizers"), ("stopwords", "corpora"), ("averaged_perceptron_tagger", "taggers")]:
    try:
        nltk.data.find(f"{path_type}/{resource}")
    except LookupError:
        try:
            nltk.download(resource, download_dir=nltk_data_path)
        except Exception as e:
            print(f"[WARNING] Failed to download {resource}: {e}")

# Load SpaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Extract adjectives and competitor mentions
def extract_adjectives_and_competitors(reviews):
    text = " ".join(reviews).lower()
    try:
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
    except Exception as e:
        print(f"[ERROR] Tokenization failed: {e}")
        return [], {}

    adjectives = [word for word, tag in tagged_words if tag in ["JJ", "JJR", "JJS"] and word.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    adjectives = [adj for adj in adjectives if adj not in stopwords]
    top_adjectives = Counter(adjectives).most_common(10)

    # Basic brand extraction from NER (fallback if manufacturer isn't enough)
    competitor_mentions = Counter()
    for doc in nlp.pipe(reviews, disable=["parser"]):
        for ent in doc.ents:
            if ent.label_ == "ORG":
                competitor_mentions[ent.text.lower()] += 1

    filtered_mentions = {k: v for k, v in competitor_mentions.items() if v > 0}

    return top_adjectives, filtered_mentions
