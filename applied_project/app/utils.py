import nltk
import spacy
import re
import os
from collections import Counter

# Setup custom NLTK data path (for deployment environments like Render)
nltk_data_path = os.getenv("NLTK_DATA", "/opt/render/nltk_data")
nltk.data.path.append(nltk_data_path)

# Ensure necessary NLTK models are available
for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

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
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)

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
