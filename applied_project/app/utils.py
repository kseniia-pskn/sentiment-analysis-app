import nltk
import spacy
import re
from collections import Counter

# Download required NLTK models (called only once in main)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy
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

    competitors = ["nivea", "neutrogena", "eucerin", "cetaphil", "cerave", "aveeno", "olay", "lubriderm", "dove", "gold bond"]
    competitor_mentions = {brand: 0 for brand in competitors}

    for word in words:
        if word in competitor_mentions:
            competitor_mentions[word] += 1

    competitor_mentions = {k: v for k, v in competitor_mentions.items() if v > 0}

    return top_adjectives, competitor_mentions
