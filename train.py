import pandas as pd
import re
import pickle
import nltk

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── Load dataset ──────────────────────────────────────────────
data = pd.read_csv("twitter.csv")
data.columns = data.columns.str.strip()

print("Columns:", data.columns.tolist())
print("Label distribution:\n", data["label"].value_counts())

TEXT_COLUMN = "tweet"
LABEL_COLUMN = "label"

# Original labels: 0 = Negative, 1 = Positive  (no reversal needed)
data = data[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

# ── Text cleaning ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
# Keep negation words — they're important for sentiment
negation_words = {"no", "not", "nor", "never", "neither", "nobody",
                  "nothing", "nowhere", "doesn't", "don't", "didn't",
                  "isn't", "aren't", "wasn't", "weren't", "can't",
                  "cannot", "couldn't", "won't", "wouldn't", "shouldn't"}
stop_words -= negation_words

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "", text)                     # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)                # strip hashtag #
    text = re.sub(r"[^a-zA-Z\s']", " ", text)           # keep letters + apostrophe
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

data["clean_text"] = data[TEXT_COLUMN].apply(clean_text)

# ── Features & labels ─────────────────────────────────────────
X = data["clean_text"]
y = data[LABEL_COLUMN]   # 0 = Negative, 1 = Positive

# ── Vectorize with TF-IDF (much better than CountVectorizer) ──
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),      # unigrams + bigrams
    sublinear_tf=True,       # log normalization
    min_df=2
)
X_vec = vectorizer.fit_transform(X)

# ── Train / Test split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train Logistic Regression (far better than Naive Bayes) ───
model = LogisticRegression(
    max_iter=1000,
    C=1.5,
    solver="lbfgs",
    class_weight="balanced",   # handles class imbalance
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ── Save model & vectorizer ───────────────────────────────────
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\nModel and vectorizer saved successfully!")
print("   Label mapping: 0 = Positive  |  1 = Negative (hate speech)")