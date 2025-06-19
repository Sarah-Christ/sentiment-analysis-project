import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Expanded and balanced dataset
data = {
    "review": [
        # Positive
        "I love this product! It’s amazing.",
        "Excellent quality, highly recommended!",
        "Very satisfied, will buy again.",
        "Great value for money.",
        "Absolutely fantastic item!",

        # Negative
        "Worst experience ever. Totally disappointed.",
        "Terrible product, broke in a week.",
        "Not worth the money at all.",
        "Very bad quality, regret buying.",
        "This is a waste of money.",

        # Neutral
        "It's okay, not great but not bad.",
        "The product is average, nothing special.",
        "Quality is decent for the price.",
        "It's just fine. Does the job.",
        "Not much to say, it's acceptable."
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "neutral", "neutral", "neutral", "neutral", "neutral"
    ]
}

df = pd.DataFrame(data)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

df['cleaned_review'] = df['review'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

