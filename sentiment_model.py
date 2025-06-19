import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load sample data
data = {
    'review': [
        'I love this product!',
        'This is the worst thing I ever bought.',
        'Not bad, okay for the price.',
        'Absolutely fantastic!',
        'Terrible experience.',
        'Good value for money.',
        'Worst purchase.',
        'I am very satisfied.',
        'Bad product.',
        'Highly recommend it!'
    ],
    'sentiment': [
        'positive',
        'negative',
        'neutral',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
        'negative',
        'positive'
    ]
}
df = pd.DataFrame(data)

# 2. Text preprocessing and vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. ✅ Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")
