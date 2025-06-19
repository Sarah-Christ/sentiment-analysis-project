from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        data = vectorizer.transform([review])
        prediction = model.predict(data)[0]
        confidence = model.predict_proba(data).max() * 100
        return render_template('index.html', prediction=prediction.capitalize(), confidence=round(confidence, 2))

if __name__ == '__main__':
    app.run(debug=True)
