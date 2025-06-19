# Sentiment Analysis Web App

This is a machine learning-based web application that performs **sentiment analysis** on product reviews. It uses **Flask** for the backend and a trained **Naive Bayes** model with **TF-IDF vectorization** for classifying review sentiments.

---

## 🚀 Features

- ✅ Classifies review text as **Positive**, **Negative**, or **Neutral**
- ✅ Simple web interface built with HTML and Flask
- ✅ Uses **joblib** to save and load the trained model
- ✅ Fully working local deployment

---

## 📁 Project Structure

sentiment-analysis-project/
│
├── app.py # Flask app
├── sentiment_model.py # Model training script
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── templates/
│ └── index.html # Web UI
└── README.md # This file


---

## 🛠️ How to Run the Project

### 1. 🔧 Install required packages

Make sure Python is installed. Then install these:

```bash
pip install flask scikit-learn joblib

--------
 Train the model (optional)
If model.pkl and vectorizer.pkl are missing, run


Run the web app:

python app.py

Open your browser and go to:
http://127.0.0.1:5000

 Technologies Used
Python

Flask

scikit-learn

HTML/CSS

joblib

--------
👩‍💻 Author
Sarah Christ
🌐 GitHub: Sarah-Christ



This project is for educational/demo purposes. Feel free to use or modify!
