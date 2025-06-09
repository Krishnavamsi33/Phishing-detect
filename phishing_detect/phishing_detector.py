import os
import re
import joblib
import logging

from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PhishingDetector:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model_path = 'models/phishing_model.pkl'
        self.vectorizer_path = 'models/tfidf_vectorizer.pkl'
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load model if it exists
        logging.info("Loading model and vectorizer if they exist.")

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

    def preprocess_text(self, text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

    def extract_features(self, text):
        # Preprocess text
        processed_text = self.preprocess_text(text)
        # Vectorize text
        if self.vectorizer:
            return self.vectorizer.transform([processed_text])
        return None

    def tune_hyperparameters(self, X_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def train(self):

        # Load dataset (you should replace this with your actual dataset)
        # Dataset should have 'text' and 'label' columns (1 for phishing, 0 for legitimate)
        data = pd.read_csv('data/phishing_data.csv')
        
        # Preprocess text
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_text'], data['label'], test_size=0.2, random_state=42
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        logging.info("Tuning hyperparameters.")

        self.model = self.tune_hyperparameters(X_train_vec, y_train)

        
        logging.info("Saving model and vectorizer.")

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        logging.info("Evaluating model.")

        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def predict(self, text):
        if not self.model:
            raise Exception("Model not trained. Please train the model first.")
        
        features = self.extract_features(text)
        if features is not None:
            prediction = self.model.predict(features)
            return bool(prediction[0])
        return False

    def evaluate(self):
        if not self.model:
            raise Exception("Model not trained. Please train the model first.")
        
        # Load test data
        data = pd.read_csv('data/phishing_data.csv')
        data['processed_text'] = data['text'].apply(self.preprocess_text)
        X_test = self.vectorizer.transform(data['processed_text'])
        y_test = data['label']
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report
