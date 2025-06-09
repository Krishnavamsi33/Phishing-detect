from flask import Flask, render_template, request, jsonify
from phishing_detector import PhishingDetector
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
detector = PhishingDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url', '')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Get the text content from the URL
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        
        # 
        
        is_phishing = detector.predict(text)
        
        return jsonify({
            'prediction': 'Phishing' if is_phishing else 'Legitimate',
            'is_phishing': is_phishing
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

