from phishing_detector import PhishingDetector
import argparse

def main():
    parser = argparse.ArgumentParser(description="Phishing Detection System")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='URL or text to analyze')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance')
    
    args = parser.parse_args()
    
    detector = PhishingDetector()
    
    if args.train:
        print("Training model...")
        detector.train()
        print("Model training complete!")
        
    elif args.predict:
        result = detector.predict(args.predict)
        print(f"Prediction: {'Phishing' if result else 'Legitimate'}")
        
    elif args.evaluate:
        print("Evaluating model...")
        accuracy, report = detector.evaluate()
        print(f"Model Accuracy: {accuracy:.2f}%")
        print("Classification Report:")
        print(report)

if __name__ == "__main__":
    main()
