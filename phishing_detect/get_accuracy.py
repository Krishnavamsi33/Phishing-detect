from phishing_detector import PhishingDetector

def main():
    detector = PhishingDetector()
    accuracy, report = detector.evaluate()
    print(f"Model Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    main()
