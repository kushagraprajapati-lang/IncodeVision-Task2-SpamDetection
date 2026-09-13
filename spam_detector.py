import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def build_model():
    # 1. Training Dataset (Ham vs Spam)
    training_data = [
        # Legitimate messages (Ham)
        ("Hey, are we still meeting for lunch today?", "ham"),
        ("Can you please send me the report by end of day?", "ham"),
        ("Hi Kushagra, let's discuss the project tomorrow.", "ham"),
        ("Your package has been delivered to your doorstep.", "ham"),
        ("Don't forget to submit Task 01 on GitHub.", "ham"),
        ("Are you free for a quick call right now?", "ham"),
        
        # Spam messages
        ("CONGRATULATIONS! You won a $1000 gift card! Claim now", "spam"),
        ("URGENT: Your account has been compromised. Reply immediately.", "spam"),
        ("Free money! Click here to claim your cash bonus today!", "spam"),
        ("Get cheap loans with 0% interest rate instant approval now!", "spam"),
        ("WINNER! You have been selected for a free iPhone! Call now", "spam")
    ]

    # Split into features (X) and target labels (y)
    X_train, y_train = zip(*training_data)

    # 2. ML Pipeline: Text Vectorizer (Bag of Words) + Naive Bayes Classifier
    model = make_pipeline(
        CountVectorizer(stop_words='english', lowercase=True),
        MultinomialNB()
    )

    # 3. Train the model
    model.fit(X_train, y_train)
    return model

def main():
    print("=" * 60)
    print("      INCODEVISION TASK 02: SPAM MESSAGE DETECTOR      ")
    print("=" * 60)
    print("Training Machine Learning Model (Multinomial Naive Bayes)...")
    
    model = build_model()
    print("Model trained successfully!\n")
    print("Type any text message to test. Type 'exit' to quit.\n")

    while True:
        try:
            user_message = input("Enter Message: ").strip()

            if user_message.lower() in ["exit", "quit"]:
                print("Exiting Spam Detector. Task 02 Complete!")
                break

            if not user_message:
                continue

            prediction = model.predict([user_message])[0]
            confidence = max(model.predict_proba([user_message])[0]) * 100
            
            label = "🚨 SPAM" if prediction == "spam" else "✅ HAM (NOT SPAM)"
            print(f"Prediction : {label}")
            print(f"Confidence : {confidence:.2f}%\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit()

if __name__ == "__main__":
    main()
    