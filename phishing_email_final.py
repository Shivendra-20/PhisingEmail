
# Import libraries for text processing and model loading
import re
import joblib
from pathlib import Path

# Get the folder where this Python script is located
BASE_DIR = Path(__file__).resolve().parent

# Load the saved machine learning model
model = joblib.load(BASE_DIR / "models" / "phishing_model.pkl")

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")


# Clean the input email using the same basic preprocessing approach
def clean_text(text):
    # Convert text to lowercase and replace non-word characters with spaces
    text = re.sub(r"\W", " ", str(text)).lower()

    # Return the cleaned email
    return text


# Predict whether an email is legitimate or phishing
def predict_email(email):
    # Clean the email text
    cleaned_email = clean_text(email)

    # Convert the email into TF-IDF features
    email_features = vectorizer.transform([cleaned_email])

    # Predict the email label
    prediction = model.predict(email_features)[0]

    # Return the predicted category
    if prediction == 1:
        return "Phishing Email"
    else:
        return "Legitimate Email"


# Run the program when this file is executed directly
if __name__ == "__main__":
    # Ask the user to enter an email
    email = input("Paste the email text: ")

    # Predict and display the result
    result = predict_email(email)
    print("\nPrediction:", result)
