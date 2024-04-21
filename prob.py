
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import string  # Add this import statement
# Load data
x_train_df = pd.read_csv("X_train.csv")
y_train_df = pd.read_csv("y_train.csv")
x_test_df = pd.read_csv("X_test.csv")
y_test_df = pd.read_csv("y_test.csv")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply preprocessing to 'Facts' column
x_train_df['Facts'] = x_train_df['Facts'].apply(preprocess_text)
x_test_df['Facts'] = x_test_df['Facts'].apply(preprocess_text)

# Define and train the model
model = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression()
)

model.fit(x_train_df['Facts'], y_train_df['winner_index'])

# Predict on test set
predictions = model.predict(x_test_df['Facts'])

# Evaluate the model
accuracy = accuracy_score(y_test_df['winner_index'], predictions)
print(f"Accuracy: {accuracy}")
def predict_winning_probability(petitioner, respondent, facts):
    # Preprocess the input text
    processed_facts = preprocess_text(facts)
    input_text = f"{petitioner} {respondent} {processed_facts}"
    # Use the trained model to predict probabilities
    probabilities = model.predict_proba([input_text])[0]
    return probabilities

# # Example inputs
# petitioner = "John"
# respondent = "Doe"
# facts = "This is a case about contract disputes.john seized does land and abused doe.neighbours saw him push doe and threaten him.john has criminal track record john has corruption and many land dispute cses against him."

# # Predict and display result
# winning_probabilities = predict_winning_probability(petitioner, respondent, facts)
# print(f"Winning Probability for Petitioner: {winning_probabilities[1]}")
# print(f"Winning Probability for Respondent: {winning_probabilities[0]}")
