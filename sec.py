import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Function to load dataset
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                data.append((record['offense'], record['IPC_section'], record['punishment']))
            except json.JSONDecodeError as e:
                print("Error parsing JSON:", e)
    return data

# Load dataset
dataset = load_dataset('ipc.jsonl')

# Split data into train and test sets
X = [record[0] for record in dataset]  # Offenses
y_section = [record[1] for record in dataset]  # Sections
y_punishment = [record[2] for record in dataset]  # Punishments
X_train, X_test, y_section_train, y_section_test, y_punishment_train, y_punishment_test = train_test_split(
    X, y_section, y_punishment, test_size=0.2, random_state=42)

# Create TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_classifier.fit(X_train_vectorized, y_section_train)

# Transform test data
X_test_vectorized = vectorizer.transform(X_test)

# Predict section
section_predictions = rf_classifier.predict(X_test_vectorized)

# Predict punishment
def predict_section_and_punishment(offense):
    offense_vectorized = vectorizer.transform([offense])
    section_prediction = rf_classifier.predict(offense_vectorized)[0]
    section_index = np.where(rf_classifier.classes_ == section_prediction)[0][0]
    punishment_prediction = rf_punishment_estimators[section_index].predict(offense_vectorized)[0]
    return section_prediction, punishment_prediction

# # Test with a sample offense
# sample_offense = "Some offense text here..."
# predicted_section, predicted_punishment = predict_punishment(sample_offense)
# print("Predicted Section:", predicted_section)
# print("Predicted Punishment:", predicted_punishment)
