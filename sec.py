# import json
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # Function to load dataset
# def load_dataset(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             try:
#                 record = json.loads(line)
#                 data.append((record['offense'], record['IPC_section'], record['punishment']))
#             except json.JSONDecodeError as e:
#                 print("Error parsing JSON:", e)
#     return data

# # Load dataset
# dataset = load_dataset('ipc.jsonl')

# # Split data into train and test sets
# X = [record[0] for record in dataset]  # Offenses
# y_section = [record[1] for record in dataset]  # Sections
# y_punishment = [record[2] for record in dataset]  # Punishments
# X_train, X_test, y_section_train, y_section_test, y_punishment_train, y_punishment_test = train_test_split(
#     X, y_section, y_punishment, test_size=0.2, random_state=42)

# # Create TfidfVectorizer
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
# X_train_vectorized = vectorizer.fit_transform(X_train)

# # Train Random Forest classifier for section prediction
# rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
# rf_classifier.fit(X_train_vectorized, y_section_train)

# # Train RandomForestClassifiers for punishment prediction
# rf_punishment_estimators = []
# for section in np.unique(y_section_train):
#     # Filter data for the current section
#     X_train_section = [X_train[i] for i in range(len(X_train)) if y_section_train[i] == section]
#     y_punishment_train_section = [y_punishment_train[i] for i in range(len(X_train)) if y_section_train[i] == section]

#     # Vectorize data
#     X_train_section_vectorized = vectorizer.transform(X_train_section)

#     # Train RandomForestClassifier
#     rf_punishment_classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
#     rf_punishment_classifier.fit(X_train_section_vectorized, y_punishment_train_section)

#     rf_punishment_estimators.append(rf_punishment_classifier)

# # Predict section
# def predict_section_and_punishment(offense):
#     offense_vectorized = vectorizer.transform([offense])
#     section_prediction = rf_classifier.predict(offense_vectorized)[0]
#     section_index = np.where(rf_classifier.classes_ == section_prediction)[0][0]
#     punishment_prediction = rf_punishment_estimators[section_index].predict(offense_vectorized)[0]
#     return section_prediction, punishment_prediction

# # Example usage
# offense = "Theft of property"
# section, punishment = predict_section_and_punishment(offense)
# print("Predicted Section:", section)
# print("Predicted Punishment:", punishment)
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lists to store valid cases
valid_cases = []

# Read dataset and handle invalid lines
with open('ipc.jsonl', 'r', encoding='utf-8') as file:
    for line_num, line in enumerate(file, start=1):
        try:
            case = json.loads(line.strip())
            valid_cases.append(case)
        except json.JSONDecodeError as e:
            print(f"Ignore line {line_num}: {e}")

# Extract features and labels from valid cases
X = [case['offense'] for case in valid_cases]
y = [case['IPC_section'] for case in valid_cases]
punishments = {case['IPC_section']: case['punishment'] for case in valid_cases}

# Tokenization, stop words removal, and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [token for token in tokens if token.isalpha()]  # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return ' '.join(tokens)


X_processed = [preprocess(text) for text in X]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_processed)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vec, y)

# Example testing with a lengthy case fact
example_case_fact = """Complainant's Account:
The complainant, Mr. Ramesh Sharma, reports that his 6-year-old daughter, Sunita Sharma, was abducted on the evening of April 30th, 2024, while returning home from school. According to Mr. Sharma, Sunita was walking home from her school bus stop when a white van pulled up beside her. Two masked individuals forcefully grabbed Sunita and dragged her into the van before speeding away. Mr. Sharma immediately reported the incident to the local police station and has been actively cooperating with the investigation ever since. He claims to have no knowledge of the abductors' identities or motives.

Defendant's Account:
The defendant, Mr. Rajesh Singh, denies any involvement in the alleged kidnapping of Sunita Sharma. According to Mr. Singh, he was at his workplace, a construction site, during the time of the incident. He states that he has never met Sunita Sharma or her family and has no reason to harm or abduct her. Mr. Singh asserts that he is being falsely accused and demands a thorough investigation to clear his name.

Witness Testimony:
An eyewitness, Mrs. Geeta Verma, corroborates Mr. Sharma's account of the abduction. Mrs. Verma, who lives near the school bus stop, claims to have seen the white van and the masked individuals forcibly taking Sunita Sharma. She immediately informed the authorities and has provided a detailed description of the van and the suspects.

Investigation Findings:
The investigation has revealed CCTV footage from nearby cameras, which captures the white van and the masked individuals at the scene of the abduction. The police are currently analyzing the footage to identify the suspects and gather further evidence.

Legal Proceedings:
Mr. Ramesh Sharma has filed a formal complaint with the police, accusing Mr. Rajesh Singh of kidnapping his daughter, Sunita Sharma. The police have initiated an investigation into the matter and are working diligently to locate and rescue Sunita. Mr. Singh maintains his innocence and awaits a fair trial to prove his innocence.
"""


def predict_section_and_punishment(case_fact):
    example_case_processed = preprocess(case_fact)
    example_case_vec = vectorizer.transform([example_case_processed])
    # Predict IPC sections with probabilities
    predicted_probs = model.predict_proba(example_case_vec)[0]  # Get probabilities for the first example
    # Get the indices of the top k probabilities
    top_k_indices = predicted_probs.argsort()[-5:][::-1]
    # Retrieve IPC sections and corresponding probabilities
    top_k_sections = model.classes_[top_k_indices]
    top_k_probabilities = predicted_probs[top_k_indices]
    # Print top k IPC sections with probabilities and respective potential punishments
    for section, probability in zip(top_k_sections, top_k_probabilities):
        print(f"IPC Section: {section}, Probability: {probability:.4f}, Potential Punishment: {punishments[section]}")
    return top_k_sections, top_k_probabilities
