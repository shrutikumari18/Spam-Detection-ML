from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df.iloc[:, [0, 1]]
df.columns = ['label', 'message']

# Features & target
X = df['message']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Text vectorization
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

# Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# User input
user_input = input("\nEnter a message to classify:\n> ")
test_vec = cv.transform([user_input])
prediction = model.predict(test_vec)

print("\n--- Result ---")
print(f"Message: {user_input}")
print(f"Prediction: {prediction[0].upper()}")
