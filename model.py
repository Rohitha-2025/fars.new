import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the preprocessed data
df = pd.read_csv('/Users/shivasaimuthyam/Downloads/processed_landmarks.csv')  # Use your actual path if needed

# Step 2: Separate features and labels
X = df.drop(columns=['class'])
y = df['class']

# Step 3: Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Predict on test data
y_pred = clf.predict(X_test)

# Step 6: Evaluate model
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the trained model
joblib.dump(clf, 'exercise_classifier_rf.joblib')
print("✅ Model trained and saved as 'exercise_classifier_rf.joblib'")
