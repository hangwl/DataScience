# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data into a pandas DataFrame
data = pd.read_csv('data.csv')

# Split the data into training and testing sets
train_data = data[:800]
test_data = data[800:]

# Extract the input (X) and output (y) variables
X_train = train_data[['feature_1', 'feature_2', ...]]
y_train = train_data['target_variable']
X_test = test_data[['feature_1', 'feature_2', ...]]
y_test = test_data['target_variable']

from sklearn.ensemble import RandomForestClassifier

# Instantiate the model
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
