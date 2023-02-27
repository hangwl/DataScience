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

# Perform data preprocessing and feature scaling if necessary

# Create a decision tree model object
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance on the testing data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Visualize the decision tree if necessary
from sklearn.tree import plot_tree
plot_tree(model)

# Alternatively, you can export the tree to a file
from sklearn.tree import export_graphviz
export_graphviz(model, out_file='tree.dot', feature_names=['feature_1', 'feature_2', ...], class_names=['class_0', 'class_1'], filled=True)

# Convert the dot file to an image file (requires Graphviz <https://pypi.org/project/graphviz/>)
!dot -Tpng tree.dot -o tree.png
