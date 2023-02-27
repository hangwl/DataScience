from sklearn import svm, metrics

# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = ...

# Create SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear') # available kernels include: linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Compute evaluation metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
print("Confusion matrix:\n", confusion_matrix)