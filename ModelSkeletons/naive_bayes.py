from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.model_selection import train_test_split

# Load data and split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GaussianNB() # or Bernoulli, Multinomial depending on nature of target variable
model.fit(X_train, y_train)

# Test the model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, roc_auc_score

y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Compute precision, recall, and F1 score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Compute ROC curve and AUC score
probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
print("AUC score:", auc)

