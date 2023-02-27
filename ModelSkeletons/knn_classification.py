from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data into a pandas DataFrame
data = pd.read_csv('data.csv')

# Separate the target variable and the input features
X = data.drop('target_variable', axis=1)
y = data['target_variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (e.g., normalization, feature scaling)

# Define the range of k values to test
k_range = range(1, 31)

# Define empty lists to store evaluation metrics for each k
cv_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform k-fold cross-validation on the training set to determine the optimal value of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
    # Fit the K-NN model on the training set and calculate evaluation metrics on the validation set
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    precision_scores.append(precision_score(y_train, y_pred))
    recall_scores.append(recall_score(y_train, y_pred))
    f1_scores.append(f1_score(y_train, y_pred))

# Find the optimal value of k
optimal_k = k_range[cv_scores.index(max(cv_scores))]

# Train the K-NN model on the entire training set using the optimal value of k
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Evaluate the performance of the model on the test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics and the optimal value of k
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("Confusion matrix:\n", confusion_mat)
print("Optimal k:", optimal_k)
