from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the data and split it into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the cross-validation iterator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the performance metrics for each fold
scores = []

# Iterate over the folds
for train_index, val_index in kfold.split(X_train, y_train):
    # Split the data into training and validation sets for this fold
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Train a logistic regression model on the training data for this fold
    model = LogisticRegression()
    model.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model on the validation data for this fold
    y_val_pred = model.predict(X_val_fold)
    score = accuracy_score(y_val_fold, y_val_pred)
    scores.append(score)

# Compute the mean and standard deviation of the performance metrics across folds
mean_score = np.mean(scores)
std_score = np.std(scores)

# Train the final model on the entire training set
final_model = LogisticRegression()
final_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_test_pred = final_model.predict(X_test)
test_score = accuracy_score(y_test, y_test_pred)
