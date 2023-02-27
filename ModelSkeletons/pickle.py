# saving a trained model as a pickle file

from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
model = LogisticRegression()
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 0, 1, 1]
model.fit(X_train, y_train)

# Save the model as a pickle file
import pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
