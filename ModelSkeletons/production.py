import pickle
from flask import Flask, request, jsonify

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Define an endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data as a JSON object
    data = request.get_json()

    # Make a prediction using the model
    prediction = model.predict([[data['feature1'], data['feature2'], data['feature3']]])

    # Convert the prediction to a string
    if prediction[0] == 0:
        prediction_str = 'Class A'
    else:
        prediction_str = 'Class B'

    # Return the prediction as a JSON object
    return jsonify({'prediction': prediction_str})

# Start the app
if __name__ == '__main__':
    app.run()
