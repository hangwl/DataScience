from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Create the model architecture
model = Sequential()

# Add the RNN layer with 32 units and input shape of (timesteps, features)
model.add(SimpleRNN(32, input_shape=(timesteps, features)))

# Add a dense layer with 1 unit and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
