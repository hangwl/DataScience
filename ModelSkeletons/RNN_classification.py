# RNN models can be applied to any task that involves sequential data analysis, where the order and context of the data are important factors in making predictions 
# or generating new output.

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define the model architecture
model = Sequential()
# choice of RNN layer (e.g., SimpleRNN, LSTM, GRU) depends on the specific task and the nature of the data
# embedding layer is used to map each word in a vocabulary to a high-dimensional vector representation, which can then be used as input to a neural network model
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len))
# LSTM (Long Short-Term Memory) layer is a type of recurrent neural network (RNN) layer that is commonly used for processing sequential data, such as text or time series. 
# Unlike a traditional RNN, which can suffer from the vanishing gradient problem, the LSTM layer is designed to overcome this issue by allowing information to flow through 
# the network for longer periods of time.
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid')) # type of layer that performs a linear operation on the input data, followed by a non-linear activation function

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
print('Test precision:', precision)
print('Test recall:', recall)

# Generate predictions
y_pred = model.predict_classes(X_test)

# Print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
