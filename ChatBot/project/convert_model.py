import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the pickle model
with open('plant_model.pkl', 'rb') as f:
    pkl_model = pickle.load(f)

# Create a Keras Sequential model
keras_model = Sequential([
    Dense(pkl_model.n_features_in_, input_shape=(pkl_model.n_features_in_,)),
    Dense(64, activation='relu'),
    Dense(len(pkl_model.classes_), activation='softmax')
])

# Copy weights from pickle model to Keras model
weights = []
for layer in keras_model.layers:
    if isinstance(layer, Dense):
        w = layer.get_weights()
        weights.append(w)

keras_model.set_weights(weights)

# Save the model in TensorFlow.js format
keras_model.save('tf_model')