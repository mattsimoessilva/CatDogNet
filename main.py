# use_model.py

from neural_network import NeuralNetwork
from image_processing import load_and_preprocess_image
import pickle

# Load the saved parameters
with open('model_params.pkl', 'rb') as f:
    params = pickle.load(f)

# Create an instance of the neural network
neural_net = NeuralNetwork(len(params['W1']), len(params['W1'][0]), len(params['W2']), alpha=0.01, batch_size=32, epochs=100, lambd=0.7)

# Assign the parameters to the neural network
neural_net.W1 = params['W1']
neural_net.b1 = params['b1']
neural_net.W2 = params['W2']
neural_net.b2 = params['b2']

# Load and preprocess the image from the given path
image_path = 'data/image.png'
X_new = load_and_preprocess_image(image_path)

# Check if the image was loaded successfully
if X_new is not None:
    # Make predictions on the new image
    predictions = neural_net.predict([X_new])

    # Print the predictions
    print("Predictions:", predictions)
else:
    print("Error loading or processing the image.")

