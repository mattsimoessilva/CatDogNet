from neural_network import NeuralNetwork
from data_loader import load_data
import pickle

# Load training data
data_path = '/content/CatDogNet'  # Update with the actual path
X_train, Y_train, X_test, Y_test = load_data(data_path, use_percent=0.5)

# Create and train the neural network
n_x = len(X_train[0])
n_h = 64  # Number of neurons in the hidden layer
n_y = len(set(Y_train))  # Number of classes (assuming classification)

# Checking values
print(f"n_x = {n_x}")
print(f"n_h = {n_h}")
print(f"n_y = {n_y}")
      
neural_net = NeuralNetwork(n_x, n_h, n_y, alpha=0.01, batch_size=32, epochs=100, lambd=0.7)
neural_net.train(X_train, Y_train)

# Save the trained parameters
params = {
    'W1': neural_net.W1,
    'b1': neural_net.b1,
    'W2': neural_net.W2,
    'b2': neural_net.b2
}

with open('model_params.pkl', 'wb') as f:
    pickle.dump(params, f)
