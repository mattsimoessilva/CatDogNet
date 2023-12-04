from neural_network import NeuralNetwork
from image_generator import image_generator
import pickle
import os
import os

data_path = '/content/CatDogNet'  # Update with the actual path
train_path = os.path.join(data_path, 'train')
train_files = [os.path.join(train_path, file) for file in os.listdir(train_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

train_generator = image_generator(train_files, batch_size=32)
test_generator = image_generator(os.path.join(data_path, 'test'), batch_size=32)

# Get the shape of the data
X_train_batch, Y_train_batch = next(train_generator)
n_x = X_train_batch.shape[1]
n_y = Y_train_batch.shape[1]

# Create and train the neural network
n_h = 64  # Number of neurons in the hidden layer
neural_net = NeuralNetwork(n_x, n_h, n_y, alpha=0.01, batch_size=batch_size, epochs=100, lambd=0.7)

# Train the network using the generator
neural_net.train_generator(train_generator)

# Save the trained parameters
params = {
    'W1': neural_net.W1,
    'b1': neural_net.b1,
    'W2': neural_net.W2,
    'b2': neural_net.b2
}

with open('model_params.pkl', 'wb') as f:
    pickle.dump(params, f)
