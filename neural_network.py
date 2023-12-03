import numpy as np

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, alpha=0.01, batch_size=32, epochs=100, lambd=0.7):
        # Initialize weights (W), biases (b) and hyperparameters
        self.W1 = np.random.uniform(-1, 1, size=(n_x, n_h))
        self.b1 = np.zeros((1, n_h))
        self.W2 = np.random.uniform(-1, 1, size=(n_h, n_y))
        self.b2 = np.zeros((1, n_y))
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambd = lambd  # Regularization parameter

    def relu(self, Z):
        # ReLU activation function
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        # Derivative of ReLU function for backpropagation
        return (Z > 0).astype(int)

    def softmax(self, Z):
        # Softmax activation function
        epsilon = 1e-8
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward_propagation(self, X):
            # Convert X to a NumPy array if it's a list
            X = np.array(X) if isinstance(X, list) else X

            # Ensure X has the correct shape (number of features)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            # Rest of your code remains the same
            self.Z1 = X.dot(self.W1) + self.b1
            self.A1 = self.relu(self.Z1)
            self.Z2 = self.A1.dot(self.W2) + self.b2
            self.A2 = self.softmax(self.Z2)
            return self.A2

    def compute_cost(self, A, Y):
        # Cross-entropy cost function with L2 regularization
        m = len(Y)
        epsilon = 1e-8
        cost = -np.sum(Y * np.log(np.maximum(A, epsilon))) / m
        L2_regularization_cost = (np.sum(self.W1**2) + np.sum(self.W2**2)) * self.lambd / (2 * m)
        return cost + L2_regularization_cost

    def backward_propagation(self, X, Y):
        # Compute the error and adjust weights & biases
        m = len(X)
        dZ2 = self.A2 - Y
        dW2 = (self.A1.T.dot(dZ2) + self.lambd * self.W2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = dZ2.dot(self.W2.T) * self.relu_derivative(self.Z1)
        dW1 = (X.T.dot(dZ1) + self.lambd * self.W1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update the weights and biases
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2

    def train(self, X, Y):
        # Training loop with mini-batch gradient descent
        m = len(X)
        permutation = np.random.permutation(m)
        shuffled_X = [X[i] for i in permutation]
        shuffled_Y = [Y[i] for i in permutation]

        for epoch in range(self.epochs):
            epoch_cost = 0
            num_batches = int(m / self.batch_size)
            for i in range(num_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = np.array(shuffled_X[start:end])  # Convert to NumPy array
                Y_batch = np.array(shuffled_Y[start:end])  # Convert to NumPy array

                A2 = self.forward_propagation(X_batch)
                batch_cost = self.compute_cost(A2, Y_batch)
                self.backward_propagation(X_batch, Y_batch)

                epoch_cost += batch_cost

            if epoch % 100 == 0:
                print(f"Cost after epoch {epoch}: {epoch_cost / num_batches}")
                
    def predict(self, X):
        # Predict the label of an unseen image
        A2 = self.forward_propagation(X)
        return np.argmax(A2, axis=1)
