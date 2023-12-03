from math_functions import dot_product, matrix_addition, scalar_multiplication, transpose
import math
import random

class NeuralNetwork:
    def __init__(self, n_x, n_h, n_y, alpha=0.01, batch_size=32, epochs=100, lambd=0.7):
        # Initialize weights (W), biases (b) and hyperparameters
        self.W1 = [[random.uniform(-1, 1) for _ in range(n_h)] for _ in range(n_x)]  # Use a better weight initialization
        self.b1 = [[0] for _ in range(n_h)]
        self.W2 = [[random.uniform(-1, 1) for _ in range(n_y)] for _ in range(n_h)]
        self.b2 = [[0] for _ in range(n_y)]
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.lambd = lambd  # Regularization parameter

    def relu(self, Z):
        # ReLU activation function
        return [[max(0, z) for z in row] for row in Z]

    def relu_derivative(self, Z):
        # Derivative of ReLU function for backpropagation
        return [[int(z > 0) for z in row] for row in Z]

    def softmax(self, Z):
        # Softmax activation function
        epsilon = 1e-8  # Small constant to avoid division by zero
        expZ = [[math.exp(z) for z in row] for row in Z]
        sum_expZ = sum([sum(row) for row in expZ]) + epsilon
        return [[z / sum_expZ for z in row] for row in expZ]

    def forward_propagation(self, X):
        # Compute the output
        self.Z1 = matrix_addition(dot_product(X, self.W1), self.b1)
        self.A1 = self.relu(self.Z1)
        self.Z2 = matrix_addition(dot_product(self.A1, self.W2), self.b2)
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_cost(self, A, Y):
        # Cross-entropy cost function with L2 regularization
        m = len(Y[0])
        epsilon = 1e-8  # Small constant to avoid division by zero
        cost = -sum([sum([y * math.log(max(a, epsilon)) for a, y in zip(row_A, row_Y)]) for row_A, row_Y in zip(A, Y)]) / m
        L2_regularization_cost = (sum([sum([w**2 for w in row]) for row in self.W1]) + sum([sum([w**2 for w in row]) for row in self.W2])) * self.lambd / (2 * m)
        return cost + L2_regularization_cost

    def backward_propagation(self, X, Y):
        # Compute the error and adjust weights & biases
        m = len(X[0])
        dZ2 = matrix_addition(self.A2, scalar_multiplication(-1, Y))
        dW2 = scalar_multiplication(1 / m, dot_product(transpose(self.A1), dZ2))
        db2 = scalar_multiplication(1 / m, [sum(row) for row in transpose(dZ2)])
        dZ1 = dot_product(dZ2, transpose(self.W2))
        dW1 = scalar_multiplication(1 / m, dot_product(transpose(X), dZ1))
        db1 = scalar_multiplication(1 / m, [sum(row) for row in transpose(dZ1)])

        # Update the weights and biases
        self.W1 = matrix_addition(self.W1, scalar_multiplication(-self.alpha, dW1))
        self.b1 = matrix_addition(self.b1, scalar_multiplication(-self.alpha, db1))
        self.W2 = matrix_addition(self.W2, scalar_multiplication(-self.alpha, dW2))
        self.b2 = matrix_addition(self.b2, scalar_multiplication(-self.alpha, db2))

    def train(self, X, Y):
        # Training loop with mini-batch gradient descent
        m = len(X[0])
        permutation = list(range(m))
        random.shuffle(permutation)
        shuffled_X = [X[i] for i in permutation]
        shuffled_Y = [Y[i] for i in permutation]

        for epoch in range(self.epochs):
            epoch_cost = 0
            num_batches = int(m / self.batch_size)
            for i in range(num_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = shuffled_X[start:end]
                Y_batch = shuffled_Y[start:end]

                A2 = self.forward_propagation(X_batch)
                batch_cost = self.compute_cost(A2, Y_batch)
                self.backward_propagation(X_batch, Y_batch)

                epoch_cost += batch_cost

            if epoch % 100 == 0:
                print(f"Cost after epoch {epoch}: {epoch_cost / num_batches}")

    def predict(self, X):
        # Predict the label of an unseen image
        A2 = self.forward_propagation(X)
        return [row.index(max(row)) for row in A2]

