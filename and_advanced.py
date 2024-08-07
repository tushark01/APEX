import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.errors = []

    def activate(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        sum_inputs = np.dot(inputs, self.weights) + self.bias
        return self.activate(sum_inputs)

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error
        return error

def plot_decision_boundary(perceptron, X, y):
    plt.figure(figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA'])
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#00FF00']), edgecolors='black')
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Perceptron Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=2)
epochs = 100
errors = []

for epoch in range(epochs):
    epoch_errors = []
    for inputs, target in zip(X, y):
        error = perceptron.train(inputs, target)
        epoch_errors.append(abs(error))
    errors.append(np.mean(epoch_errors))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), errors)
plt.title("Perceptron Learning Curve")
plt.xlabel("Epochs")
plt.ylabel("Average Error")
plt.show()

# Plot decision boundary
plot_decision_boundary(perceptron, X, y)

# Test the perceptron
for inputs in X:
    print(f"Input: {inputs}, Output: {perceptron.predict(inputs)}")

# Visualize weights and bias
plt.figure(figsize=(8, 6))
plt.bar(['Weight 1', 'Weight 2', 'Bias'], [perceptron.weights[0], perceptron.weights[1], perceptron.bias])
plt.title("Perceptron Weights and Bias")
plt.ylabel("Value")
plt.show()