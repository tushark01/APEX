import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

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

# Training data for AND gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Create and train perceptron
perceptron = Perceptron(input_size=2)
epochs = 100

for _ in range(epochs):
    for inputs, target in zip(X, y):
        perceptron.train(inputs, target)

# Test the perceptron
for inputs in X:
    print(f"Input: {inputs}, Output: {perceptron.predict(inputs)}")