import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SklearnPerceptron

def accuracy(y_true, y_predicted):
    return np.sum(y_true == y_predicted) / len(y_true)

# Define dataset
X = [0.21, 0.52, 0.92, 0.36, 0.55, 0.45, 0.34, 0.67, 0.04, 0.77]
y = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1]
X = np.array([[x] for x in X])
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print("*" * 100)
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)
print("X_train:")
print(X_train)
print("*" * 100)

# Initialize and train Perceptron model using scikit-learn
p = SklearnPerceptron(eta0=0.01, max_iter=1000)
p.fit(X_train, y_train)

# Make predictions and calculate accuracy
prediction = p.predict(X_test)
print("Predictions:")
print(prediction)
print("Accuracy:", accuracy(y_test, prediction))

# Predict new data
new_data = np.array([[0.31], [0.88]])
new_predictions = p.predict(new_data)
print("Predictions for new data:")
print(new_predictions)

# Custom Perceptron implementation
def unitStep(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    def __init__(self, lr, itrn=100):
        self.lr = lr
        self.itrn = itrn
        self.activation = unitStep
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        # Learn weights
        for _ in range(self.itrn):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)

                # Weight update
                error = self.lr * (y[idx] - y_predicted)
                self.weights += error * x_i
                self.bias += error

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_output)
        return y_predicted

# Example usage of the custom Perceptron
custom_perceptron = Perceptron(lr=0.01, itrn=1000)
custom_perceptron.fit(X_train, y_train)
custom_prediction = custom_perceptron.predict(X_test)
print("Custom Perceptron Predictions:")
print(custom_prediction)
print("Custom Perceptron Accuracy:", accuracy(y_test, custom_prediction))