import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with a quadratic relationship
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 2 * X**2 + 1 * X + 3 + np.random.randn(100, 1)

# Add a bias term to X
X_b = np.c_[np.ones((100, 1)), X, X**2]

# Set the learning rate and number of iterations
learning_rate = 0.01
n_iterations = 1000

# Initialize random coefficients
theta = np.random.randn(3, 1)

# Perform gradient descent
for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

# Print the final coefficients
print("Final Coefficients (theta):", theta)

# Plot the data and the quadratic regression curve
plt.scatter(X, y, color='blue')
x_values = np.linspace(0, 2, 100).reshape(-1, 1)
X_values_b = np.c_[np.ones((100, 1)), x_values, x_values**2]
plt.plot(x_values, X_values_b.dot(theta), color='red', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quadratic Regression with Gradient Descent')
plt.show()
