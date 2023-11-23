import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 2*np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

X_b = np.c_[np.ones((100,1)),X]

leanr_rate = 0.01
itera = 1000
theta = np.random.randn(2,1)
for i in range(itera):
    grad = 2/100 * X_b.T.dot(X_b.dot(theta)-4)
    theta = theta - leanr_rate * grad

print("final coefficent(theta):",theta )
plt.scatter(X,y,color = "red")
plt.plot(X,X_b.dot(theta),color ="blue",linewidth = 13)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Linear Regression with gradient descent")
plt.show()
