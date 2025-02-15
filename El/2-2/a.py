import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(42)
W1 = np.random.randn(2,2) * 0.01
b1 = np.zeros((1,2))
W2 = np.random.randn(2,1) * 0.01
b2 = np.zeros((1,1))

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return np.where(z>0,1,0)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z*(1-z)

lr = 0.01
epochs = 10000

for epoch in range(epochs):
    z1 = np.dot(X,W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)

    loss = np.mean(-(y*np.log(a2) + (1-y)*np.log(1-a2)))

    dz2 = a2-y
    dw2 = np.dot(a1.T,dz2)/len(y)
    db2 = np.sum(dz2, axis = 0,keepdims = True)/len(y)
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(X.T,dz1)/len(y)
    db1 = np.sum(dz1,axis = 0,keepdims = True)

    W1 = W1 - lr*dw1
    b1 = b1 - lr*db1
    W2 = W2 - lr*dw2
    b2 = b2 - lr*db2

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

print('Testing Result: \n')

for i in range(len(X)):
    z1 = np.dot(X[i],W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1,W2) + b2
    a2 = sigmoid(z2)
    print(f'Input: {X[i]}, Predicted: {a2[0][0]:.4f}, Actual: {y[i][0]}')