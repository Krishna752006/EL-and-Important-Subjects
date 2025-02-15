#Import all the necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('/srv/shareddata/datasets/ps2/Churn_Modelling/Churn_Modelling.csv')
X = data.drop(['Exited'], axis=1).values
y = data['Exited'].values.reshape(-1, 1)

# Step 3 Use Standard Scaler(X_scaled)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

input_size = X_train.shape[1]  # Number of input features
hidden_size = 10  # Number of neurons in the hidden layer
output_size = 1  # Binary classification (1 output node)

def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

#Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of the sigmoid function
def sigmoid_derivative(z):
    return z * (1 - z)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def compute_cost(A2, y):
    m = y.shape[0]
    cost = -(1/m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    return np.squeeze(cost)

def backward_propagation(X, y, Z1, A1, A2, W2):
    m = X.shape[0]

    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2

def train_model(X,y,hidden_size,learning_rate,epochs):
    input_size=X.shape[1]
    output_size=1
    W1,b1,W2,b2=initialize_parameters(input_size,hidden_size,output_size)
    for i in range(epochs):
        Z1,A1,Z2,A2=forward_propagation(X,W1,b1,W2,b2)
        y_pred=A2>0
        cost=compute_cost(A2,y_train)
        dW1,db1,dW2,db2=backward_propagation(X,y,Z1,A1,Z2,A2)
        W1,b1,W2,b2=update_parameters(W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate)
    return W1,b1,W2,b2

def predict(X,W1,b1,W2,b2):
    Z1=np.dot(X,W1)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(A1,W2)+b2
    A2=sigmoid(Z2)
    predictions=A2>0.5
    return predictions

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)