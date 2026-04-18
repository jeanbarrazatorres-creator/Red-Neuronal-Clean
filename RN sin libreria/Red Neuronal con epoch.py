import numpy as np 

np.random.seed(0)

X = np.array([
    [0,0],
    [0,1],
    [1,1],
    [1,1]
])

Y = np.array([
    [0],
    [0],
    [0],
    [1]
])

W1 = np.random.randn(3,2)
B1 = np.random.randn(3)

W2 = np.random.randn(1,3)
B2 = np.random.randn(1)

Learning_Rate = 0.01
Epochs = 1000

def ReLu(z):
    return np.maximum(0,z)
def ReLU_derivative(z):
    return (z > 0).astype(float)
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

for epoch in range(Epochs):

    total_Loss = 0 

    for i in range(len(X)):

        x = X[i]
        y = Y[i]

        z1 = np.dot(W1 , x) + B1
        a1 = ReLu(z1)

        z2 = np.dot(W2, a1) + B2
        y_hat = sigmoid(z2)

        loss = -(y*np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        total_Loss += loss 

        dz2 = y_hat - y 
        dW2 = dz2.reshape(1,1) @ a1.reshape(1,3)
        dB2 = dz2

        da1 = np.dot(W2.T, dz2)
        dz1 = da1 * ReLU_derivative(z1)

        dW1 = dz1.reshape(3,1) @ x.reshape(1,2)
        dB1 = dz1

        W2 -= Learning_Rate * dW2
        B2 -= Learning_Rate * dB2

        W1 -= Learning_Rate * dW1 
        B1 -= Learning_Rate * dB1

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", total_Loss)

print("\nResultados finales:")
for i in range(len(X)):
    x = X[i]
    z1 = np.dot(W1,x) + B1
    a1 = ReLu(z1)
    z2 = np.dot(W2,a1) + B2
    y_hat = sigmoid(z2)
    print(X[i], "→", y_hat)