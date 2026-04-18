import numpy as np 

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y = np.array([
    [0],[0],[0],[1]
])

W1 = np.random.randn(3,2)
B1 = np.random.randn(1,3)

W2 = np.random.randn(1,3)
B2 = np.random.randn(1,1)

learning_rate = 0.005
epochs = 1000
batch_size = 2

def ReLU(z):
    return np.maximum(0,z)

def ReLU_Derivative(z):
    return (z>0).astype(float)

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    indecis = np.random.permutation(len(X))
    X_shuffled = X[indecis]
    Y_shuffled = Y[indecis] 

    for i in range(0,len(X), batch_size):

        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        Z1 = X_batch @ W1.T + B1
        A1 = ReLU(Z1)

        Z2 = A1 @ W2.T + B2
        Y_hat = Sigmoid(Z2)

        loss = np.mean(
            Y_batch*np.log(Y_hat) + 
            (1-Y_batch)*np.log(1-Y_hat)
        )

        DZ2 = Y_hat - Y_batch
        DW2 = (DZ2.T @ A1) / batch_size
        DB2 = np.mean(DZ2, axis=0, keepdims=True)

        DA1 = DZ2 @ W2
        DZ1 = DA1 * ReLU_Derivative(Z1)

        DW1 = (DZ1.T @ X_batch) / batch_size
        DB1 = np.mean(DZ1, axis=0, keepdims=True)

        W2 -= learning_rate * DW2 
        B2 -= learning_rate * DB2
        W1 -= learning_rate * DW1
        B1 -= learning_rate * DB1

    if epoch % 100 == 0:
        print("El epoch es de:",epoch, "Loss es de:",loss)

print("\nResultados finales:")

Z1 = X @ W1.T + B1
A1 = ReLU(Z1)
Z2 = A1 @ W2.T + B2
Y_hat = Sigmoid(Z2)

print(Y_hat)