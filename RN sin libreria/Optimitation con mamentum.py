import numpy as np 

np.random.seed(0)

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
Y = np.array([
    [0],
    [0],
    [0],
    [1]
])

X_Traind = X[:3]
Y_Traind = Y[:3]

X_Val = X[:3]
Y_Val = Y[:3]

W1 = np.random.randn(3,2)
B1 = np.random.randn(1,3)

W2 = np.random.randn(1,3)
B2 = np.random.randn(1,1)

#Mamwntun 

vW1 = np.zeros_like(W1)
vB1 = np.zeros_like(B1)
vW2 = np.zeros_like(W2)
vB2 = np.zeros_like(B2)

Learning_rate = 0.001
beta = 0.9
epochs = 1000
batch_size = 2 

train_losses = []
van_losses = []

def ReLU(z):
    return np.maximum(0,z)
def ReLU_derivative(z):
    return (z>0).astype(float)
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_loss(Y_true, Y_pred):
    return -np.mean(
        Y_true*np.log(Y_pred + 1e-8) +
        (1 - Y_true)*np.log(1 - Y_pred + 1e-8)
    )

for epoch in range(epochs):
    
    indices = np.random.permutation(len(X_Traind))
    X_Shuffled = X_Traind[indices]
    Y_shuffled = Y_Traind[indices]

    for i in range(0,len(X_Traind), batch_size):

        X_batch = X_Shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        Z1 = X_batch @ W1.T + B1
        A1 = ReLU(Z1)

        Z2 = A1 @ W2.T + B2
        Y_hat = Sigmoid(Z2)

        DZ2 = Y_hat - Y_batch
        DW2 = (DZ2.T @ A1) / batch_size
        DB2 = np.mean(DZ2, axis=0, keepdims = True)

        DA1 = DZ2 @ W2
        DZ1 = DA1 * ReLU_derivative(Z1)

        DW1 = (DZ1.T @ X_batch) / batch_size
        DB1 = np.mean(DZ1, axis=0, keepdims = True)

        #momentum update
        vW2 = beta * vW2 + (1 - beta)*vW2
        vB2 = beta * vB2 + (1 - beta)*vB2
        vW1 = beta * vW1 + (1 - beta)*vW1
        vB1 = beta * vB1 + (1 - beta)*vB1

        Z1_train = X_Traind @ W1.T + B1
        A1_train = ReLU(Z1_train)
        Z2_train = A1_train @ W2.T + B2 
        Y_traind_hat = Sigmoid(Z2_train)

        train_loss = compute_loss(Y_Traind, Y_traind_hat)
        train_losses.append(train_loss)

        Z1_val = X_Val @ W1.T + B1
        A1_val = ReLU(Z1_val)
        Z2_val = A1_val @ W2.T + B2
        Y_val_hat = Sigmoid(Z2_val)

        val_loss = compute_loss(Y_Val, Y_val_hat)
        van_losses.append(val_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("\nResultados finales:")

Z1 = X @ W1.T + B1
A1 = ReLU(Z1)
Z2 = A1 @ W2.T + B2
Y_hat = Sigmoid(Z2)

print(Y_hat)
