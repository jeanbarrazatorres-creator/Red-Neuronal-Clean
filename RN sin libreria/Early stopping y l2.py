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

X_train = X[:3]
Y_train = Y[:3]

X_val = X[:3]
Y_val = Y[:3]

W1 = np.random.randn(3,2)
B1 = np.random.randn(1,3)
W2 = np.random.randn(1,3)
B2 = np.random.randn(1,1)

mW1 = np.zeros_like(W1)
vW1 = np.zeros_like(W1)
mB1 = np.zeros_like(B1)
vB1 = np.zeros_like(B1)
mW2 = np.zeros_like(W2)
vW2 = np.zeros_like(W2)
mB2 = np.zeros_like(B2)
vB2 = np.zeros_like(B2)

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
t = 0 
learning_rate = 0.005
epochs = 1000
batch_size = 2 
lambda_l2 = 0.01
best_val_loss = float("inf")
patience = 500
counter = 0 

def ReLU(z):
    return np.maximum(0,z)
def ReLU_derevative(z):
    return (z>0).astype(float)
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))
def compute_loss(Y_true, Y_pred):
    return -np.mean(
        Y_true * np.log(Y_pred + 1e-8) +
        (1 - Y_true) * np.log(1 - Y_pred + 1e-8)
    )


for epoch in range(epochs):

    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    for i in range(0, len(X_train), batch_size):

        X_batch = X_shuffled[i:i+batch_size]
        Y_batch = Y_shuffled[i:i+batch_size]

        Z1 = X_batch @ W1.T + B1 
        A1 = ReLU(Z1)

        Z2 = A1 @ W2.T + B2 
        Y_hat = Sigmoid(Z2)

        DZ2  = Y_hat - Y_batch
        DW2 = (DZ2.T @ A1) / batch_size
        DB2 = np.mean(DZ2, axis=0, keepdims=True)

        DA1 = DZ2 @ W2
        DZ1 = DA1 * ReLU_derevative(Z1)

        DW1 = (DZ1.T @ X_batch) / batch_size
        DB1 = np.mean(DZ1, axis=0, keepdims=True)

        #regulation L2

        DW2 += lambda_l2 * W2
        DW1 += lambda_l2 * W1

        t += 1

        mW2 = beta1 * mB2 + (1-beta1) * DW2
        vW2 = beta2 * vW2 + (1-beta2) * (DW2 ** 2)
        mW2_corr = mW2 / (1-beta1**t)
        vW2_corr = vW2 / (1-beta2**t)

        W2 -= learning_rate * mW2_corr / (np.sqrt(vW2_corr) + epsilon)

        mB2 = beta1 * mB2 + (1-beta1) * DB2
        vB2 = beta2 * vB2 + (1-beta2) * (DB2 ** 2)
        mB2_corr = mB2 / (1-beta1**t)
        vB2_corr = vB2 / (1-beta2**t)

        B2 -= learning_rate * mB2_corr / (np.sqrt(vB2_corr) + epsilon)

        mW1 = beta1 * mW1 + (1-beta1) * DW1
        vW1 = beta2 * vW1 + (1-beta2) * (DW1**2)
        mW1_corr = mW1 / (1-beta1**t)
        vW1_corr = vW1 / (1-beta2**t)

        W1 -= learning_rate * mW1_corr / (np.sqrt(vW1_corr) + epsilon)

        mB1 = beta1 * mB1 + (1-beta1) * DB1
        vB1 = beta2 * vB1 + (1-beta2) * (DB2**2)
        mB1_corr = mB1 / (1-beta1**t)
        vB1_corr = vB1 / (1-beta2**t)

        B1 -= learning_rate * mB1_corr / (np.sqrt(vB1_corr) + epsilon)

        Z1_train = X_train @ W1.T + B1
        A1_train = ReLU(Z1_train)
        Z2_train = A1_train @ W2.T + B2
        Y_train_hat = Sigmoid(Z2_train)

        train_loss = compute_loss(Y_train, Y_train_hat)

        l2 = lambda_l2 * (np.sum(W1**2) + np.sum(W2**2))

        train_loss += l2

        Z1_val = X_val @ W1.T + B1
        A1_val = ReLU(Z1_val)
        Z2_val = A1_val @ W2.T + B2
        Y_val_hat = Sigmoid(Z2_val)

        val_loss = compute_loss(Y_val, Y_val_hat)

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            counter = 0
        else:
            conter += 1 

        if counter > patience:
            print("early spotpping activado")
            break
        if epoch % 200 == 0:

          print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# RESULTADOS FINALES
# =========================

print("\nPredicciones finales:")

Z1 = X @ W1.T + B1
A1 = ReLU(Z1)

Z2 = A1 @ W2.T + B2
Y_hat = Sigmoid(Z2)

print(Y_hat)

