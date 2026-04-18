import numpy as np
import matplotlib.pyplot as plt

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

X_val = X[3:]
Y_val = Y[3:]

def he_int(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

params = {
    "W1": he_int(2,6),
    "B1": np.zeros((1,6)),
    "W2": he_int(6,1),
    "B2": np.zeros((1,1))
}

def int_adam(params):
    adam = {}
    for key in params:
        adam["m" + key] = np.zeros_like(params[key])
        adam["v" + key] = np.zeros_like(params[key])
    return adam
adam = int_adam(params)

def ReLU(z):
    return np.maximum(0,z)
def ReLU_deriv(z):
    return (z > 0).astype(float)
def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def apply_dropout(A, rate=0.3):
    mask = (np.random.rand(*A.shape) > rate).astype(float)
    return A * mask / (1 - rate), mask 


def compute_loss(Y, Y_hat, paramas, l2=0.01):
    bce = -np.mean(
        Y * np.log(Y_hat + 1e-8) + 
        (1 - Y ) * np.log(1 - Y_hat + 1e-8)
    )
    l2_term = l2 * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
    return bce + l2_term 

def forward(X, params, trainig=True):
    Z1 = X @ params["W1"] + params["B1"]
    A1 = ReLU(Z1)

    if trainig:
        A1, mask = apply_dropout(A1)
    else:
        mask = None
    
    Z2 = A1 @ params["W2"] + params["B2"]
    A2 = Sigmoid(Z2)

    cache = (Z1, A1, Z2, A2, mask)
    return A2, cache 

def backward(X, Y, params, cache, adam, t, lr=0.01, l2=0.01):

    Z1, A1, Z2, A2, mask = cache

    m = X.shape[0]

    DZ2 = A2 - Y 
    DW2 = (A1.T @ DZ2) / m + l2 * params["W2"]
    DB2 = np.mean(DZ2, axis=0, keepdims=True)

    DA1 = DZ2 @ params["W2"].T 

    if mask is not None:
        DA1 *= mask / (1 - 0.3)
    
    DZ1 = DA1 * ReLU_deriv(Z1)
    DW1 = (X.T @ DZ1) / m + l2 * params["W1"]
    DB1 = np.mean(DZ1, axis=0,keepdims=True)
    
    grads = {"W1": DW1,"B1": DB1, "W2": DW2, "B2": DB2 }

    for key in params:
        adam["m" + key] = 0.9 * adam["m" + key] + 0.1 * grads[key]
        adam["v" + key] = 0.999 * adam["v" + key] + 0.001 * (grads[key] ** 2)

        m_hat = adam["m" + key] / (1 - 0.9 ** t)
        v_hat = adam["v" + key] / (1 - 0.999 ** t)

        params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

def get_batches(X, Y, batch_size=2):
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    for i in range(0,len(X), batch_size):
        yield X[i:i+batch_size], Y[i:i+batch_size]

epochs = 3000
patience = 200
best_val = float("inf")
counter = 0 

train_losses = []
val_losses = []

for epoch in range(1, epochs+1):

    for Xb, Yb, in get_batches(X_train, Y_train):
        Y_hat, cache = forward(Xb, params, trainig=True)
        backward(Xb, Yb, params, cache, adam, t=epoch)

    train_pred,_= forward(X_train, params, trainig=False)
    val_pred,_ = forward(X_val, params, trainig=False)

    train_loss = compute_loss(Y_train, train_pred, params)
    val_loss = compute_loss(Y_val, val_pred, params)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
    else:
        counter += 1
    if counter > patience:
        print("Early stop activado")
        break 
    if epoch % 200 == 0:
        print(f"epoch:{epoch} | train {train_loss:.4f} | val {val_loss:.4f}")

final_pred,_ = forward(X, params, trainig=False)
print("Prediciones finales:")
print(final_pred)

plt.plot(train_losses, label = "train")
plt.plot(val_losses, label = "validation")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title("trainig curve")
plt.show() 