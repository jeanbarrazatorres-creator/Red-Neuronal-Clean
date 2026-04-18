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
    "W1": he_int(2,3),
    "B1": np.zeros((1,3)),
    "gamma1": np.ones((1,3)),
    "beta1": np.zeros((1,3)),

    # residual
    "Wres": he_int(2,3),
    "Bres": np.zeros((1,3)),

    "W2": he_int(3,1),
    "B2": np.zeros((1,1)),
    "gamma2": np.ones((1,1)),
    "beta2": np.zeros((1,1))
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

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_dropout(A, rate=0.3):
    mask = (np.random.rand(*A.shape) > rate).astype(float)
    return A * mask / (1 - rate), mask 


def batch_norm(Z, gamma, beta, eps=1e-8):

    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(var + eps)
    out = gamma * Z_norm + beta 

    return out , Z_norm, mean , var 


def compute_loss(Y, Y_hat, params, l2 = 0.01):

    bce = -np.mean(
        Y * np.log(Y_hat + 1e-8) +
        (1 - Y) * np.log(1 - Y_hat + 1e-8)
    )

    l2_term = l2 * (
        np.sum(params["W1"] ** 2) + 
        np.sum(params["W2"] ** 2) +
        np.sum(params["Wres"] ** 2)
    )

    return bce + l2_term


def forward(X, params, trainig=True):

    # layer 1
    Z1  = X @ params["W1"] + params["B1"]

    Z1_bn, Z1_norm, mean1, var1 = batch_norm(
        Z1,
        params["gamma1"],
        params["beta1"]
    )

    A1 = ReLU(Z1_bn)

    # residual
    X_res = X @ params["Wres"] + params["Bres"]

    A1 = A1 + X_res

    if trainig:
        A1, mask = apply_dropout(A1)
    else: 
        mask = None
    
    # output
    Z2 = A1 @ params["W2"] + params["B2"]

    Z2_bn, Z2_norm, mean2, var2 = batch_norm(
        Z2,
        params["gamma2"],
        params["beta2"]
    )

    A2 = sigmoid(Z2_bn)

    cache = (
        Z1, Z1_bn, Z1_norm, mean1, var1,
        Z2, Z2_bn, Z2_norm, mean2, var2,
        A1, A2, mask, X, X_res
    )

    return A2, cache


def backward(X, Y, params, cache, adam, t, lr=0.01, l2=0.01):

    (
        Z1, Z1_bn, Z1_norm, mean1, var1,
        Z2, Z2_bn, Z2_norm, mean2, var2,
        A1, A2, mask, X_input, X_res
    ) = cache

    m = X.shape[0]
    eps = 1e-8

    # output
    DZ2_bn = A2 - Y 

    dgamma2 = np.sum(DZ2_bn * Z2_norm, axis=0, keepdims=True)
    dbeta2 = np.sum(DZ2_bn, axis=0, keepdims=True)

    DZ2_norm = DZ2_bn * params["gamma2"]

    dvar2 = np.sum(
        DZ2_norm * (Z2 - mean2) * -0.5 * (var2 + eps) ** (-1.5),
        axis=0,
        keepdims=True
    )

    dmean2 = np.sum(
        DZ2_norm * -1 /  np.sqrt(var2 + eps),
        axis=0,
        keepdims=True
    ) + dvar2 * np.mean(-2 * (Z2 - mean2), axis=0, keepdims=True)

    DZ2 = (
        DZ2_norm / np.sqrt(var2 + eps)
        + dvar2 * 2 *  (Z2 - mean2) / m 
        + dmean2 / m
    )

    DW2 = (A1.T @ DZ2) / m + l2 * params["W2"]
    DB2 = np.mean(DZ2, axis=0, keepdims=True)

    DA1 = DZ2 @ params["W2"].T

    # dropout backward
    if mask is not None:
        DA1 *= mask / (1 - 0.3)

    # residual split
    D_residual = DA1
    D_main = DA1

    # residual weights
    DWres = (X_input.T @ D_residual) / m + l2 * params["Wres"]
    DBres = np.mean(D_residual, axis=0, keepdims=True)

    # ReLU
    DZ1_ReLU = D_main * ReLU_deriv(Z1_bn)

    dgamma1 = np.sum(DZ1_ReLU * Z1_norm, axis=0, keepdims=True)
    dbeta1 = np.sum(DZ1_ReLU, axis=0, keepdims=True)

    DZ1_norm = DZ1_ReLU * params["gamma1"]

    dvar1 = np.sum(
        DZ1_norm * (Z1 - mean1) * -0.5 * (var1 + eps) ** (-1.5),
        axis=0,
        keepdims=True
    )

    dmean1 = np.sum(
        DZ1_norm * -1 / np.sqrt(var1 + eps),
        axis=0,
        keepdims=True
    ) + dvar1 * np.mean(-2 * (Z1 -mean1), axis=0, keepdims=True)

    DZ1 = (
        DZ1_norm / np.sqrt(var1 + eps)
        + dvar1 * 2 * (Z1 - mean1) / m 
        + dmean1 / m 
    )

    DW1 = (X_input.T @ DZ1) / m + l2 * params["W1"]
    DB1 = np.mean(DZ1, axis=0, keepdims=True)

    grads = {
        "W1": DW1,
        "B1": DB1,
        "W2": DW2,
        "B2": DB2,
        "gamma1": dgamma1,
        "beta1": dbeta1,
        "gamma2": dgamma2,
        "beta2": dbeta2,
        "Wres": DWres,
        "Bres": DBres
    }

    for key in params:
        adam["m" + key] = 0.9 * adam["m" + key] + 0.1 * grads[key]
        adam["v" + key] = 0.999 * adam["v" + key] + 0.001 * (grads[key] ** 2)

        m_hat = adam["m" + key] / (1 - 0.9 ** t)
        v_hat = adam["v" + key] / (1 - 0.999 ** t)

        params[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def get_batches(X, Y, batch_size=2):

    idx = np.random.permutation(len(X))
    X, Y  = X[idx], Y[idx]

    for i in range(0, len(X), batch_size):
        yield  X[i:i+batch_size], Y[i:i+batch_size]


epochs = 3000
patience = 200

best_val = float("inf")
counter = 0 

train_losses = []
val_losses = []


for epoch in range(1, epochs+1):

    for XB, YB in get_batches(X_train, Y_train):
        Y_hat, cache = forward(XB, params, trainig=True)
        backward(XB, YB, params, cache, adam, t = epoch)

    train_pred,_ = forward(X_train, params, trainig=False)
    val_pred ,_ = forward(X_val, params, trainig=False)

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
        print(f"epoch:{epoch} | train:{train_loss:.4f} | val:{val_loss:.4f}")


final_pred,_ = forward(X, params, trainig=False)

print("Prediciones finales")
print(final_pred)


plt.plot(train_losses, label = "train")
plt.plot(val_losses, label = "validation")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training curve")
plt.show()
plt.pause(0.001)