import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------- DATOS ----------------
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y = np.array([
    [1],
    [0],
    [0],
    [1]
])

X_train = X[:3]
Y_train = Y[:3]

X_val = X[3:]
Y_val = Y[3:]


# ---------------- INIT ----------------
def he_int(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

params = {
    "W1": he_int(2,6), "B1": np.zeros((1,6)),
    "gamma1": np.ones((1,6)), "beta1": np.zeros((1,6)),

    "Wres": he_int(2,6), "Bres": np.zeros((1,6)),

    "W2": he_int(6,10), "B2": np.zeros((1,10)),
    "gamma2": np.ones((1,10)), "beta2": np.zeros((1,10)),

    "W3": he_int(10,6), "B3": np.zeros((1,6)),
    "gamma3": np.ones((1,6)), "beta3": np.zeros((1,6)),

    "W4": he_int(6,1), "B4": np.zeros((1,1)),
    "gamma4": np.ones((1,1)), "beta4": np.zeros((1,1))
}

def int_adam(params):
    adam = {}
    for key in params:
        adam["m"+key] = np.zeros_like(params[key])
        adam["v"+key] = np.zeros_like(params[key])
    return adam

adam = int_adam(params)


# ---------------- ACTIVACIONES ----------------
def ReLU(z):
    return np.maximum(0,z)

def ReLU_deriv(z):
    return (z>0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))


# ---------------- DROPOUT ----------------
def apply_dropout(A, rate):
    mask = (np.random.rand(*A.shape) > rate).astype(float)
    return A * mask / (1 - rate), mask


# ---------------- BATCH NORM ----------------
def batch_norm(Z, gamma, beta, eps=1e-8):

    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean)/np.sqrt(var+eps)
    out = gamma*Z_norm + beta

    cache = (Z, Z_norm, mean, var, gamma, eps)

    return out, cache


def batch_norm_backward(dout, cache):

    Z, Z_norm, mean, var, gamma, eps = cache
    m = Z.shape[0]

    dgamma = np.sum(dout * Z_norm, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    dZ_norm = dout * gamma

    dvar = np.sum(dZ_norm*(Z-mean)*-0.5*(var+eps)**(-1.5), axis=0, keepdims=True)

    dmean = np.sum(dZ_norm*-1/np.sqrt(var+eps), axis=0, keepdims=True) + \
            dvar*np.mean(-2*(Z-mean),axis=0,keepdims=True)

    dZ = (
        dZ_norm/np.sqrt(var+eps)
        + dvar*2*(Z-mean)/m
        + dmean/m
    )

    return dZ, dgamma, dbeta


# ---------------- LOSS ----------------
def compute_loss(Y, Y_hat, params, l2=0.01):

    bce = -np.mean(
        Y*np.log(Y_hat+1e-8) +
        (1-Y)*np.log(1-Y_hat+1e-8)
    )

    l2_term = l2*(
        np.sum(params["W1"]**2)+
        np.sum(params["W2"]**2)+
        np.sum(params["W3"]**2)+
        np.sum(params["W4"]**2)+
        np.sum(params["Wres"]**2)
    )

    return bce + l2_term


# ---------------- FORWARD ----------------
def forward(X, params, training=True, rate=0.3):

    # ---- capa 1 ----
    Z1 = X@params["W1"] + params["B1"]
    Z1, cache1 = batch_norm(Z1, params["gamma1"], params["beta1"])
    A1 = ReLU(Z1)

    X_res = X@params["Wres"] + params["Bres"]
    A1 = A1 + X_res

    if training:
        A1, mask1 = apply_dropout(A1, rate)
    else:
        mask1 = None

    # ---- capa 2 ----
    Z2 = A1@params["W2"] + params["B2"]
    Z2, cache2 = batch_norm(Z2, params["gamma2"], params["beta2"])
    A2 = ReLU(Z2)

    if training:
        A2, mask2 = apply_dropout(A2, rate)
    else:
        mask2 = None

    # ---- capa 3 ----
    Z3 = A2@params["W3"] + params["B3"]
    Z3, cache3 = batch_norm(Z3, params["gamma3"], params["beta3"])
    A3 = ReLU(Z3)

    if training:
        A3, mask3 = apply_dropout(A3, rate)
    else:
        mask3 = None

    # ---- salida ----
    Z4 = A3@params["W4"] + params["B4"]
    Z4, cache4 = batch_norm(Z4, params["gamma4"], params["beta4"])
    A4 = sigmoid(Z4)

    cache = (cache1,cache2,cache3,cache4,
             A1,A2,A3,A4,
             mask1,mask2,mask3,
             X)

    return A4, cache


# ---------------- BACKWARD ----------------
def backward(X,Y,params,cache,adam,t,lr=0.01,l2=0.01,rate=0.3):

    (cache1,cache2,cache3,cache4,
     A1,A2,A3,A4,
     mask1,mask2,mask3,
     X_input) = cache

    m = X.shape[0]

    # ---- salida ----
    dZ4 = A4 - Y
    dZ4, dgamma4, dbeta4 = batch_norm_backward(dZ4, cache4)

    dW4 = (A3.T@dZ4)/m + l2*params["W4"]
    dB4 = np.mean(dZ4,axis=0,keepdims=True)

    # ---- capa 3 ----
    dA3 = dZ4@params["W4"].T

    if mask3 is not None:
        dA3 *= mask3/(1-rate)

    dZ3 = dA3*ReLU_deriv(A3)
    dZ3, dgamma3, dbeta3 = batch_norm_backward(dZ3, cache3)

    dW3 = (A2.T@dZ3)/m + l2*params["W3"]
    dB3 = np.mean(dZ3,axis=0,keepdims=True)

    # ---- capa 2 ----
    dA2 = dZ3@params["W3"].T

    if mask2 is not None:
        dA2 *= mask2/(1-rate)

    dZ2 = dA2*ReLU_deriv(A2)
    dZ2, dgamma2, dbeta2 = batch_norm_backward(dZ2, cache2)

    dW2 = (A1.T@dZ2)/m + l2*params["W2"]
    dB2 = np.mean(dZ2,axis=0,keepdims=True)

    # ---- capa 1 ----
    dA1 = dZ2@params["W2"].T

    if mask1 is not None:
        dA1 *= mask1/(1-rate)

    dWres = (X_input.T@dA1)/m
    dBres = np.mean(dA1,axis=0,keepdims=True)

    dZ1 = dA1*ReLU_deriv(A1)
    dZ1, dgamma1, dbeta1 = batch_norm_backward(dZ1, cache1)

    dW1 = (X_input.T@dZ1)/m + l2*params["W1"]
    dB1 = np.mean(dZ1,axis=0,keepdims=True)

    grads = {
        "W1":dW1,"B1":dB1,
        "W2":dW2,"B2":dB2,
        "W3":dW3,"B3":dB3,
        "W4":dW4,"B4":dB4,
        "Wres":dWres,"Bres":dBres,
        "gamma1":dgamma1,"beta1":dbeta1,
        "gamma2":dgamma2,"beta2":dbeta2,
        "gamma3":dgamma3,"beta3":dbeta3,
        "gamma4":dgamma4,"beta4":dbeta4
    }

    for key in params:
        adam["m"+key] = 0.9*adam["m"+key] + 0.1*grads[key]
        adam["v"+key] = 0.999*adam["v"+key] + 0.001*(grads[key]**2)

        m_hat = adam["m"+key]/(1-0.9**t)
        v_hat = adam["v"+key]/(1-0.999**t)

        params[key] -= lr*m_hat/(np.sqrt(v_hat)+1e-8)


# ---------------- TRAIN ----------------
epochs = 3000
patience = 200

best_val = float("inf")
counter = 0

train_losses=[]
val_losses=[]

for epoch in range(1,epochs+1):

    Y_hat,cache = forward(X_train,params,True)
    backward(X_train,Y_train,params,cache,adam,epoch)

    train_pred,_ = forward(X_train,params,False)
    val_pred,_ = forward(X_val,params,False)

    train_loss = compute_loss(Y_train,train_pred,params)
    val_loss = compute_loss(Y_val,val_pred,params)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        counter=0
    else:
        counter+=1

    if counter>patience:
        print("Early stop activado")
        break

    if epoch%200==0:
        print(epoch,train_loss,val_loss)


# ---------------- RESULTADOS ----------------
final_pred,_ = forward(X,params,False)

print("Predicciones finales")
print(final_pred)

plt.plot(train_losses,label="train")
plt.plot(val_losses,label="val")
plt.legend()
plt.show()