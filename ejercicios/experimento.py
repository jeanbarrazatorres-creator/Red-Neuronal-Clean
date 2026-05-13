import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ========================
# CONFIG
# ========================

LR = 0.005
EPOCHS = 5000
PATIENCE = 200
SEED = 42

torch.manual_seed(SEED)

# ========================
# DATOS
# ========================

def cargar_datos():
    x = torch.tensor([
        [19,1,1,0],[19,1,1,1],[19,1,0,0],
        [20,1,1,0],[20,1,1,1],[20,1,0,0],
        [21,2,1,0],[21,2,1,1],[21,2,0,0],
        [22,3,1,0],[22,3,0,0],[22,3,1,1],
        [25,5,1,0],[25,5,0,0],[25,5,1,1],
        [26,6,1,0],[26,6,0,0],[26,6,1,1],
        [27,7,1,0],[27,7,1,1],[27,7,0,0],
        [28,8,1,0],[28,8,1,1],[28,8,0,0],
        [29,9,1,0],[29,9,0,0],[29,9,1,1],
        [30,10,1,0],[30,10,1,1],[30,10,0,0],
        [40,15,1,0],[40,15,1,1],[40,15,0,0],
        [41,12,1,0],[41,12,1,1],[41,12,0,0],
        [45,14,1,0],[45,14,1,1],[45,14,0,0],
        [50,30,1,0],[50,30,1,1],[50,30,0,0],
    ], dtype=torch.float32)

    y = torch.tensor([
        [0],[0],[0],[0],[0],[0],
        [0],[0],[0],[1],[0],[0],
        [1],[0],[0],[1],[0],[0],
        [1],[0],[0],[1],[1],[0],
        [1],[0],[1],[1],[1],[0],
        [1],[1],[0],[1],[1],[0],
        [1],[1],[0],[1],[1],[0],
        
    ], dtype=torch.float32)

    return x, y

# ========================
# PREPROCESAMIENTO
# ========================

def dividir(x, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.3, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def normalizar_train(x):
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    return (x - mean) / std, mean, std

def aplicar_norm(x, mean, std):
    return (x - mean) / std

# ========================
# MODELO (con dropout)
# ========================

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3),   # 🔥 regularización

            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),   # 🔥 regularización

            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

# ========================
# ENTRENAMIENTO
# ========================

def train_epoch(model, X, y, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_epoch(model, X, y, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = loss_fn(pred, y)
    return loss.item()

def entrenar(model, X_train, y_train, X_val, y_val, mean, std):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")
    counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, X_train, y_train, loss_fn, optimizer)
        val_loss = eval_epoch(model, X_val, y_val, loss_fn)

        # 🔥 Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std
            }, "modelo.pth")
        else:
            counter += 1

        if counter >= PATIENCE:
            print("Early stopping en epoch", epoch)
            break

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

# ========================
# EVALUACIÓN
# ========================

def evaluar(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

    y_true = y_test.numpy()
    y_pred = preds.numpy()

    acc = (preds == y_test).float().mean()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nResultados:")
    print("Accuracy:", float(acc))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

# ========================
# MAIN
# ========================

def main():
    x, y = cargar_datos()

    # ✅ dividir primero
    X_train, X_val, X_test, y_train, y_val, y_test = dividir(x, y)

    # ✅ normalizar correctamente
    X_train, mean, std = normalizar_train(X_train)
    X_val = aplicar_norm(X_val, mean, std)
    X_test = aplicar_norm(X_test, mean, std)

    model = MLP(input_size=4)

    entrenar(model, X_train, y_train, X_val, y_val, mean, std)

    checkpoint = torch.load("modelo.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluar(model, X_test, y_test)

if __name__ == "__main__":
    main()