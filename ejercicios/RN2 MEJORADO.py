import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# ------------------------
# Datos
# ------------------------

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
    [0],[0],[0],
    [0],[0],[0],
    [0],[0],[0],
    [1],[0],[0],
    [1],[0],[0],
    [1],[0],[0],
    [1],[0],[0],
    [1],[1],[0],
    [1],[0],[1],
    [1],[1],[0],
    [1],[1],[0],
    [1],[1],[0],
    [1],[1],[0],
    [1],[1],[0]
], dtype=torch.float32)

# ------------------------
# Normalización
# ------------------------

mean = x.mean(dim=0)
std = x.std(dim=0)
x = (x - mean) / std

# ------------------------
# Train / Test split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# ------------------------
# Modelo con CLASS + DROPOUT
# ------------------------

class RedAcceso(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 8)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

model = RedAcceso()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ------------------------
# Entrenamiento
# ------------------------

for epoch in range(5000):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print("Epoch", epoch, "Loss", loss.item())

# ------------------------
# Accuracy real
# ------------------------

with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    acc = (preds == y_test).float().mean()
    print("\nAccuracy en datos nuevos:", float(acc))

# ------------------------
# App final
# ------------------------

def evaluar():
    a = float(input("Edad: "))
    b = float(input("Experiencia: "))
    c = float(input("Credencial (1/0): "))
    d = float(input("Antecedentes (1/0): "))

    entrada = torch.tensor([[a,b,c,d]])
    entrada = (entrada - mean) / std

    with torch.no_grad():
        logit = model(entrada)
        prob = torch.sigmoid(logit)

        if prob >= 0.8:
            print("Puede acceder", float(prob))
        else:
            print("No puede acceder", float(prob))

while True:
    print("\n1) Evaluar persona")
    print("2) Salir")
    op = input("Opción: ")

    if op == "1":
        evaluar()
    elif op == "2":
        break
