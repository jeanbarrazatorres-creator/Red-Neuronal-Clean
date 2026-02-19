import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.model_selection import train_test_split

import torch

import torch

x = torch.tensor([
    # Edad, Ingresos, Deuda, Historial

    
    [16, 500, 0, 0],
    [16, 800, 100, 0],
    [17, 900, 200, 0],
    [17, 1000, 0, 0],

    
    [22, 1500, 200, 0],
    [23, 1200, 100, 0],
    [25, 2000, 500, 1],
    [28, 2500, 1000, 1],
    [29, 2700, 500, 1],

    
    [30, 3000, 1500, 0],
    [33, 3200, 2000, 0],
    [35, 4000, 3000, 0],
    [38, 4500, 1000, 1],
    [40, 5000, 1000, 1],
    [45, 6000, 2500, 1],

    
    [50, 7000, 2000, 1],
    [55, 8000, 1500, 1],
    [60, 9000, 1000, 1],

    
    [27, 2200, 1800, 0],
    [32, 3500, 500, 1],
    [41, 4800, 3500, 0],
    [36, 3900, 800, 1],
    [26, 2100, 2100, 0],
    [48, 6500, 5000, 0],
], dtype=torch.float32)


y = torch.tensor([
    
    [0],[0],[0],[0],

    
    [0],[0],[1],[1],[1],

    
    [0],[0],[0],[1],[1],[1],

    
    [1],[1],[1],

    
    [0],[1],[0],[1],[0],[0]
], dtype=torch.float32)


mean = x.mean(dim=0)
std = x.std(dim=0)
x = (x - mean) / std 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

class RedPrestamo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,16)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16,8)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(8,1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop1(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)
    
model = RedPrestamo()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)

for epoch in range (20000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

with torch.no_grad():
    train_logits = model(x_train)
    train_probs = torch.sigmoid(train_logits)
    train_preds = (train_probs >= 0.5).float()
    train_acc = (train_preds == y_train).float().mean()
     
    test_logits = model(x_test)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).float()
    test_acc = (test_preds == y_test).float().mean()

edad = float(input("ingrese su edad:"))
ingresos = float(input("ingresu su sueldo:"))
deuda = float(input("ingresu deuda:"))
historial = float(input("historial bueno? si=1 y no=0:"))

persona = torch.tensor([[edad,ingresos,deuda,historial]], dtype = torch.float32)
persona = (persona - mean) / std 

model.eval()
with torch.no_grad():
    logits = model(persona)
    prob = torch.sigmoid(logits)
    aprobacion = (prob >= 0.8).float()


print("\nProbabilidad de aprobar:", float(prob))

if aprobacion.item() == 1:
    print("Se aprobo el prestamo")
else:
    print("no se aprobo su prestamo ")