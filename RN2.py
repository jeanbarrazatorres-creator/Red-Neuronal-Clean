import torch
import torch.nn as nn
import torch.optim as optim 


x = torch.tensor([

    [19,1,1,0],
    [19,1,1,1],
    [19,1,0,0],

    [20,1,1,0],
    [20,1,1,1],
    [20,1,0,0],

    [21,2,1,0],
    [21,2,1,1],
    [21,2,0,0],

    [22,3,1,0],
    [22,3,0,0],
    [22,3,1,1],

    [25,5,1,0],
    [25,5,0,0],
    [25,5,1,1],

    [26,6,1,0],
    [26,6,0,0],
    [26,6,1,1],

    [27,7,1,0],
    [27,7,1,1],
    [27,7,0,0],

    [28,8,1,0],
    [28,8,1,1],
    [28,8,0,0],

    [29,9,1,0],
    [29,9,0,0],
    [29,9,1,1],

    [30,10,1,0],
    [30,10,1,1],
    [30,10,0,0],

    [40,15,1,0],
    [40,15,1,1],
    [40,15,0,0],

    [41,12,1,0],
    [41,12,1,1],
    [41,12,0,0],

    [45,14,1,0],
    [45,14,1,1],
    [45,14,0,0],

    [50,30,1,0],
    [50,30,1,1],
    [50,30,0,0],
], dtype = torch.float32)

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
], dtype = torch.float32)

mean = x.mean(dim=0)
std = x.std(dim=0)
x = (x - mean ) / std

model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.005)

for epoch in range(20000):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def evaluar():
    a = float(input("ingrese su edad: "))
    b = float(input("ingrese su años de experiencia: "))
    c = float(input("tiene credencial si = 1 , no = 0: "))
    d = float(input("tiene antecedentes si = 1, no = 0: "))

    entrada = torch.tensor([[a,b,c,d]])
    entrada = (entrada - mean ) / std 

    with torch.no_grad():
        logit = model(entrada)
        prob = torch.sigmoid(logit)

        if prob >= 0.8:
            print("Puede acceder al sistema")
            
        else:
            print("No puede acceder")
            exit()
    

while True:
    print("Bien venido al sistema")
    print("Si quiere entra coloque si y si quiere salir ponga no")
    Entrar = input("Quiere entrar:")

    if Entrar.lower() == "si":
        evaluar()
    elif Entrar.lower() == "no":
        break