import torch 
import torch.nn as nn 
import torch.optim as optim 

# datos de entrenamiento 
x = torch.tensor([
    [120,3,1,1],
    [90,2,0,0],
    [350,6,3,1],
    [100,2,1,0],
    [200,4,2,1],
    [80,1,0,0],
    [400,8,4,1],
    [150,3,1,0],
    [25,1,0,0],
    [500,10,3,1],
    [60,2,0,1],
    [170,3,2,0],
    [300,5,2,1],
    [45,1,0,0],
    [35,2,0,1],
    [75,2,1,0],
    [80,2,1,1],
    [220,4,2,0],
    [170,3,1,1],
    [400,7,3,0],
], dtype=torch.float32)

y = torch.tensor([
    [1],[0],[1],[0],[1],
    [0],[1],[0],[0],[1],
    [0],[1],[1],[0],[0],
    [0],[0],[1],[1],[1]
], dtype=torch.float32)

#normalizacion z-score

mean = x.mean(dim=0)
std = x.std(dim=0)
x = (x-mean)/ std 

#modelo sin sigmoid 

model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16,8),
    nn.ReLU(),
    nn.Linear(8,1)
)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=00.5)


#entrenamineto

for epoch in range(15000):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 5000 ==0:
        print(epoch, loss.item())


def evaluar():
     a = float(input("Valor 1: "))
     b = float(input("Valor 2: "))
     c = float(input("valor 3: "))
     d = float(input("Valor 4: "))

     entrada = torch.tensor([[a,b,c,d]])
     entrada = (entrada - mean)/ std #misma normalizacion 

     with torch.no_grad():
         logit = model(entrada)
         prob = torch.sigmoid(logit)

         if prob >= 0.8:
             print("Resultado: es seguro", float(prob))
         elif prob >= 0.6:
             print("Resultado: posiblemente seguro",float(prob))
         else:
             print("Resultado: poco seguro",float(prob))

#loop tipo app

while True:
    print("op = 1 evaluar")
    print("op = 2 salir ")
    op = input("Opcion: ")

    if op == "1":
        evaluar()
    elif op == "2":
        break
    