import torch 
import torch as nn 

class CNN(nn.Module):
    def __int__(self):
        super().__init__()

        self.con1 = nn.conv2d(
            in_channels  = 3,
            out_channels = 16,
            karnel_size = 3,
            stride = 1,
            padding = 1

        )
    def forward(self, x):
        x = self.conv1(x)
        return x
    
def build_model():
    model = CNN
    return model 
