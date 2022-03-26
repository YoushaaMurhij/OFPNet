
import torch
import torch.nn as nn
import torchvision


class Trasformer():
    
    

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        
        self.transformer_model = nn.Transformer(d_model=256)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
