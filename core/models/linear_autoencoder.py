__author__ = 'Youshaa Murhij'

import os
import torch
from torch import nn

class LAE(nn.Module):
    def __init__(self):
        super(LAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(23, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True), 
            nn.Linear(32, 16))
        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True), 
            nn.Linear(32, 32)
            )   # nn.Tanh()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.permute(x, (0, 3, 1, 2))

        return x

def main():
    model = LAE().cuda(0)
    model.eval()
    x = torch.rand((1, 23, 256, 256)).cuda(0)
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    main()