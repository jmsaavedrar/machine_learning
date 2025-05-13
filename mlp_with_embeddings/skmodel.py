import torch.nn as nn 

class skMLP(nn.Module):
    def __init__(self):
        super().__init__()        
        self.head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),            
            nn.ReLU(),

            nn.Linear(1024, 512),        
            nn.BatchNorm1d(512),
            nn.ReLU(),

            
            nn.Linear(512, 250),
        )

    def forward(self, x):
        logits = self.head(x)
        return logits