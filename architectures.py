import torch.nn as nn
from torch import softmax


class DiseaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modules = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        out = self.modules(x)
        out_probs = softmax(out)
        return out, out_probs
