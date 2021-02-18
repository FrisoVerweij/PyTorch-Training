import torch.nn as nn
from torch import softmax


class SimpleModel(nn.Module):
    def __init__(self, input_length, output_length):
        super(SimpleModel, self).__init__()
        self.all_modules = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_length)
        )

    def forward(self, x):
        out = self.all_modules(x)
        out_probs = softmax(out, dim=-1)
        return out, out_probs
