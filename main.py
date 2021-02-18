import torch
import numpy as np
from utils import get_dataloaders, get_model, get_optimizer
from architectures import DiseaseModel


def train(model_name, dataset_name, optimizer_name, loss_function):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(model_name)
    train_loader, validation_loader, test_loader = get_dataloaders(dataset_name)
    optimizer = get_optimizer(optimizer_name, model, lr=0.01)

    for inputs, targets in train_loader:
        model.train()
        model.zero_grad()

        output = model(input)
        loss = loss_function.backwards()

