from custom_datasets import get_xor_dataloaders, get_disease_dataloaders, get_glass_dataloaders, get_letter_dataloaders
from architectures import SimpleModel
from torch import optim
from torch import nn


def get_dataloaders(dataset_name):
    if dataset_name == "Disease":
        return get_disease_dataloaders()
    elif dataset_name == "XOR":
        return get_xor_dataloaders()
    elif dataset_name == "Glass":
        return get_glass_dataloaders()
    elif dataset_name == "Letter":
        return get_letter_dataloaders()
    else:
        raise Exception("Choose one of the following datasets:\nDisease\nXOR\nGlass\nLetter")


def get_model(model_name):
    if model_name == "Disease":
        return SimpleModel(input_length=10, output_length=6)
    elif model_name == "XOR":
        return SimpleModel(input_length=2, output_length=2)
    elif model_name == "Glass":
        return SimpleModel(input_length=9, output_length=7)
    elif model_name == "Letter":
        return SimpleModel(input_length=16, output_length=26)
    else:
        raise Exception("Choose one of the following models:\nDisease\nXOR\nGlass\nLetter")


def get_optimizer(optimizer_name, model: nn.Module, lr: float, decay=0, momentum=0):
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    elif optimizer_name == "Adagrad":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
    else:
        raise Exception("Choose one of the following optimizers:\nAdam\nSGD\nAdagrad\nRMSprop")

