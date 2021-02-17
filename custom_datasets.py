from torch.utils.data import DataLoader, Dataset, random_split
import torch
import numpy as np


def get_disease_dataloaders():
    dataset = CustomDataset("./Datasets/DiseaseDataset.mld", 6)
    trainset, validationset, testset = random_split(dataset, [15000, 2500, 2500])
    return DataLoader(trainset, batch_size=64), DataLoader(validationset, batch_size=64), DataLoader(testset, batch_size=64)


def get_glass_dataloaders():
    dataset = CustomDataset("./Datasets/Glass.mld", 7)
    trainset, validationset, testset = random_split(dataset, [150, 30, 34])
    return DataLoader(trainset, batch_size=64), DataLoader(validationset, batch_size=64), DataLoader(testset, batch_size=64)


def get_letter_dataloaders():
    dataset = CustomDataset("./Datasets/LetterDataset.mld", 26)
    trainset, validationset, testset = random_split(dataset, [15000, 2500, 2500])
    return DataLoader(trainset, batch_size=64), DataLoader(validationset, batch_size=64), DataLoader(testset, batch_size=64)


def get_xor_dataloaders():
    dataset = CustomDataset("./Datasets/XorDataset.mld", 2)
    trainset, validationset, testset = random_split(dataset, [15000, 2500, 2500])
    return DataLoader(trainset, batch_size=64), DataLoader(validationset, batch_size=64), DataLoader(testset, batch_size=64)


class CustomDataset(Dataset):
    def __init__(self, dir, target_length):
        self.datapoints = []
        self.targets = []
        self.target_labels = []

        # Read dataset file and split datapoints
        all_data = open(dir, "r").read()
        split_data = all_data.splitlines()

        for data_entry in split_data:
            split_entry = data_entry.split(';')

            entry_input = torch.Tensor([float(x) for x in split_entry[:-1]])
            entry_label = split_entry[-1]
            self.datapoints.append(entry_input)

            if entry_label not in self.target_labels:
                self.target_labels.append(entry_label)

            entry_target = torch.zeros(target_length)
            entry_target[self.target_labels.index(entry_label)] = 1.

            self.targets.append(entry_target)

    def target_to_label(self, target):
        index = int(torch.argmax(target))
        return self.target_labels[index]

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, item):
        return self.datapoints[item], self.targets[item]
