import torch
import numpy as np
from torch import nn
from utils import get_dataloaders, get_model, get_optimizer


def train(model_name, dataset_name, optimizer_name, loss_function, epochs=30):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(model_name).to(device)
    train_loader, validation_loader, test_loader = get_dataloaders(dataset_name)
    optimizer = get_optimizer(optimizer_name, model, lr=0.0001)

    for epoch in range(epochs):
        # Train phase
        model.train()
        for inputs, targets in train_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate output, loss and apply gradient changes
            output, output_predictions = model(inputs)
            loss = loss_function(output, torch.argmax(targets, dim=-1))
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()

        # Prepare trackers
        validation_loss_tracker = []
        validation_accuracy_tracker = []
        for inputs, targets in validation_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate output, loss and apply gradient changes
            output, output_predictions = model(inputs)
            loss = loss_function(output, torch.argmax(targets, dim=-1))

            # Track accuracy and loss values
            prediction_indices = torch.argmax(output_predictions, dim=-1)
            target_indices = torch.argmax(targets, dim=-1)
            num_correct = (prediction_indices == target_indices).sum().item()
            accuracy = num_correct / len(targets)
            validation_accuracy_tracker.append(accuracy)
            validation_loss_tracker.append(loss.item())

        # Test phase
        # Prepare trackers
        test_loss_tracker = []
        test_accuracy_tracker = []
        for inputs, targets in test_loader:
            # Prepare data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Calculate output, loss and apply gradient changes
            output, output_predictions = model(inputs)
            loss = loss_function(output, torch.argmax(targets, dim=-1))

            # Track accuracy and loss values
            prediction_indices = torch.argmax(output_predictions, dim=-1)
            target_indices = torch.argmax(targets, dim=-1)
            num_correct = (prediction_indices == target_indices).sum().item()
            accuracy = num_correct / len(targets)
            test_accuracy_tracker.append(accuracy)
            test_loss_tracker.append(loss.item())

        print("[Epoch %d/%d]\t[Validation loss: %.3f\tValidation accuracy: %.3f]\t[Test loss: %.3f\tTest accuracy: %.3f]" %
              (epoch + 1, epochs, np.mean(validation_loss_tracker), np.mean(validation_accuracy_tracker), np.mean(test_loss_tracker), np.mean(test_accuracy_tracker)))

    return model


if __name__ == "__main__":
    model = train(
        model_name="Disease",
        dataset_name="Disease",
        optimizer_name="Adam",
        loss_function=nn.CrossEntropyLoss(),
        epochs=50
    )