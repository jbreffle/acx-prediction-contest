"""Torch models for the ACX project."""

# Imports
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# TODO Switch to using a params dictionary for all hyperparameters
# TODO: Switch to following the methods of
# <https://github.com/pytorch/examples/blob/main/mnist/main.py#L134>
# And then consider the methods of
# <https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/evaluate.py>


# Functions
def get_default_params():
    """Function to initilize default hyperparameters"""
    params = type("Params", (), {})()
    # Data
    params.batch_size = 100
    params.train_size = 0.8
    # Model
    params.hidden_layer_sizes = [100, 50, 10]
    # Training
    params.log_interval = 10  # How often to log results in epochs
    params.lr = 0.00001  # Learning rate
    params.weight_decay = 0.001
    params.n_epochs = 100
    params.gamma = 0.8  # Multiplicative factor of learning rate decay
    # Device
    params.use_cuda = True
    params.device = torch.device(
        "cuda" if torch.cuda.is_available() and params.use_cuda else "cpu"
    )
    # Not implemented yet
    params.dry_run = None  # not implemented yet
    params.save_model = None  # not implemented yet
    params.test_batch_size = None  # not implemented yet
    return params


def train_and_evaluate(
    model, train_loader, test_loader, scheduler, optimizer, loss_function, params
):
    """..."""
    train_loss = []
    test_loss = []
    start_time = time.time()
    for epoch_idx in range(1, params.n_epochs + 1):
        step_train_loss = train(
            train_loader,
            model,
            optimizer,
            loss_function,
            params,
        )
        step_test_loss = test(model, params.device, test_loader, loss_function)
        scheduler.step()
        # Append losses
        train_loss.append(step_train_loss)
        test_loss.append(step_test_loss)
        # print loss and runtime
        if (epoch_idx + 1) % params.log_interval == 0:
            txt = "Epoch [{:4d}/{:4d}], Train loss: {:07.4f}, Test loss: {:07.4f}, Run Time: {:05.2f}"
            print(
                txt.format(
                    epoch_idx + 1,
                    params.n_epochs,
                    step_train_loss,
                    step_test_loss,
                    time.time() - start_time,
                )
            )
    return train_loss, test_loss


def train(
    train_loader,
    model,
    optimizer,
    loss_function,
    params,
):
    model.train()
    # Train set
    step_train_loss = 0
    for X, y in train_loader:
        X, y = X.to(params.device), y.to(params.device)
        preds = model(X)
        loss = loss_function(preds, y)
        optimizer.zero_grad()
        loss.backward()
        step_train_loss += loss.item()
        optimizer.step()
    step_train_loss = step_train_loss / len(train_loader)

    return step_train_loss


def test(model, device, test_loader, loss_function, silent=True):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
    # test_loss /= len(test_loader.dataset)

    if not silent:
        print(f"Test loss: {test_loss:.4f}")
    return test_loss


# Models
class Net(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.fc4 = nn.Linear(hidden_layer_sizes[2], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.01)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.01)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLPCondensedVariable(nn.Module):
    """
    Multi-layer perceptron for non-linear regression.
    """

    def __init__(self, input_n, n_hidden_units, output_n, nLayers):
        super().__init__()
        # Input layer
        layers = [nn.Linear(input_n, n_hidden_units), nn.ReLU()]
        # Hidden layers
        for _ in range(nLayers):
            layers.extend([nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU()])
        # Output layer
        layers.append(nn.Linear(n_hidden_units, output_n))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPcondensed(nn.Module):
    """
    Multi-layer perceptron for non-linear regression.
    """

    def __init__(self, input_n, n_hidden_units, output_n):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_n, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, output_n),
        )

    def forward(self, x):
        return self.layers(x)


class SLPnet(nn.Module):
    """
    Single layer perceptron for non-linear regression.
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=3, out_features=9)
        self.hidden_1 = nn.Linear(in_features=9, out_features=9)
        self.output = nn.Linear(in_features=9, out_features=1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        return self.output(x)