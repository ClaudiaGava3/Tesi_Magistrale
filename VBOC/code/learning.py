import numpy as np
import torch
import torch.nn as nn
torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
from torch.nn.functional import mse_loss


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network. """
    def __init__(self, nx, nh, num_hl, activation=nn.ReLU(), ub=None):
        super().__init__()
        net = [nn.Linear(nx, nh), activation]
        for _ in range(num_hl):
            net.append(nn.Linear(nh, nh))
            net.append(activation)
        net.append(nn.Linear(nh, 1))
        # net.append(activation)
        self.linear_stack = nn.Sequential(*net)
        self.ub = ub if ub is not None else 1
        self.initialize_weights()

    def forward(self, x):
        out = self.linear_stack(x) * self.ub 
        return out #(out + 1) * self.ub / 2
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    

class RegressionNN:
    """ Class that compute training and test of a neural network. """
    def __init__(self, params, model, loss_fn, optimizer):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.beta = params.beta
        self.batch_size = params.batch_size

    def training(self, x_train_val, y_train_val, split, epochs):
        """ Training of the neural network. """

        # Split the data into training and validation
        x_train, x_val = x_train_val[:split], x_train_val[split:]
        y_train, y_val = y_train_val[:split], y_train_val[split:]

        loss_evol_train = np.zeros(epochs)
        loss_evol_val = np.zeros(epochs)
        loss_lp = 1

        n = len(x_train)
        for e in range(epochs):
            self.model.train()
            # Shuffle the data
            idx = torch.randperm(n)
            x_perm, y_perm = x_train[idx], y_train[idx]
            # Split in batches 
            x_batches = torch.split(x_perm, self.batch_size)
            y_batches = torch.split(y_perm, self.batch_size)
            for x, y in zip(x_batches, y_batches):
                # Forward pass
                y_pred = self.model(x)
                # Compute the loss
                loss = self.loss_fn(y_pred, y)
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_lp = self.beta * loss_lp + (1 - self.beta) * loss.item()

            loss_evol_train[e] = loss_lp
            # Validation
            loss_val = self.validation(x_val, y_val)
            loss_evol_val[e] = loss_val

            if e % int(epochs / 10) == 0:
                print(f'Epoch [{e}/{epochs}], Train Loss: {loss_lp:.5f}, Val Loss: {loss_val:.5f}')

        return loss_evol_train, loss_evol_val


    def validation(self, x_val, y_val):
        """ Compute the loss wrt to validation data. """
        x_batches = torch.split(x_val, self.batch_size)
        y_batches = torch.split(y_val, self.batch_size)
        self.model.eval()
        tot_loss = 0
        y_out = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred = self.model(x)
                y_out.append(y_pred)
                loss = self.loss_fn(y_pred, y)
                tot_loss += loss.item()
            y_out = torch.cat(y_out, dim=0)
        return tot_loss / len(x_batches)
    
    def testing(self, x_test, y_test):
        """ Compute the RMSE wrt to training or test data. """
        x_batches = torch.split(x_test, self.batch_size)
        y_batches = torch.split(y_test, self.batch_size)
        self.model.eval()
        y_pred = []
        with torch.no_grad():
            for x, y in zip(x_batches, y_batches):
                y_pred.append(self.model(x))
            y_pred = torch.cat(y_pred, dim=0)
            rmse = torch.sqrt(mse_loss(y_pred, y_test)).item()
            rel_err = (y_pred - y_test) / y_test  # torch.maximum(y_test, torch.Tensor([1.]).to(self.device))
        return rmse, rel_err    
