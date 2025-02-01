import torch as nn
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def train_model_bgd(x_train, target, model, learning_rate, n_epochs):
    """Batch Gradient Descent"""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model.forward(x_train)
        loss = loss_fn(y_hat, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
    return losses

def train_model_sgd(x_train, target, model, learning_rate, n_epochs):
    """Stochastic Gradient Descent"""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for i in range(len(x_train)):
            optimizer.zero_grad()

            x, y = x_train[i].unsqueeze(0), target[i].unsqueeze(0)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
        
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(x_train)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    return losses


def train_model_msgd(x_train, target, model, learning_rate, n_epochs, batch_size=2):
    """Mini-Batch Gradient Descent"""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses = []

    dataset = TensorDataset(x_train, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return losses


def train_model_gd(x_train, target, model, learning_rate, n_epochs, batch_size=None):
    """
    Generalized Gradient Descent (GD)

    Full batch GD: batch_size [int] = None
    Stochastic GD: batch_size [int] = 1
    Mini-batch GD: batch_size [int] = N (N can be any size)

    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    losses = []

    if batch_size is None:
        batch_size = len(x_train)

    dataset = TensorDataset(x_train, target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()

            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

    weight = {name: param.clone().detach() for name, param in model.named_parameters()}

    return losses, weight
