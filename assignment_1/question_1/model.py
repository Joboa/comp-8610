import torch

#### (d-1) Adaline: Batch Gradient descent ####
class AdalineBGD:
    def __init__(self, n_features, learning_rate):
        self.n_features = n_features
        self.weights = torch.zeros(n_features, 1, dtype=torch.float32)
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Perform a forward pass.
        x: Input tensor of shape [N, D]
        """
        x = x.to(self.weights.dtype)
        net_inputs = torch.mm(x, self.weights)  # Shape: [N, 1]
        return net_inputs.flatten()  # Flatten to [N]

    def backward(self, x, yhat, y):
        """
        Compute the gradients for batch gradient descent.
        x: Input tensor of shape [N, D]
        yhat: Predicted output tensor of shape [N]
        y: Actual output tensor of shape [N]
        """
        grad_loss_yhat = (yhat - y).view(-1, 1)  # Reshape to [N, 1]
        grad_yhat_weights = x  # Shape: [N, D]

        # Compute the gradient of weights: [D, 1]
        grad_loss_weights = torch.mm(grad_yhat_weights.t(), grad_loss_yhat) / y.size(0)
        return -self.learning_rate * grad_loss_weights

    def update_weights(self, grad_w):
        """
        Update the model's weights using the computed gradients.
        grad_w: Gradient tensor of shape [D, 1]
        """
        self.weights += grad_w

# Training
def loss(y, yhat):
    return torch.mean((yhat - y)**2)


def train(model, x, y, n_epochs, batch_size=1):
    cost = []

    for epoch in range(n_epochs):
        idx = torch.randperm(y.size(0), dtype=torch.long)  # 0,4,1,3
        mini_batches = torch.split(idx, batch_size)  # [1,2], [2,4] -> 2
        # print(mini_batches)

        for batch_idx in mini_batches:
            yhat = model.forward(x[batch_idx]).float().view(-1, 1)

            # print(yhat.shape)

            # compute gradients
            grad_w = model.backward(x[batch_idx], yhat, y[batch_idx])

            # update weight
            model.weights += grad_w

            batch_loss = loss(y[batch_idx], yhat)
            print(f"Mean squared error: {batch_loss:4f}")

        yhat = model.forward(x)
        current_loss = loss(y, yhat)
        print(f"Epoch: {epoch + 1}, MSE: {current_loss:4f}")
        cost.append(current_loss)

    return cost


#### (d-1) Adaline: Stochastic Gradient descent ####
class AdalineSGD():
    def __init__(self, n_features, learning_rate):
        self.n_features = n_features
        self.weights = torch.zeros(n_features, 1, dtype=float)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.to(self.weights.dtype)
        net_inputs = torch.mm(x, self.weights)
        activations = net_inputs
        return activations.flatten()

    def backward(self, x, yhat, y):
        grad_loss_yhat = (yhat - y)  # [batch_size, 1]
        # print(grad_loss_yhat.shape)
        grad_yhat_weights = x  # [batch_size, n_features]
        # print(grad_yhat_weights.shape)

        grad_loss_weights = torch.mm(
            grad_yhat_weights.t(), grad_loss_yhat / y.size(0))

        return (-1) * self.learning_rate * grad_loss_weights

def trainSGD(model, x, y, n_epochs):
    """
    Train the model using standard stochastic gradient descent (SGD).
    
    Args:
        model: Adaline model instance.
        x: Input tensor of shape [N, D].
        y: Target tensor of shape [N, 1].
        n_epochs: Number of training epochs.
    
    Returns:
        cost: List of mean squared errors over epochs.
    """
    cost = []  
    for epoch in range(n_epochs):
        # Shuffle the dataset
        idx = torch.randperm(y.size(0), dtype=torch.long)

        for i in idx:
            # Select a single data point
            xi = x[i].view(1, -1).float()  # Shape [1, D]
            yi = y[i].view(1, -1).float()  # Shape [1, 1]

            # Forward pass
            yhat = model.forward(xi).view(-1, 1).float()  # Shape [1, 1]

            # Compute gradients
            grad_w = model.backward(xi, yhat, yi)

            # Update weights
            model.weights += grad_w

        # Compute loss for the entire dataset after the epoch
        yhat = model.forward(x)
        current_loss = loss(y, yhat)
        cost.append(current_loss.item())
        print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {current_loss:.4f}")

    return cost



def trainBGD(model, x, y, n_epochs):
    cost = []

    for epoch in range(n_epochs):

        # Predictions for the entire dataset
        yhat = model.forward(x).view(-1, 1)

        # Compute gradients
        grad_w = model.backward(x, yhat, y)

        # Update weights
        model.update_weights(grad_w)

        # Compute and store the loss
        current_loss = loss(y, yhat)
        cost.append(current_loss.item())
        print(f"Epoch {epoch + 1}/{n_epochs}, MSE: {current_loss:.4f}")

    return cost
