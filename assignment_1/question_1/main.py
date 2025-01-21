import numpy as np
import torch
import matplotlib.pyplot as plt
from config import CONFIG

from model import Adaline, AdalineBGD, trainBGD, trainSGD, loss


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def synthetic_data(mu: int, sigma: float, N: int) -> np.ndarray:
    """
    Generate x samples from a Guassian distribution N(mean=0, std=1)
    mu: mean, sigma: standard deviation, N = size of dataset
    """

    return np.random.normal(loc=mu, scale=sigma, size=N)

def plot_data(x, y, fig_x, fig_y):
    plt.figure(figsize=(fig_x,fig_y))
    plt.scatter(x, y)
    plt.title("Cubic function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    # ### (a) ####
    x: np.ndarray = synthetic_data(mu=0, sigma=1, N=10)

    # ### (b) ####
    eps: np.ndarray = synthetic_data(mu=0, sigma=0.5, N=10) # sigma = sqrt(variance) , var=0.25

    y: np.ndarray = 1 - (0.5*x )+ (2*x**2) - (0.3*x**3) + eps

    # plot_data(x, y, 10,8)

    # Torch: x, y
    X = torch.from_numpy(x).view(-1,1).float()
    Y = torch.from_numpy(y).view(-1,1).float()
    # print(X.shape, Y.shape)
    # print(X.size(), Y.size)
    # model = AdalineBGD(n_features=X.size(1), learning_rate=0.01)
    # cost = trainBGD(model, X, Y, n_epochs=20)

    model = Adaline(n_features=X.size(1), learning_rate=0.01)
    cost = trainSGD(model, X, Y, n_epochs=20)

    plt.plot(range(len(cost)), cost)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.show()