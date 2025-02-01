import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def synthetic_data(mu: int, sigma: float, N: int) -> np.ndarray:
    """
    Generate x samples from a Guassian distribution N(mean=0, std=1)
    mu: mean, sigma: standard deviation, N = size of dataset
    """

    return np.random.normal(loc=mu, scale=sigma, size=N)

def plot_scattered_data(x, y, fig_x, fig_y):
    plt.figure(figsize=(fig_x,fig_y))
    plt.scatter(x, y)
    plt.title("Cubic function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_epoch_losses(x, y, fig_x, fig_y):
    plt.figure(figsize=(fig_x,fig_y))
    plt.plot(x, y)
    plt.title("MSE against Losses")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.show()