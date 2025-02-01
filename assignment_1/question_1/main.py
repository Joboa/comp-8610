import numpy as np
import torch
from config import CONFIG
from utils import synthetic_data, set_seed, plot_epoch_losses
from model import NNModel
from train import train_model_gd, train_model_bgd, train_model_sgd

set_seed(CONFIG["seed"])
# ### (a) ####
x: np.ndarray = synthetic_data(mu=0, sigma=1, N=5000)

# ### (b) ####
# sigma = sqrt(variance) , var=0.25
eps: np.ndarray = synthetic_data(mu=0, sigma=0.5, N=5000)

# ### (c) ####
y: np.ndarray = 1 - (0.5*x) + (2*x**2) - (0.3*x**3) + eps

# plot_data(x, y, 10, 8)
# Torch: x, y
X = torch.from_numpy(x).view(-1, 1).float()
Y = torch.from_numpy(y).view(-1, 1).float()

model = NNModel("sigmoid")

# print(model.activation)
losses1 = train_model_sgd(x_train=X,
                         target=Y,
                         model=model,
                         learning_rate=0.001,
                         n_epochs=5)

# losses2 = train_model_sgd(x_train=X,
#                          target=Y,
#                          model=model,
#                          learning_rate=0.001,
#                          n_epochs=10)
# losses3, weight3 = train_model_gd(x_train=X,
#                          target=Y,
#                          model=model,
#                          learning_rate=0.01,
#                          n_epochs=100,
#                          batch_size=2)

# print(f"weight1: {weight1} weight2: {weight2}")

plot_epoch_losses(range(len(losses1)), losses1, 10, 8)
# import matplotlib.pyplot as plt

# plt.plot(range(len(losses1)), losses1,range(len(losses2)), losses2)
# plt.show()